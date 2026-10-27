"""Real-time velocity features.  >>> Lab 12 (capstone) <<<

Card testing = a script fires several tiny charges within minutes. Daily batch features
can't see that: by the time tonight's batch runs, the card is drained. We need a feature
that is updated *on every transaction*, with an offline history that matches it exactly.

Definition (identical offline and online - that is the whole point):

    After each transaction at time t, the customer's velocity state is
        cust_txn_count_1h       = # of their transactions in (t - 1h, t]   (incl. this one)
        cust_small_txn_count_1h = same, counting only amount < SMALL_AMOUNT
    and it becomes visible at t + 1 microsecond.

So when a new transaction arrives, it sees the state left by the customer's PREVIOUS
transaction (never itself). Offline we write one snapshot row per transaction; online the
serving app pushes the same snapshot to Feast right after scoring (PushSource).
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from fraudlab.config import SIM_START, get_settings
from fraudlab.data.generate import daterange, load_transactions

VELOCITY_WINDOW = pd.Timedelta(hours=1)
SMALL_AMOUNT = 20.0
SNAPSHOT_DELAY = pd.Timedelta(microseconds=1)
VELOCITY_FEATURES = ["cust_txn_count_1h", "cust_small_txn_count_1h"]


def velocity_snapshots(txns: pd.DataFrame) -> pd.DataFrame:
    """One row per transaction: customer_id, event_timestamp (= t + 1us), the two counts."""
    df = txns[["customer_id", "event_timestamp", "amount"]].sort_values(
        ["customer_id", "event_timestamp"], kind="stable"
    )
    df = df.assign(one=1, small=(df["amount"] < SMALL_AMOUNT).astype("int64"))
    rolled = (
        df.set_index("event_timestamp")
        .groupby("customer_id")[["one", "small"]]
        .rolling(VELOCITY_WINDOW)  # time-based, closed="right": (t - 1h, t]
        .sum()
        .reset_index()
    )
    return pd.DataFrame({
        "customer_id": rolled["customer_id"].astype("int64"),
        # microsecond precision like every other table (DuckDB can't cast TIMESTAMPTZ_NS)
        "event_timestamp": (rolled["event_timestamp"] + SNAPSHOT_DELAY).astype("datetime64[us, UTC]"),
        "cust_txn_count_1h": rolled["one"].astype("int64"),
        "cust_small_txn_count_1h": rolled["small"].astype("int64"),
    })


def build_velocity_tables(start: date, end: date) -> int:
    """Backfill offline snapshots, one parquet file per day (idempotent)."""
    out_dir = get_settings().features_dir / "customer_velocity"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for day in daterange(start, end):
        # the previous day is needed for windows that start before midnight
        txns = load_transactions(max(SIM_START, day - timedelta(days=1)), day)
        snaps = velocity_snapshots(txns)
        snaps = snaps[snaps["event_timestamp"].dt.date == day]
        path = out_dir / f"{day:%Y-%m-%d}.parquet"
        tmp = path.with_suffix(".tmp")
        snaps.to_parquet(tmp, index=False)
        tmp.replace(path)
        n += 1
    return n


class VelocityTracker:
    """Online side: keep each customer's last-hour transactions in Redis (a sorted set per
    customer, in a separate Redis DB from Feast's) and push the new snapshot to Feast."""

    PUSH_SOURCE = "customer_velocity_push"

    def __init__(self, store, redis_client):
        self.store, self.redis = store, redis_client

    def record(self, customer_id: int, amount: float, ts: pd.Timestamp) -> dict:
        from feast.data_source import PushMode

        key, now = f"velocity:{customer_id}", ts.timestamp()
        member = f"{now:.6f}:{amount:.2f}:{id(ts)}"
        pipe = self.redis.pipeline()
        pipe.zadd(key, {member: now})
        pipe.zremrangebyscore(key, "-inf", now - VELOCITY_WINDOW.total_seconds())
        pipe.zrange(key, 0, -1)
        pipe.expire(key, int(VELOCITY_WINDOW.total_seconds()) + 60)
        _, _, members, _ = pipe.execute()
        amounts = [float(m.decode().split(":")[1]) for m in members]
        snapshot = {
            "customer_id": int(customer_id),
            "event_timestamp": ts + SNAPSHOT_DELAY,
            "cust_txn_count_1h": len(amounts),
            "cust_small_txn_count_1h": sum(a < SMALL_AMOUNT for a in amounts),
        }
        self.store.push(self.PUSH_SOURCE, pd.DataFrame([snapshot]), to=PushMode.ONLINE)
        return snapshot
