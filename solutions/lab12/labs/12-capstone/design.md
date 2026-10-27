# Design: real-time velocity features (fraud_model_v2)

## 1. Problem and success metric
Since 2026-04-15 card testers run bursts of 3–6 charges under $15 within one hour. The v1
champion's 7-day recall dropped from ~0.65 to <0.1, and retraining on post-attack data only
reached pr_auc ~0.3. Success: on the same post-attack validation week, v2 beats a
v1 retrained on the same window by ≥ 0.1 pr_auc, with p99 API latency still under 50 ms.

## 2. Feature definitions
After each transaction at time t, for its customer:
* `cust_txn_count_1h`: number of the customer's transactions in (t − 1h, t], including this one.
* `cust_small_txn_count_1h`: same, counting only amount < $20.
The snapshot becomes visible at t + 1µs, so a transaction sees the state left by the
customer's *previous* transaction, never itself. An unknown customer, or a snapshot older
than the 1h TTL, gives missing → the model treats it as "no recent activity".

## 3. Offline / online consistency
Offline: `velocity_snapshots` uses a per-customer pandas time-rolling window (closed right)
over raw transactions, one row per transaction, backfilled per day. Online: the API keeps
each customer's last-hour transactions in a Redis sorted set (scores = timestamps), trims
everything ≤ t − 1h, and pushes the identical snapshot through the Feast PushSource after
scoring. Same window, same edges, same "after the transaction" visibility, so both paths
agree. After a Redis flush or restart, counts restart from 0 for up to one hour: an
acceptable, bounded under-count. (A stream processor with replay would remove it.)

## 4. Feast design
`customer_velocity_source` (FileSource over daily snapshot files) → `customer_velocity_push`
(PushSource) → `customer_velocity` feature view, ttl 1h (an older snapshot carries no
information about the last hour). New service `fraud_model_v2` = v1 views + velocity.
`fraud_model_v1` stays untouched: the current champion and its monitoring depend on it.

## 5. Validation and rollout
Train v1 and v2 on the same window ending 2026-04-30 (the latest data, so the champion
never trained on our validation week). `champion_metrics_on` re-scores the
champion with its own features on the same validation transactions. Don't promote if v2
loses on pr_auc, fails the quality gate, or raises the alert rate beyond budget. Rollout:
shadow-score v2 for a few days, then move `@champion`. Rollback = move the alias back and
`/admin/reload`. The velocity pushes keep running regardless, so the feature stays warm.

## 6. Risks
* Latency: one extra Redis round trip and push per request. Pipeline the Redis calls; measure p99.
* Hot customers / memory: sorted sets are trimmed to one hour and expire. Cap members per key.
* Clock skew between API replicas: use the event time from the payment system, not local clocks.
* Abuse: attackers spacing charges > 1h apart. Add a 24h window feature later and monitor recall by segment.
