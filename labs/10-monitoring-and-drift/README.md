# Lab 10: Monitoring, drift and automated retraining

> **Mission:** notice when the world changes under your model, and close the loop: detect → retrain → fair comparison → promote.

**Time:** 90 min (plus a ~45 min backfill you can let run) · **You'll touch:** `src/fraudlab/monitoring/drift.py`, `dags/fraud_pipeline.py` · **Prereqs:** Lab 08

## Why this matters

On **2026-04-15** fraudsters in this simulation switch from big-ticket electronics to
"card testing": bursts of tiny charges at gift-card and gaming merchants. Your model has
never seen this. Nothing crashes, no error is logged. The model just quietly lets fraud
through. Monitoring is how you find out before the chargebacks do.

## Concepts
* **Data drift**: P(X) changes (inputs look different). Measurable *without labels*,
  for example with the **Population Stability Index**:
  `PSI = Σ (cur% − ref%) · ln(cur% / ref%)` over bins of the reference distribution.
  Rule of thumb: < 0.1 stable, 0.1–0.2 watch, > 0.2 investigate.
* **Concept drift**: P(y | X) changes (same-looking inputs, different meaning). Only
  **labels** reveal it: track recall/precision on recent labelled predictions.
* **Reference window**: what "normal" looks like, here the last 7 days of the
  champion's training window.
* **Closing the loop**: drift → retrain → champion re-scored on the same validation data
  → promote only if better. Never auto-promote without a gate.

## Steps

### Step 1: Implement `psi`, `performance`, `detect_drift`
All three are pure functions in `src/fraudlab/monitoring/drift.py`, with the spec in
their docstrings. Fast feedback:
```bash
uv run pytest checks/test_lab10_monitoring.py -k "psi or performance or detect"
```
Read `run_drift_report` (provided): reference vs current, 7-day rolling performance,
JSON report, and an MLflow run per champion version with one *step* per day.

### Step 2: Try it on one day
Pick a day the pipeline already scored (Lab 08):
```bash
uv run fraudlab monitor --day 2026-04-08
cat data/reports/drift/2026-04-08.json
```

### Step 3: Put it in the DAG
Add a `monitor` task (`fraudlab monitor --day {{ ds }}`) between `score` and
`decide_retrain`. Change `decide_retrain` to return `"train"` when monitoring printed
`DRIFT=true` (pull the `monitor` XCom) **or** it's Sunday.

### Step 4: Live through the attack
```bash
make airflow CMD="backfill create --dag-id fraud_pipeline --from-date 2026-04-09 --to-date 2026-04-30 --max-active-runs 1"
```
That's 22 runs, about 2–3 minutes each. While it runs, open MLflow → experiment
**fraud-monitoring** → the `monitor-vN` run → **Model metrics**, and watch `recall_7d`
and `psi_*` grow day by day.

✅ After 2026-04-15, `performance_drift` becomes true within a few days, `train` runs on
drift days, and newer versions get promoted.

### Step 5: Investigate like an on-call engineer
* Did **any** PSI cross 0.2? Why not, given that fraud changed completely? (Fraud is ~1% of
  traffic. Even a total change in fraud barely moves the overall input distribution.)
* Plot recall over time (`data/reports/drift/*.json` → `perf_7d.recall`). When was drift
  detected, and why the delay?
* Which model versions were promoted after 04-15, and by how much did each beat the
  champion on the same data (`pr_auc` vs `champion_pr_auc`)? Look for a version that
  **beat the champion and was still rejected**. Which rule stopped it, and why is that
  the right call?
* Recall recovers only **partially**. Look at which features could express "4 tiny
  charges within an hour". None of them can: every feature is a *daily* batch aggregate.
  Retraining can't fix missing signal. That's your capstone.

### Step 6: Check
```bash
make check LAB=10
```

## Break it (optional)
* Set `RECALL_DROP_ALERT = 0.9`: how many false alarms before 04-15? What does alert
  fatigue cost a team?
* Inject *data* drift: multiply `amount` by 1.3 in `generate.py` for one day (a currency
  bug), score it, and run the monitor. Which PSI fires?

## Production notes
* Tools: Evidently, NannyML, WhyLabs, Arize, Fiddler, or SQL plus dashboards. The hard part
  isn't PSI, it's choosing references, thresholds and owners.
* **Labels are late** in most domains. Use proxies meanwhile (score distribution, flag
  rate, manual review outcomes) and estimate performance (NannyML's CBPE).
* Monitor **segments** (merchant category, channel, new customers). An attack on 1% of
  traffic disappears in global averages, exactly as you just saw.
* Automated retraining needs guardrails: data validation, quality gates, fair
  comparison, a canary, and a human-visible audit trail of *why* a model was promoted.

## Quiz
1. Data drift vs concept drift: which one happened on 04-15, and which monitor caught it?
2. Why compare to the champion's *training* window rather than "yesterday"?
3. Why a 7-day rolling window and a minimum number of frauds?
4. Retraining after the attack helped only partly. Give two reasons.
