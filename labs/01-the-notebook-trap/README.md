# Lab 01: The notebook trap

> **Mission:** train the kind of fraud model a lot of teams start with, then work out why its great score is a lie.

**Time:** 45–60 min · **You'll touch:** `src/fraudlab/splits.py`, `labs/01-the-notebook-trap/answers.md` · **Prereqs:** Lab 00

## Why this matters

The most expensive bugs in ML don't crash. They produce **beautiful offline metrics**
that collapse in production. The usual culprits are *leakage* (the model sees
information it won't have at prediction time) and *evaluation that doesn't match
reality* (testing on the past while training on the future). Everything after this
lab is tooling that makes these mistakes structurally hard to make.

## Concepts
* **Random vs temporal split.** In production you always predict the *future* from the
  *past*. A random split lets the model peek across time.
* **Target leakage.** A feature computed from the label itself or from the future, like
  "fraud rate of this merchant, computed over the whole dataset".
* **Training/serving skew.** Even without leakage, a feature you can't compute *at request
  time, in the same way* is a feature you can't use.

## Steps

### Step 1: Run the baseline
```bash
uv run fraudlab baseline --split random
```
Read `src/fraudlab/baseline.py` while it trains (~20 s). Write down `roc_auc`, `pr_auc`
and `recall_at_alert`. (`src/fraudlab/metrics.py` explains each, and why accuracy is useless here.)

✅ You have three numbers written down. Before the next step, **predict**: will a
temporal split make them better, worse or the same?

### Step 2: Implement `temporal_split`
Open `src/fraudlab/splits.py` and implement `temporal_split` according to its docstring.
Details that matter:
* the cutoff is midnight **UTC** on the cutoff date;
* timestamps might be timezone-naive, so treat those as UTC;
* keep the original row order.

✅ `uv run pytest checks/test_lab01_baseline.py -k temporal` passes.

### Step 3: Run with a temporal split
```bash
uv run fraudlab baseline --split temporal
```
Compare with Step 1. Surprised? The split was *a* problem but not *the* problem: the
features are still computed with knowledge of the future.

### Step 4: Leakage hunt
Read `make_naive_features` line by line. For each feature, ask: *at 14:05 on Jan 10th,
when this transaction arrives, could I compute exactly this number?* Then measure:
```bash
uv run fraudlab baseline --split temporal --drop cust_chargeback_count
uv run fraudlab baseline --split temporal --drop cust_chargeback_count --drop merch_fraud_rate
```
✅ You can explain why the metrics drop when you remove those features, and why that drop
is *good news* about your evaluation.

### Step 5: Write it down
```bash
cp labs/01-the-notebook-trap/answers.template.md labs/01-the-notebook-trap/answers.md
```
Answer the questions in your own words (replace each `_your answer_`). Ask Claude to
review your answers. That's part of the lab.

### Step 6: Check
```bash
make check LAB=01
```

## Break it (optional)
* Make the leak worse: add a feature `cust_fraud_ever = cust_chargeback_count > 0`.
  How high can you push `pr_auc`?
* Shrink the training set to January only. Does the random split hide the change?

## Production notes
* Labels in fraud arrive **late**: chargebacks take days to weeks. A feature like
  "merchant fraud rate" may only use *matured* labels. You'll implement this in Lab 04.
* The honest way to compute "as-of" features is a **point-in-time join**. Doing it by
  hand in pandas is error-prone. That's the main reason feature stores exist (Labs 04–05).
* `baseline.py` has more production problems than leakage: a pickle with no version, no
  record of params or data, hard-coded paths, no tests. Labs 02–03 fix the record-keeping.

## Quiz
1. Why can a random split overstate performance even with *no* leaky features?
2. `cust_avg_amount` is computed over all of a customer's transactions. Name two ways that leaks.
3. What is training/serving skew, and how does it differ from leakage?
4. The fraud team can review 1% of transactions per day. Which metric best reflects their reality?
