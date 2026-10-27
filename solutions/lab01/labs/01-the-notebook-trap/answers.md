# Lab 01: reference answers

1. **Which features leak, and how?**
   * `cust_chargeback_count`: sums `is_fraud` over *all* the customer's rows, including the
     current transaction's own label and future ones. A fraudulent transaction literally
     carries its label inside a feature.
   * `merch_fraud_rate`: the same thing per merchant: the label of this transaction and of
     future transactions feed the rate. Small merchants leak the most (one fraud among 5 rows = 0.2).
   * `cust_avg_amount`, `cust_txn_count`, `merch_txn_count`, `amount_to_cust_avg`: computed
     over the whole dataset, so they include *future* transactions (and the big fraudulent
     amounts themselves). Less blatant, still not available at prediction time.

2. **How would they have to be computed at serving time?**
   As-of the transaction time, from past data only: `cust_avg_amount` over a trailing
   window ending *before* the transaction (e.g. the last 30 days up to last midnight), and
   `merch_fraud_rate` from transactions whose labels are already known (chargebacks
   mature after days, so the window must end a week or so in the past), smoothed toward a
   prior because small merchants give noisy rates. Training must use exactly that same
   as-of logic, which is what a point-in-time join does.

3. **Random vs temporal vs temporal-without-leaks.**
   Random split: how well the model interpolates within a period it has partly seen (an
   optimistic upper bound). Temporal split: how it does on the future, but still with leaky
   features, so still optimistic. Temporal without leaky features: the closest to what
   production will see. Report that one, or better, the Lab 06 number computed from
   point-in-time features.

4. **Other production problems in `baseline.py`.**
   Pickle with no version, metadata or signature (and pickle can execute code on load); no
   record of params, data window, code version or metrics; hard-coded dates and paths; the
   threshold isn't part of the model; feature code lives in the training script, so serving
   would have to re-implement it (skew); no tests; no data validation; you can't reproduce
   or roll back a model.
