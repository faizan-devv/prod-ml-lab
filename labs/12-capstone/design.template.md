# Design: real-time velocity features (fraud_model_v2)

## 1. Problem and success metric
What exactly is failing, for whom, and which number must move (and by how much) for this to be a success?

_your answer_

## 2. Feature definitions
Precise definitions of each new feature: window, edges (inclusive/exclusive), does a transaction count itself, units, defaults for unknown customers.

_your answer_

## 3. Offline / online consistency
How is the offline history computed? How is the online value updated? Why will they always agree? What happens on a service restart, or a Redis flush?

_your answer_

## 4. Feast design
Sources, feature view(s), TTL and why, the new feature service, and why `fraud_model_v1` must not change.

_your answer_

## 5. Validation and rollout
How do you compare v2 with the champion fairly? What would make you *not* promote? Shadow/canary plan. How do you roll back?

_your answer_

## 6. Risks
Latency, Redis memory, hot customers, clock skew, abuse. At least three, with mitigations.

_your answer_
