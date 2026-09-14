# Everbench

Everbench evaluates evolving predictors against continuously arriving observations. Autonomous model research extends
that domain with controlled proposals and evidence-based replacement of a serving predictor.

## Autonomous model research

**Champion**:
The predictor currently responsible for serving predictions and learning from observations.
_Avoid_: Current model, production model

**Candidate**:
A proposed successor to a champion, compared from a fresh state on an archive week.
_Avoid_: Experiment model, new model

**Candidate program**:
The immutable source and hyperparameters that construct a fresh, untrained candidate. Learned state and observations
are not part of the candidate program.
_Avoid_: Search configuration, bounded model space

**Archive week**:
One immutable Parquet file containing every observation made available during one UTC week. It is the common offline
corpus used to compare fresh champion and candidate instances.
_Avoid_: Archive shard, research history

**Serving state**:
Bounded learned parameters and sufficient statistics used by a champion while serving and learning. It excludes
observations and raw payload identifiers.
_Avoid_: Model data, training history

**Model snapshot**:
A durable checkpoint of serving state used for restart recovery. It is not a source of offline research observations.
_Avoid_: Model artifact, training archive

**Weekly comparison**:
Measurements produced by replaying one archive week causally through fresh champion and candidate instances. The
research agent may iterate on this same corpus; River's delayed progressive validation defines the replay semantics.
It is not a separate holdout.
_Avoid_: Sealed cohort, live score

**Research evaluation**:
Repeatable causal measurements produced by replaying the same archive week through fresh champion and candidate
instances while the agent edits a candidate program.
_Avoid_: Production score, live evaluation

**Objective**:
The owner-defined, agent-immutable criteria that determine whether a candidate may replace a champion.
_Avoid_: Reward, agent goal

**Reflection**:
One weekly research session that uses an archive week and owner context to evaluate a configured budget of candidate
programs, then selects at most one for promotion.
_Avoid_: Self-edit, retraining
