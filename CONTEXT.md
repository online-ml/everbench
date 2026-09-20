# Everbench

Everbench evaluates evolving predictors against continuously arriving observations. Autonomous model research extends
that domain with controlled proposals and evidence-based replacement of a serving predictor.

## Live benchmarks

**Observation**:
A prediction request containing the information available at its forecast origin.
_Avoid_: Training row, labelled sample

**Resolution**:
The final outcome of an observation: a known target or an explicitly unavailable target. Pending observations have no resolution yet.
_Avoid_: Default negative, expired event

**Forecast origin**:
The moment at which the observation's input was measured. It determines the forecast's target time, independently of when processing completes.
_Avoid_: Insertion time

**Live generation**:
One serving model version and the observations admitted after it begins serving. A promoted champion starts a new live generation while keeping its archive-trained serving state.
_Avoid_: Catch-up window

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
research agent may iterate on this same corpus; predictions and learning respect the original observation and target availability times.
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
