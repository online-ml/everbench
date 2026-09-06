# Everbench

Everbench evaluates evolving predictors against continuously arriving observations. Autonomous model research extends
that domain with controlled proposals and evidence-based replacement of a serving predictor.

## Autonomous model research

**Champion**:
The predictor currently responsible for serving predictions and learning from observations.
_Avoid_: Current model, production model

**Candidate**:
A proposed successor to a champion, frozen before its promotion evidence is collected.
_Avoid_: Experiment model, new model

**Candidate program**:
The complete editable Python program that constructs a candidate. Its interface is fixed, but its feature extraction,
online models, hyperparameters, drift logic, ensembles, and stacking are not restricted.
_Avoid_: Search configuration, bounded model space

**Research history**:
Observations and prediction outcomes that a research agent is permitted to inspect.
_Avoid_: Training data, archive

**Promotion evidence**:
Measurements collected without exposing their underlying observations to the research agent before a candidate is
frozen.
_Avoid_: Test data, validation score

**Research evaluation**:
Repeatable causal measurements visible to the research agent while it edits a candidate program. Research evaluation
is disjoint from promotion evidence.
_Avoid_: Promotion run, production score

**Objective**:
The owner-defined, agent-immutable criteria that determine whether a candidate may replace a champion.
_Avoid_: Reward, agent goal

**Reflection**:
One bounded attempt to use research history and owner context to propose and evaluate a candidate.
_Avoid_: Self-edit, retraining
