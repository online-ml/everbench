# Use completed archive weeks for autonomous research

Everbench publishes exactly one immutable Parquet file for each completed UTC availability week. Autonomous research
runs once for each new file and may evaluate a task-configured budget of candidate programs. Each comparison causally
replays that same week through fresh, untrained champion and candidate instances, and at most the best eligible
candidate is promoted. River's `evaluate.progressive_val_score` owns delayed progressive-validation semantics.
Archives own observations, candidate programs own model definitions, and operational snapshots
contain only bounded serving state; live model state and recent database rows are not research inputs. This trades
faster sub-week iteration and a separate holdout for reproducible comparisons, a simple weekly cadence, and a clear
data-retention boundary.
