# Everbench

https://everbench-production.up.railway.app

This is a platform to host live, never-ending benchmarks. The goal is to evaluate machine learning models on streaming data tasks, be it regression, classification, clustering, anomaly detection, etc.

Each task has a worker that collects events. Every model assigned to a task makes a prediction for each event. Another worker collects labels once the ground truth is available. This allows updating each model, and updating evaluation metrics. The system stores only what it has to, while events and labels are archived into Parquet files.

## Autonomous online research

Tasks may opt into a Karpathy-style autonomous research loop with an `AutoResearchConfig`. An `AutoClassifier`
holds the serving champion, immutable objective, generation, and arbitrary problem context. Everbench keeps the
coding agent outside that portable River-compatible class.

Everbench publishes exactly one immutable Parquet file for each completed UTC availability week. When a new week is
available, the agent may replace a task's complete `candidate.py`: feature extraction over the raw event payload,
River model families, hyperparameters, drift handling, ensembles, and stacking are all in scope. Every evaluation
constructs fresh champion and candidate instances and causally replays the same archive week through both, preserving
the original event and delayed-label availability times with River's `evaluate.progressive_val_score`. The best
candidate is promoted only when it beats that fresh
champion under the task-owned metric and resource constraints. `candidate_budget_per_week` configures how many
candidate programs the researcher may evaluate during that single weekly run.

Candidate execution has a fixed dependency/import policy, subprocess timeout, file descriptor/output limits, source
size limit, model artifact size limit, and a scrubbed environment. Candidate source cannot import operating-system,
filesystem, process, or network modules. Archives—not model artifacts—own the observations. Serving models may retain
bounded learned parameters and sufficient statistics, but candidate programs are instructed not to retain raw rows or
payload identifiers.

Useful commands:

```console
uv run everbench auto bootstrap tasks/wiki_liftwing/task.py
uv run everbench auto reflect tasks/wiki_liftwing/task.py
uv run everbench auto status tasks/wiki_liftwing/task.py
uv run everbench auto-worker-all
```

The Railway research service uses `EVERBENCH_SERVICE_ROLE=researcher`. It requires `OPENAI_API_KEY` in addition to
the database, archive, and model-signing settings used by the other services.
