# Contributing

## Local development

```bash
cp .env.example .env
uv sync --locked
uv run everbench migrate
```

Set `DATABASE_URL` in `.env` to a SQLite file path. The file is ignored by Git. Production uses one Railway service with a persistent volume mounted at `/data`; the web server, task worker, and weekly researcher share that file. Set `DATABASE_URL=sqlite:////data/everbench.db` and `EVERBENCH_ARCHIVE_ROOT=/data/archives` there.

When upgrading an existing installation, stop workers before running `everbench migrate`, then restart them with the new code. Keep the SQLite database on persistent storage and take a consistent backup before changing the schema.

Run the benchmark worker in one terminal:

```bash
uv run everbench debug worker tasks/wiki_liftwing/task.py
```

Run the web server in another terminal:

```bash
uv run everbench api
```

The local Flask server reloads automatically when Python, template, or static files change. Production continues to use Gunicorn.

Run the quality checks before committing:

```bash
uv run prek install
uv run prek run --all-files
uv run python -m pytest -q
```

To see the dashboard move without waiting for real-world labels, run the local synthetic task instead:

```bash
uv run everbench debug worker tasks/dummy/task.py
```

## Code conventions

Use keyword-only parameters for application functions and constructors: `def collect(*, task, source): ...`. Use `@dataclass(kw_only=True)` for records, with `slots=True` for frequently created records. Ruff enforces zero positional parameters with `PLR0917` and bans named tuples with `TID251`; `self` and `cls` are implicit. Python special methods and callbacks invoked positionally by River, Flask or the standard library have narrowly scoped exemptions. Submitted models implement `predict_one(*, event_id, event)` (or `predict_proba_one` / `score_one`) and optional `learn_one(*, event_id, event, label)`; argument names are part of that interface. Keep README paragraphs on one physical line.

## Task structure

A task exports one `TASK = TaskDefinition(...)`. Its sources yield `Observation` and `LabelInput` records; the collector handles persistence. `LabelPolicy` defines a target horizon, matching tolerance and deadline default. A known target of zero is distinct from an unavailable target (`None`); absence of a resolution means the target is still pending. Task modules contain feed parsing and target rules, while shared modules handle database state, learning and archive replay.

The learner predicts pending observations, processes resolved targets, then checkpoints model state using the ready-queue sequence. Backtesting and autonomous research share River's `stream.simulate_qa` for delayed replay; due labels are applied before the next observation, including time ties. On promotion, the archive-trained candidate starts a fresh live generation with reset metrics and only sees observations arriving after the promotion boundary. This keeps restart recovery and live evaluation independent of the candidate's historical training window.
