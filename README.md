# everbench

Everbench runs live predict-then-learn benchmarks. A task file defines a data
source and labels; the shared runtime stores events, makes predictions, learns
when labels arrive, and exposes a small dashboard.

## Local development

```bash
cp .env.example .env
uv sync --locked
uv run alembic upgrade head
```

Set `DATABASE_URL` in `.env`. The file is ignored by Git.

Run the benchmark worker in one terminal:

```bash
uv run everbench debug worker tasks/wiki_leftwing.py
```

Run the web server in another terminal:

```bash
uv run everbench api
```

The local Flask server reloads automatically when Python, template, or static
files change. Production continues to use Gunicorn.

Run the quality checks before committing:

```bash
uv run prek install
uv run prek run --all-files
```

Prek runs project-pinned Ruff (formatting, linting), ty (type checking), and
checks that `uv.lock` matches `pyproject.toml`. GitHub Actions runs the same
checks and the unit tests on every pull request and push to `main`.

Open http://127.0.0.1:8000 to see the task list, runtime statistics, and model
leaderboard. The dashboard is server-rendered by Flask and uses a vendored
HTMX 2 file for its five-second live refresh; no Node build step is required.

To see the dashboard move without waiting for real-world labels, run the local
synthetic task instead:

```bash
uv run everbench debug worker tasks/dummy.py
```

## Models

Models are uploaded directly to a task as signed pickles. One request validates
and registers the model:

```bash
signature="$(python sign_model.py model.pkl)"
curl -X POST http://127.0.0.1:8000/api/tasks/wiki-leftwing/models \
  -H 'X-API-Key: …' \
  -H "X-Everbench-Artifact-Signature: $signature" \
  -F 'model=@model.pkl' \
  -F 'model_id=my-model' \
  -F 'owner=your-name' \
  -F 'class_definition=<your_model.py'
```

Pickled models are supported only when signed with
`EVERBENCH_MODEL_SIGNING_KEY`; pickle uploads are trusted code, not safe files
from arbitrary users. To request the signing key, email
maxhalford25@gmail.com. Treat it as a secret: it grants the ability to submit
Python code that the server will unpickle. Set the received value as
`EVERBENCH_MODEL_SIGNING_KEY` locally. The API page contains a reusable
standard-library `sign_model.py` example. Everbench
checks the online-learning protocol, deep-copies the model, and replays the
five newest labelled archived examples before registering it. The dashboard
links to the complete API documentation. Set a separate
`EVERBENCH_API_KEY`: every documented API route requires it in the
`X-API-Key` header.

Predictors do not have to learn online. A scoring-only model is still evaluated
against every arriving label, but is never given an in-place update. Model
artifacts use [cloudpickle](https://github.com/cloudpipe/cloudpickle), so a
class defined in your upload module travels with its pickle—there are no
task-defined or server-installed model kinds. For the English Wikipedia task,
the provided Lift Wing example is such a scoring-only model:

```bash
uv run python tasks/liftwing/liftwing_revertrisk.py --user-agent 'your-tool (you@example.com)'
python sign_model.py liftwing.pkl
```

It calls Wikimedia with the revision ID and treats the returned revert
probability as its prediction. Upload it through the same multipart endpoint,
passing `tasks/liftwing/liftwing_revertrisk.py` as `class_definition`.

## Tasks

Task-specific code lives in `tasks/`. Copy `tasks/wiki_leftwing.py` to define
the stream URLs, event ID, frozen features, label extraction, `PROBLEM_TYPE`,
and River `METRICS`. Supported problem types are `regression`,
`binary_classification`, `multiclass_classification`, `clustering`, and
`anomaly_detection`.

Metrics are configured as River metric instances in the task file. They are
kept separately for each model, updated from the prediction made before a
label arrived, and checkpointed in Postgres with the model's cumulative
prediction and label counts. For uncommon prediction semantics, a task may
also provide `predict_for(model, features)`; its result is stored and passed
to every configured metric. If metrics need different representations of that
result (for example an accuracy threshold versus a probability loss), define
`metric_inputs_for(metric, y_true, prediction)` in the task file.

## Deployment

Use two Railway services from this same GitHub repository, sharing the same
Postgres database:

```bash
uv run everbench worker-all
uv run gunicorn --bind "0.0.0.0:${PORT:-8000}" 'everbench.api:create_app()'
```

Set the first command as the Worker service's custom start command, and the
second as the Web service's. Keep the Worker at one replica: it supervises one
runtime for every top-level `tasks/*.py` file. Adding a task file and pushing
to `main` therefore starts it automatically after Railway redeploys the
worker; no Railway service needs to be created per task. The Web service is
the only service that needs a public domain. Configure its healthcheck path as
`/api/health`.

Add a Railway Postgres service and expose its `DATABASE_URL` to both services.
Set the shared secrets and R2 variables (`EVERBENCH_API_KEY`,
`EVERBENCH_MODEL_SIGNING_KEY`, `S3_BUCKET_NAME`, `S3_ENDPOINT_URL`,
`S3_ACCESS_KEY_ID`, and `S3_SECRET_ACCESS_KEY`) as Railway shared variables.
The shared start script applies migrations before starting either service and
serializes them with a Postgres advisory lock. Enable GitHub autodeploys on
`main` (and “Wait for CI” if available) for both services.

The worker uses bounded in-memory write batches and resumes committed work from
Postgres after restart. Set `EVERBENCH_ARCHIVE_ROOT` to durable, shared storage
visible to both the worker and web service to enable its built-in compactor.
Every hour by default it writes fully
processed rows that are older than `EVERBENCH_ARCHIVE_AFTER_DAYS` (30 by
default) to content-addressed Parquet files partitioned by UTC ISO week, then
deletes the corresponding events, labels, predictions, and processing receipts
from Postgres. The archive file is atomically published before that transaction,
so a failed compaction leaves the source rows available for a later retry.
