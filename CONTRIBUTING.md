# Contributing

## Local development

```bash
cp .env.example .env
uv sync --locked
uv run alembic upgrade head
```

Set `DATABASE_URL` in `.env`. The file is ignored by Git.

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
```

To see the dashboard move without waiting for real-world labels, run the local synthetic task instead:

```bash
uv run everbench debug worker tasks/dummy/task.py
```
