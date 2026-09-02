"""Click commands for workers, reports, migrations, and the Flask API."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import click

from everbench import artifacts, store
from everbench.db import make_session_factory
from everbench.tasks import load_task
from everbench.workers import collect_events, collect_labels


def configure_logging() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")


@click.group()
def main() -> None:
    """Run live predict-then-learn benchmarks."""
    configure_logging()


@main.command("collect-events")
@click.argument("task_file", type=click.Path(exists=True, dir_okay=False, path_type=str))
def collect_events_command(task_file: str) -> None:
    """Collect events requiring predictions for TASK_FILE."""
    collect_events(make_session_factory(), load_task(task_file))


@main.command("worker")
@click.argument("task_file", type=click.Path(exists=True, dir_okay=False, path_type=str))
def worker(task_file: str) -> None:
    """Run a task's collectors and learner in one supervised process."""
    from everbench.runtime import run_task

    run_task(make_session_factory(), load_task(task_file))


@main.command("collect-labels")
@click.argument("task_file", type=click.Path(exists=True, dir_okay=False, path_type=str))
def collect_labels_command(task_file: str) -> None:
    """Collect labels and finalise delayed negatives for TASK_FILE."""
    collect_labels(make_session_factory(), load_task(task_file))


@main.command()
@click.argument("task_file", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--once", is_flag=True, help="Run one learner cycle, then exit.")
def learner(task_file: str, once: bool) -> None:
    """Predict then train active models for TASK_FILE."""
    task = load_task(task_file)
    from everbench.workers import learner as run_learner

    run_learner(make_session_factory(), task, once)


@main.command()
@click.argument("task_file", type=click.Path(exists=True, dir_okay=False, path_type=str))
def report(task_file: str) -> None:
    """Print persisted River metrics for TASK_FILE."""
    task = load_task(task_file)
    with make_session_factory()() as session:
        rows = store.task_leaderboard(session, task.TASK_NAME)
    for row in rows:
        metrics = " ".join(f"{name}={value:.6f}" for name, value in row["metrics"].items() if value is not None)
        click.echo(f"{row['model_id']}: predictions={row['predictions']} labels={row['labels']} {metrics}".rstrip())


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=lambda: int(os.getenv("PORT", "8000")), show_default=True)
@click.option("--debug/--no-debug", default=True, show_default=True, help="Enable Flask's development reloader.")
def api(host: str, port: int, debug: bool) -> None:
    """Run the Flask API locally; use Gunicorn in production."""
    from everbench.api import create_app

    create_app().run(host=host, port=port, debug=debug)


@main.command("sign-model")
@click.argument("model_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def sign_model(model_file: Path) -> None:
    """Print the SHA-256 and required upload signature for a pickle file."""
    payload = model_file.read_bytes()
    click.echo(f"sha256={artifacts.sha256(payload)}")
    click.echo(f"signature={artifacts.sign(payload)}")


if __name__ == "__main__":
    main()
