"""Click commands for workers, reports, migrations, and the Flask API."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import click
from alembic.config import Config as AlembicConfig
from sqlalchemy import inspect

from alembic import command
from everbench import artifacts, reporting
from everbench.collectors import collect_source
from everbench.db import make_engine, make_session_factory
from everbench.schema import Base
from everbench.tasks import discover_tasks, load_task


def configure_logging() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")


@click.group()
def main() -> None:
    """Run live predict-then-learn benchmarks."""
    configure_logging()


@main.group()
def debug() -> None:
    """Run individual components for local diagnosis."""


@debug.command("collect")
@click.argument("task_file", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.argument("source_name")
def collect_command(*, task_file: str, source_name: str) -> None:
    """Collect a named source from TASK_FILE."""
    from threading import Event

    task = load_task(path=task_file)
    source = next((source for source in task.sources if source.name == source_name), None)
    if source is None:
        raise click.ClickException(
            f"unknown source {source_name!r}; choose from {', '.join(s.name for s in task.sources)}"
        )
    collect_source(sessions=make_session_factory(), task=task, source=source, stop=Event())


@debug.command("worker")
@click.argument("task_file", type=click.Path(exists=True, dir_okay=False, path_type=str))
def worker(*, task_file: str) -> None:
    """Run a task's collectors and learner in one supervised process."""
    from everbench.runtime import run_task

    run_task(sessions=make_session_factory(), task=load_task(path=task_file))


@main.command("worker-all")
@click.option(
    "--tasks-directory",
    default="tasks",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=str),
)
@click.option("--task", "task_names", multiple=True, help="Run only the named task (repeatable).")
@click.option("--schedule-research", is_flag=True, help="Run weekly research in this worker process.")
def worker_all(*, tasks_directory: str, task_names: tuple[str, ...], schedule_research: bool) -> None:
    """Run every top-level task definition in one supervised process."""
    from everbench.runtime import run_tasks

    run_tasks(
        sessions=make_session_factory(),
        tasks=discover_tasks(directory=tasks_directory, task_names=task_names),
        schedule_research=schedule_research,
    )


@main.group("auto")
def auto() -> None:
    """Bootstrap, inspect, and run autonomous model research."""


@auto.command("bootstrap")
@click.argument("task_file", type=click.Path(exists=True, dir_okay=False, path_type=str))
def auto_bootstrap(*, task_file: str) -> None:
    """Register and pre-train TASK_FILE's initial auto champion."""
    from everbench.auto.service import AutoResearchRunner

    report = AutoResearchRunner(sessions=make_session_factory(), task=load_task(path=task_file)).bootstrap()
    click.echo(f"{report.task_name}/{report.model_id}: {report.status} {report.detail}".rstrip())


@auto.command("reflect")
@click.argument("task_file", type=click.Path(exists=True, dir_okay=False, path_type=str))
def auto_reflect(*, task_file: str) -> None:
    """Run one propose-evaluate-promote reflection for TASK_FILE."""
    from everbench.auto.service import AutoResearchRunner

    report = AutoResearchRunner(sessions=make_session_factory(), task=load_task(path=task_file)).reflect()
    click.echo(
        f"{report.task_name}/{report.model_id}: {report.status} generation={report.generation} "
        f"experiment={report.experiment_id} {report.detail}".rstrip()
    )


@auto.command("status")
@click.argument("task_file", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--limit", default=10, show_default=True, type=click.IntRange(1, 100))
def auto_status(*, task_file: str, limit: int) -> None:
    """Show recent autonomous research experiments for TASK_FILE."""
    from everbench.auto import store as auto_store

    task = load_task(path=task_file)
    config = task.AUTO_RESEARCH
    if config is None:
        raise click.ClickException(f"task {task.TASK_NAME!r} does not configure autonomous research")
    with make_session_factory()() as session:
        rows = auto_store.recent_experiments(
            session=session, task_name=task.TASK_NAME, model_id=config.model_id, limit=limit
        )
    if not rows:
        click.echo(f"{task.TASK_NAME}/{config.model_id}: no experiments")
        return
    for row in rows:
        evaluation = row.evaluation or {}
        score = evaluation.get("candidate_score")
        improvement = evaluation.get("improvement")
        score_text = f" score={score:.6f} improvement={improvement:.6f}" if score is not None else ""
        click.echo(
            f"{row.started_at.isoformat()} generation={row.parent_generation} status={row.status}"
            f"{score_text} {row.hypothesis or row.error or ''}".rstrip()
        )


@main.command("auto-worker-all")
@click.option(
    "--tasks-directory",
    default="tasks",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=str),
)
def auto_worker_all(*, tasks_directory: str) -> None:
    """Run one weekly research pass for every task that opts in."""
    from everbench.auto.service import auto_worker

    auto_worker(sessions=make_session_factory(), tasks=discover_tasks(directory=tasks_directory))


@main.command()
def migrate() -> None:
    """Initialize SQLite from the canonical schema, then apply future revisions."""
    root = Path.cwd()
    alembic_config = root / "alembic.ini"
    if not alembic_config.is_file():
        raise click.ClickException(f"alembic.ini not found in project directory: {root}")
    engine = make_engine()
    try:
        if engine.dialect.name != "sqlite":
            raise click.ClickException("migrate requires a SQLite DATABASE_URL")
        fresh = not inspect(engine).has_table("benchmark_events")
        if fresh:
            Base.metadata.create_all(engine)
            command.stamp(AlembicConfig(str(alembic_config)), "head")
        else:
            command.upgrade(AlembicConfig(str(alembic_config)), "head")
    finally:
        engine.dispose()


@main.command("register-tasks")
@click.option(
    "--tasks-directory",
    default="tasks",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=str),
)
@click.option("--task", "task_names", multiple=True, help="Register only the named task (repeatable).")
def register_tasks_command(*, tasks_directory: str, task_names: tuple[str, ...]) -> None:
    """Register deployed task definitions for the dashboard."""
    tasks = discover_tasks(directory=tasks_directory, task_names=task_names)
    with make_session_factory().begin() as session:
        reporting.register_tasks(session=session, task_names=[task.TASK_NAME for task in tasks])


@debug.command()
@click.argument("task_file", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--once", is_flag=True, help="Run one learner cycle, then exit.")
def learner(*, task_file: str, once: bool) -> None:
    """Predict then train active models for TASK_FILE."""
    task = load_task(path=task_file)
    from everbench.learner import learner as run_learner

    run_learner(sessions=make_session_factory(), task=task, once=once)


@debug.command()
@click.argument("task_file", type=click.Path(exists=True, dir_okay=False, path_type=str))
def report(*, task_file: str) -> None:
    """Print persisted River metrics for TASK_FILE."""
    task = load_task(path=task_file)
    with make_session_factory()() as session:
        rows = reporting.task_leaderboard(session=session, task_name=task.TASK_NAME)
    for row in rows:
        metrics = " ".join(f"{name}={value:.6f}" for name, value in row["metrics"].items() if value is not None)
        click.echo(f"{row['model_id']}: predictions={row['predictions']} labels={row['labels']} {metrics}".rstrip())


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=lambda: int(os.getenv("PORT", "8000")), show_default=True)
@click.option("--debug/--no-debug", default=True, show_default=True, help="Enable Flask's development reloader.")
def api(*, host: str, port: int, debug: bool) -> None:
    """Run the Flask API locally; use Gunicorn in production."""
    from everbench.api import create_app

    create_app().run(host=host, port=port, debug=debug)


@debug.command("sign-model")
@click.argument("model_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def sign_model(*, model_file: Path) -> None:
    """Print the SHA-256 and required upload signature for a pickle file."""
    payload = model_file.read_bytes()
    click.echo(f"sha256={artifacts.sha256(payload=payload)}")
    click.echo(f"signature={artifacts.sign(payload=payload)}")


if __name__ == "__main__":
    main()
