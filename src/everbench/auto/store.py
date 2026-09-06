"""Persistence for autonomous research audit records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from everbench.schema import AutoExperiment


def begin_experiment(
    session: Session,
    task_name: str,
    model_id: str,
    parent_generation: int,
    researcher: str,
    research_summary: dict[str, Any],
    promotion_start_sequence: int,
    promotion_end_sequence: int,
    champion_artifact_id: str,
) -> AutoExperiment:
    experiment = AutoExperiment(
        experiment_id=str(uuid4()),
        task_name=task_name,
        model_id=model_id,
        parent_generation=parent_generation,
        researcher=researcher,
        status="running",
        research_summary=research_summary,
        promotion_start_sequence=promotion_start_sequence,
        promotion_end_sequence=promotion_end_sequence,
        champion_artifact_id=champion_artifact_id,
    )
    session.add(experiment)
    return experiment


def finish_experiment(
    experiment: AutoExperiment,
    *,
    status: str,
    hypothesis: str | None = None,
    proposal: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    candidate_artifact_id: str | None = None,
    error: str | None = None,
) -> None:
    if experiment.status != "running":
        raise ValueError("only a running experiment can be finished")
    if status not in {"rejected", "promoted", "failed"}:
        raise ValueError(f"invalid terminal experiment status: {status!r}")
    experiment.status = status
    experiment.hypothesis = hypothesis
    experiment.proposal = proposal
    experiment.evaluation = evaluation
    experiment.candidate_artifact_id = candidate_artifact_id
    experiment.error = error[:2_000] if error else None
    experiment.completed_at = datetime.now(UTC)


def recent_experiments(session: Session, task_name: str, model_id: str, limit: int = 20) -> list[AutoExperiment]:
    return list(
        session.scalars(
            select(AutoExperiment)
            .where(AutoExperiment.task_name == task_name, AutoExperiment.model_id == model_id)
            .order_by(AutoExperiment.started_at.desc())
            .limit(limit)
        )
    )


def latest_experiment(session: Session, task_name: str, model_id: str) -> AutoExperiment | None:
    return session.scalar(
        select(AutoExperiment)
        .where(
            AutoExperiment.task_name == task_name,
            AutoExperiment.model_id == model_id,
        )
        .order_by(AutoExperiment.started_at.desc())
        .limit(1)
    )
