"""Everbench orchestration for bootstrapping and running reflections."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from everbench import archive_store, artifacts, model_store
from everbench.auto import store
from everbench.auto.classifier import AutoClassifier
from everbench.auto.code_execution import build_candidate_model, evaluate_candidate_source
from everbench.auto.code_researcher import (
    OpenAICodeResearcher,
    ResearchRequest,
    describe_objective,
    summarize_research,
)
from everbench.auto.config import AutoResearchConfig
from everbench.auto.dataset import ArchiveWeek
from everbench.auto.everbench import EverbenchAutoClassifier
from everbench.auto.research import Candidate, ConstraintResult
from everbench.config import CONFIG
from everbench.heartbeat import Heartbeat
from everbench.metrics import MetricTracker
from everbench.schema import ArchiveManifest, AutoExperiment, ModelRegistration
from everbench.tasks import TaskDefinition


class NoNewPromotionEvidence(RuntimeError):
    """A reflection was skipped until a new complete archive week exists."""


@dataclass(frozen=True)
class AutoRunReport:
    task_name: str
    model_id: str
    experiment_id: str | None
    status: str
    generation: int
    detail: str = ""


def _config(task: TaskDefinition) -> AutoResearchConfig:
    config = task.AUTO_RESEARCH
    if not isinstance(config, AutoResearchConfig):
        raise ValueError(f"task {task.TASK_NAME!r} does not define an AutoResearchConfig")
    return config


def _model_artifact(session: Session, registration: ModelRegistration):
    snapshot = model_store.latest_snapshot(session, registration.task_name, registration.model_id)
    artifact_id = snapshot.artifact_id if snapshot is not None else registration.artifact_id
    artifact_record = model_store.artifact(session, artifact_id) if artifact_id else None
    if artifact_record is None:
        raise RuntimeError(f"auto model artifact missing for {registration.model_id}")
    return artifact_record


def _load_auto(session: Session, registration: ModelRegistration) -> tuple[AutoClassifier, str]:
    artifact_record = _model_artifact(session, registration)
    wrapped = artifacts.loads(artifact_record.payload, artifact_record.signature)
    if not isinstance(wrapped, EverbenchAutoClassifier):
        raise TypeError(f"{registration.model_id!r} is not an EverbenchAutoClassifier artifact")
    return wrapped.auto_classifier, artifact_record.artifact_id


def _payload(auto_classifier: AutoClassifier) -> bytes:
    payload = artifacts.dumps(EverbenchAutoClassifier(auto_classifier))
    if len(payload) > CONFIG.max_model_snapshot_bytes:
        raise ValueError(
            f"serialized auto model is {len(payload):,} bytes; limit is {CONFIG.max_model_snapshot_bytes:,} bytes"
        )
    return payload


def _with_model_size_constraint(outcome: Any, max_bytes: int) -> Any:
    """Attach deployability evidence to an evaluated candidate."""
    candidate_bytes = len(artifacts.dumps(outcome.trained_candidate))
    constraint = ConstraintResult(
        name="serialized_model_size",
        passed=candidate_bytes <= max_bytes,
        detail=f"{candidate_bytes:,} bytes <= {max_bytes:,} bytes",
    )
    evaluation = replace(
        outcome.evaluation,
        constraints=outcome.evaluation.constraints + (constraint,),
    )
    return replace(outcome, evaluation=evaluation)


def _reset_generation_metrics(session: Session, task: TaskDefinition, model_id: str) -> None:
    """Start leaderboard metrics at the promoted generation boundary."""
    session.execute(
        text(
            """UPDATE benchmark_model_events
                  SET evaluated_at = now()
                WHERE task_name = :task_name AND model_id = :model_id
                  AND prediction_status = 'predicted' AND evaluated_at IS NULL"""
        ),
        {"task_name": task.TASK_NAME, "model_id": model_id},
    )
    tracker = MetricTracker.fresh(task.PROBLEM_TYPE, task.METRICS)
    model_store.save_metric_state(
        session,
        task.TASK_NAME,
        model_id,
        tracker.definition,
        tracker.payload(),
        tracker.predictions,
        tracker.observations,
        tracker.values(),
    )


def _source_for_registration(session: Session, registration: ModelRegistration) -> str:
    if registration.artifact_id is None:
        raise RuntimeError(f"auto model registration artifact missing for {registration.model_id}")
    artifact_record = model_store.artifact(session, registration.artifact_id)
    source = (artifact_record.metadata_ or {}).get("source_code") if artifact_record is not None else None
    if not isinstance(source, str) or not source.strip():
        raise RuntimeError(f"auto model source missing for {registration.model_id}")
    return source


def _latest_archive_week(session: Session, task_name: str) -> tuple[date, ArchiveManifest] | None:
    cutoff = datetime.now(UTC) - timedelta(days=CONFIG.archive_after_days)
    week_start = archive_store.latest_complete_archive_week(session, task_name, cutoff)
    if week_start is None:
        return None
    manifest = archive_store.archive_for_week(session, task_name, week_start)
    if manifest is None:
        return None
    return week_start, manifest


def _prepare_archive_week(manifest: ArchiveManifest) -> ArchiveWeek:
    return ArchiveWeek.open(manifest)


def _bootstrap_auto_model(sessions: sessionmaker[Session], task: TaskDefinition) -> AutoRunReport:
    """Register the initial champion, trained on the latest complete archive week."""
    config = _config(task)
    with sessions() as session:
        existing = model_store.model_registration(session, task.TASK_NAME, config.model_id)
        if existing is not None:
            auto_classifier, _ = _load_auto(session, existing)
            return AutoRunReport(task.TASK_NAME, config.model_id, None, "existing", auto_classifier.generation)
        archived_week = _latest_archive_week(session, task.TASK_NAME)
    source = config.seed_path.read_text()
    initial_model = build_candidate_model(
        source,
        timeout_seconds=config.candidate_timeout_seconds,
        max_source_bytes=config.max_candidate_source_bytes,
        max_output_bytes=CONFIG.max_model_snapshot_bytes,
    )
    checkpoint_label_available_at = None
    checkpoint_event_sequence = None
    trained_observations = 0
    trained_week: date | None = None
    if archived_week is not None:
        trained_week, manifest = archived_week
        with _prepare_archive_week(manifest) as prepared:
            if len(prepared) < config.min_archive_observations:
                raise ValueError(
                    f"bootstrap archive week needs at least {config.min_archive_observations:,} observations; "
                    f"received {len(prepared):,}"
                )
            outcome = evaluate_candidate_source(
                source,
                initial_model,
                prepared,
                config.objective,
                max_prediction_time_ratio=config.max_prediction_time_ratio,
                timeout_seconds=config.candidate_timeout_seconds,
                max_source_bytes=config.max_candidate_source_bytes,
                max_output_bytes=CONFIG.max_model_snapshot_bytes,
            )
            initial_model = outcome.trained_candidate
            checkpoint_label_available_at = outcome.checkpoint_label_available_at
            checkpoint_event_sequence = outcome.checkpoint_event_sequence
            trained_observations = len(prepared)
    auto_classifier = AutoClassifier(initial_model, config.objective, context=config.context)
    initial_model_bytes = len(artifacts.dumps(auto_classifier.model))
    if initial_model_bytes > config.max_candidate_model_bytes:
        raise ValueError(
            f"serialized bootstrap model is {initial_model_bytes:,} bytes; "
            f"auto promotion limit is {config.max_candidate_model_bytes:,} bytes"
        )
    payload = _payload(auto_classifier)
    with sessions.begin() as session:
        model_store.lock_model_registrations(session, task.TASK_NAME)
        existing = model_store.model_registration(session, task.TASK_NAME, config.model_id)
        if existing is not None:
            loaded, _ = _load_auto(session, existing)
            return AutoRunReport(task.TASK_NAME, config.model_id, None, "existing", loaded.generation)
        if model_store.active_model_count(session, task.TASK_NAME) >= CONFIG.max_active_models_per_task:
            raise ValueError(f"task {task.TASK_NAME!r} has reached its active model limit")
        artifact_record = model_store.store_artifact(
            session,
            payload,
            artifacts.sign(payload),
            {
                "source": "auto-bootstrap",
                "generation": 0,
                "archive_week": trained_week.isoformat() if trained_week is not None else None,
                "source_code": source,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            },
        )
        model_store.register_model(session, task.TASK_NAME, config.model_id, config.owner, artifact_record.artifact_id)
        model_store.save_pickle_snapshot(
            session,
            task.TASK_NAME,
            config.model_id,
            payload,
            checkpoint_label_available_at,
            checkpoint_event_sequence,
            None,
        )
    return AutoRunReport(
        task.TASK_NAME,
        config.model_id,
        None,
        "bootstrapped",
        0,
        (
            f"trained on {trained_observations:,} observations from archive week {trained_week}"
            if trained_week is not None
            else "registered fresh; no complete archive week is available"
        ),
    )


def _past_experiments(rows: list[AutoExperiment]) -> tuple[dict[str, Any], ...]:
    experiments = []
    for row in reversed(rows):
        experiments.append(
            {
                "generation": row.parent_generation,
                "status": row.status,
                "hypothesis": row.hypothesis,
                "source_sha256": (row.proposal or {}).get("source_sha256"),
                "evaluation": row.evaluation,
                "error": row.error,
            }
        )
    return tuple(experiments)


def _evaluation(outcome: Any, config: AutoResearchConfig) -> dict[str, Any]:
    evaluation = outcome.evaluation
    if config.objective.metric.bigger_is_better:
        improvement = evaluation.candidate_score - evaluation.champion_score
    else:
        improvement = evaluation.champion_score - evaluation.candidate_score
    return {
        "metric": type(config.objective.metric).__name__,
        "champion_score": evaluation.champion_score,
        "candidate_score": evaluation.candidate_score,
        "improvement": improvement,
        "observations": evaluation.observations,
        "constraints": [
            {"name": item.name, "passed": item.passed, "detail": item.detail} for item in evaluation.constraints
        ],
        "timing_seconds": {
            "champion_predict": outcome.champion_predict_seconds,
            "candidate_predict": outcome.candidate_predict_seconds,
        },
    }


def _reflect_once(
    sessions: sessionmaker[Session],
    task: TaskDefinition,
    *,
    researcher: Any | None = None,
) -> AutoRunReport:
    """Compare fresh definitions on one complete archive week."""
    config = _config(task)
    prepared: ArchiveWeek | None = None
    try:
        with sessions() as session:
            registration = model_store.model_registration(session, task.TASK_NAME, config.model_id)
            if registration is None:
                raise LookupError(f"auto model {config.model_id!r} is not bootstrapped")
            auto_classifier, champion_artifact_id = _load_auto(session, registration)
            champion_source = _source_for_registration(session, registration)
            registration_artifact_id = registration.artifact_id
            if registration_artifact_id is None:
                raise RuntimeError(f"auto model registration artifact missing for {config.model_id}")
            archived_week = _latest_archive_week(session, task.TASK_NAME)
            if archived_week is None:
                raise NoNewPromotionEvidence("waiting for a complete archive week")
            week_start, manifest = archived_week
            recent = store.recent_experiments(session, task.TASK_NAME, config.model_id)
        prepared = _prepare_archive_week(manifest)
        return _reflect_prepared_once(
            sessions,
            task,
            config,
            auto_classifier,
            champion_artifact_id,
            champion_source,
            registration_artifact_id,
            prepared,
            recent,
            week_start,
            researcher,
        )
    finally:
        if prepared is not None:
            prepared.close()


def _reflect_prepared_once(
    sessions: sessionmaker[Session],
    task: TaskDefinition,
    config: AutoResearchConfig,
    auto_classifier: AutoClassifier,
    champion_artifact_id: str,
    champion_source: str,
    registration_artifact_id: str,
    prepared: ArchiveWeek,
    recent: list[AutoExperiment],
    week_start: date,
    researcher: Any | None,
) -> AutoRunReport:
    if len(prepared) < config.min_archive_observations:
        raise NoNewPromotionEvidence(
            f"archive week {week_start} has {len(prepared):,} observations; "
            f"waiting for at least {config.min_archive_observations:,}"
        )
    comparison = prepared
    comparison_start, comparison_end = comparison.sequence_bounds()
    summary = summarize_research(comparison)
    summary["archive_week"] = week_start.isoformat()
    backend = researcher or OpenAICodeResearcher(candidate_budget=config.candidate_budget_per_week)
    researcher_name = getattr(backend, "model", type(backend).__name__)
    champion = build_candidate_model(
        champion_source,
        timeout_seconds=config.candidate_timeout_seconds,
        max_source_bytes=config.max_candidate_source_bytes,
        max_output_bytes=CONFIG.max_model_snapshot_bytes,
    )
    objective = auto_classifier.objective
    objective_description = describe_objective(objective)
    objective_description["max_serialized_model_bytes"] = config.max_candidate_model_bytes
    research_request = ResearchRequest(
        context=auto_classifier.context,
        objective=objective_description,
        research_summary=summary,
        champion_source=champion_source,
        previous_experiments=_past_experiments(recent),
    )

    # Create the durable running row only once all local preparation has
    # succeeded. From here onward, every failure is recorded below.
    with sessions.begin() as session:
        model_store.lock_model_registrations(session, task.TASK_NAME)
        previous = store.experiment_for_cohort(
            session,
            task.TASK_NAME,
            config.model_id,
            comparison_start,
            comparison_end,
        )
        if previous is not None:
            raise NoNewPromotionEvidence(
                f"archive week {week_start} was already evaluated by experiment {previous.experiment_id}"
            )
        experiment = store.begin_experiment(
            session,
            task.TASK_NAME,
            config.model_id,
            auto_classifier.generation,
            str(researcher_name),
            summary,
            comparison_start,
            comparison_end,
            champion_artifact_id,
        )
        experiment_id = experiment.experiment_id

    def research_evaluate(source: str) -> dict[str, Any]:
        outcome = _with_model_size_constraint(
            evaluate_candidate_source(
                source,
                champion,
                comparison,
                objective,
                max_prediction_time_ratio=config.max_prediction_time_ratio,
                timeout_seconds=config.candidate_timeout_seconds,
                max_source_bytes=config.max_candidate_source_bytes,
                max_output_bytes=CONFIG.max_model_snapshot_bytes,
            ),
            config.max_candidate_model_bytes,
        )
        return _evaluation(outcome, config)

    try:
        code_proposal = backend.research(research_request, research_evaluate)
        outcome = _with_model_size_constraint(
            evaluate_candidate_source(
                code_proposal.source,
                champion,
                comparison,
                objective,
                max_prediction_time_ratio=config.max_prediction_time_ratio,
                timeout_seconds=config.candidate_timeout_seconds,
                max_source_bytes=config.max_candidate_source_bytes,
                max_output_bytes=CONFIG.max_model_snapshot_bytes,
            ),
            config.max_candidate_model_bytes,
        )
        promoted = auto_classifier.consider(
            Candidate(
                outcome.trained_candidate,
                parent_generation=auto_classifier.generation,
                hypothesis=code_proposal.hypothesis,
            ),
            outcome.evaluation,
        )
        source_sha256 = hashlib.sha256(code_proposal.source.encode()).hexdigest()
        evaluation = _evaluation(outcome, config)
        payload = _payload(auto_classifier) if promoted else None
        with sessions.begin() as session:
            experiment = session.get(AutoExperiment, experiment_id)
            if experiment is None:
                raise RuntimeError(f"experiment {experiment_id} disappeared")
            registration = session.scalar(
                select(ModelRegistration)
                .where(
                    ModelRegistration.task_name == task.TASK_NAME,
                    ModelRegistration.model_id == config.model_id,
                )
                .with_for_update()
            )
            if registration is None or registration.artifact_id != registration_artifact_id:
                raise RuntimeError("champion changed while the reflection was running")
            candidate_artifact_id = None
            if promoted and payload is not None:
                artifact_record = model_store.store_artifact(
                    session,
                    payload,
                    artifacts.sign(payload),
                    {
                        "source": "auto-promotion",
                        "generation": auto_classifier.generation,
                        "hypothesis": code_proposal.hypothesis,
                        "source_code": code_proposal.source,
                        "source_sha256": source_sha256,
                        "experiment_id": experiment_id,
                        "archive_week": week_start.isoformat(),
                    },
                )
                candidate_artifact_id = artifact_record.artifact_id
                registration.artifact_id = candidate_artifact_id
                model_store.save_pickle_snapshot(
                    session,
                    task.TASK_NAME,
                    config.model_id,
                    payload,
                    outcome.checkpoint_label_available_at,
                    outcome.checkpoint_event_sequence,
                    None,
                )
                _reset_generation_metrics(session, task, config.model_id)
            store.finish_experiment(
                experiment,
                status="promoted" if promoted else "rejected",
                hypothesis=code_proposal.hypothesis,
                proposal={"source_code": code_proposal.source, "source_sha256": source_sha256},
                evaluation=evaluation,
                candidate_artifact_id=candidate_artifact_id,
            )
        return AutoRunReport(
            task.TASK_NAME,
            config.model_id,
            experiment_id,
            "promoted" if promoted else "rejected",
            auto_classifier.generation,
            f"{type(config.objective.metric).__name__} improvement={evaluation['improvement']:.6f}",
        )
    except Exception as error:
        logging.exception("auto reflection %s failed", experiment_id)
        with sessions.begin() as session:
            experiment = session.get(AutoExperiment, experiment_id)
            if experiment is not None and experiment.status == "running":
                store.finish_experiment(experiment, status="failed", error=f"{type(error).__name__}: {error}")
        raise


class AutoResearchRunner:
    """Everbench-owned lifecycle around the portable AutoClassifier.

    The runner owns archive access, agent invocation, isolated weekly
    comparison, and atomic promotion. None of those concerns leak into the
    River-compatible classifier.
    """

    def __init__(
        self,
        sessions: sessionmaker[Session],
        task: TaskDefinition,
        *,
        researcher: Any | None = None,
    ) -> None:
        self.sessions = sessions
        self.task = task
        self.researcher = researcher

    def bootstrap(self) -> AutoRunReport:
        return _bootstrap_auto_model(self.sessions, self.task)

    def reflect(self) -> AutoRunReport:
        return _reflect_once(self.sessions, self.task, researcher=self.researcher)


def auto_worker(
    sessions: sessionmaker[Session],
    tasks: list[TaskDefinition],
) -> None:
    """Run one weekly research pass for every configured task, then exit."""
    configured = [task for task in tasks if isinstance(task.AUTO_RESEARCH, AutoResearchConfig)]
    if not configured:
        raise ValueError("no tasks define autonomous research")
    with Heartbeat(sessions, None, "auto-researcher"):
        for task in configured:
            runner = AutoResearchRunner(sessions, task)
            try:
                report = runner.bootstrap()
                if report.status == "bootstrapped":
                    logging.info("%s: %s", task.TASK_NAME, report.detail)
                report = runner.reflect()
                logging.info("%s: auto reflection %s: %s", task.TASK_NAME, report.status, report.detail)
            except NoNewPromotionEvidence as error:
                logging.info("%s: %s", task.TASK_NAME, error)
            except Exception:
                logging.exception("%s: autonomous research cycle failed", task.TASK_NAME)
