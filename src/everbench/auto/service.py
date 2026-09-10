"""Everbench orchestration for bootstrapping and running reflections."""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from time import monotonic
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from everbench import archive, archive_store, artifacts, model_store
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
from everbench.auto.dataset import PreparedTemporalData
from everbench.auto.everbench import EverbenchAutoClassifier, complete_observations, iter_complete_observations
from everbench.auto.research import Candidate
from everbench.config import CONFIG
from everbench.heartbeat import Heartbeat
from everbench.schema import AutoExperiment, ModelRegistration
from everbench.tasks import TaskDefinition


class NoNewPromotionEvidence(RuntimeError):
    """A reflection was skipped until a fresh sealed cohort has matured."""


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


def _source_for_registration(session: Session, registration: ModelRegistration) -> str:
    if registration.artifact_id is None:
        raise RuntimeError(f"auto model registration artifact missing for {registration.model_id}")
    artifact_record = model_store.artifact(session, registration.artifact_id)
    source = (artifact_record.metadata_ or {}).get("source_code") if artifact_record is not None else None
    if not isinstance(source, str) or not source.strip():
        raise RuntimeError(f"auto model source missing for {registration.model_id}")
    return source


def _bootstrap_auto_model(sessions: sessionmaker[Session], task: TaskDefinition) -> AutoRunReport:
    """Register the task's initial champion, pre-trained on mature history."""
    config = _config(task)
    with sessions() as session:
        existing = model_store.model_registration(session, task.TASK_NAME, config.model_id)
        if existing is not None:
            auto_classifier, _ = _load_auto(session, existing)
            return AutoRunReport(task.TASK_NAME, config.model_id, None, "existing", auto_classifier.generation)
        observations = complete_observations(session, task, config.history_limit, config.maturity_margin_seconds)
    if len(observations) < config.min_research_observations + config.promotion_observations:
        required = config.min_research_observations + config.promotion_observations
        raise ValueError(f"bootstrap needs at least {required:,} mature observations; received {len(observations):,}")
    source = config.candidate_path.read_text()
    initial_model = build_candidate_model(
        source,
        timeout_seconds=config.candidate_timeout_seconds,
        max_source_bytes=config.max_candidate_source_bytes,
        max_output_bytes=CONFIG.max_model_snapshot_bytes,
    )
    auto_classifier = AutoClassifier(initial_model, config.objective, context=config.context)
    # Bootstrap is ordinary offline pre-training over fully mature historical
    # outcomes. It has no future target leakage and avoids reproducing a cold
    # stream's initial positive-only feedback window.
    for observation in sorted(observations, key=lambda row: (row.available_at, row.sequence)):
        auto_classifier.learn_one(observation.x, observation.y)
    payload = _payload(auto_classifier)
    last = max(observations, key=lambda row: (row.label_available_at, row.sequence))
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
            last.label_available_at,
            last.sequence,
            None,
        )
    return AutoRunReport(
        task.TASK_NAME,
        config.model_id,
        None,
        "bootstrapped",
        0,
        f"trained on {len(observations):,} mature observations",
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
    """Propose, causally evaluate, audit, and atomically promote one candidate."""
    config = _config(task)
    prepared: PreparedTemporalData | None = None
    try:
        with sessions() as session:
            registration = model_store.model_registration(session, task.TASK_NAME, config.model_id)
            if registration is None:
                raise LookupError(f"auto model {config.model_id!r} is not bootstrapped")
            auto_classifier, champion_artifact_id = _load_auto(session, registration)
            current_source = _source_for_registration(session, registration)
            registration_artifact_id = registration.artifact_id
            if registration_artifact_id is None:
                raise RuntimeError(f"auto model registration artifact missing for {config.model_id}")
            prepared = PreparedTemporalData.from_observations(
                iter_complete_observations(session, task, config.history_limit, config.maturity_margin_seconds)
            )
            recent = store.recent_experiments(session, task.TASK_NAME, config.model_id)
            manifests = archive_store.task_archives(session, task.TASK_NAME) if config.retain_raw_examples else []
        return _reflect_prepared_once(
            sessions,
            task,
            config,
            auto_classifier,
            champion_artifact_id,
            current_source,
            registration_artifact_id,
            prepared,
            recent,
            manifests,
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
    current_source: str,
    registration_artifact_id: str,
    prepared: PreparedTemporalData,
    recent: list[AutoExperiment],
    manifests: list[Any],
    researcher: Any | None,
) -> AutoRunReport:
    split = prepared.split(config.promotion_observations, config.min_research_observations)
    research_span = (split.research[-1].available_at - split.research[0].available_at).total_seconds()
    if research_span < config.min_research_span_seconds:
        raise NoNewPromotionEvidence(
            f"research history spans {research_span / 3_600:.1f}h; waiting for "
            f"{config.min_research_span_seconds / 3_600:.1f}h before causal evaluation"
        )
    promotion_start, promotion_end = split.promotion.sequence_bounds()
    snapshot = auto_classifier.research_snapshot()
    summary = summarize_research(split.research, include_raw_examples=config.retain_raw_examples)
    for example in summary.get("raw_examples", []):
        example["current_champion_prediction"] = snapshot.champion.predict_proba_one(example["event"])
    if manifests:
        summary["archived_examples"] = [
            {"event_id": event_id, "event": event, "label": label}
            for event_id, event, label in archive.latest_labelled_examples(manifests, limit=5)
        ]
    backend = researcher or OpenAICodeResearcher(max_experiments=config.max_research_experiments)
    researcher_name = getattr(backend, "model", type(backend).__name__)
    inner_promotion_size = min(
        config.research_evaluation_observations,
        max(len(split.research) // 4, 1),
    )
    inner_split = prepared.split(
        inner_promotion_size,
        len(split.research) - inner_promotion_size,
        stop=len(split.research),
    )
    research_request = ResearchRequest(
        context=snapshot.context,
        objective=describe_objective(snapshot.objective),
        research_summary=summary,
        current_source=current_source,
        previous_experiments=_past_experiments(recent),
    )

    # Create the durable running row only once all local preparation has
    # succeeded. From here onward, every failure is recorded below.
    with sessions.begin() as session:
        model_store.lock_model_registrations(session, task.TASK_NAME)
        previous = store.latest_experiment(session, task.TASK_NAME, config.model_id)
        if previous is not None and promotion_start <= previous.promotion_end_sequence:
            raise NoNewPromotionEvidence(
                f"latest promotion cohort overlaps experiment {previous.experiment_id}; waiting for "
                f"{config.promotion_observations:,} new mature observations"
            )
        experiment = store.begin_experiment(
            session,
            task.TASK_NAME,
            config.model_id,
            auto_classifier.generation,
            str(researcher_name),
            summary,
            promotion_start,
            promotion_end,
            champion_artifact_id,
        )
        experiment_id = experiment.experiment_id

    def research_evaluate(source: str) -> dict[str, Any]:
        outcome = evaluate_candidate_source(
            source,
            snapshot.champion,
            inner_split,
            snapshot.objective,
            max_prediction_time_ratio=config.max_prediction_time_ratio,
            timeout_seconds=config.candidate_timeout_seconds,
            max_source_bytes=config.max_candidate_source_bytes,
            max_output_bytes=CONFIG.max_model_snapshot_bytes,
        )
        return _evaluation(outcome, config)

    try:
        code_proposal = backend.research(research_request, research_evaluate)
        outcome = evaluate_candidate_source(
            code_proposal.source,
            snapshot.champion,
            split,
            snapshot.objective,
            max_prediction_time_ratio=config.max_prediction_time_ratio,
            timeout_seconds=config.candidate_timeout_seconds,
            max_source_bytes=config.max_candidate_source_bytes,
            max_output_bytes=CONFIG.max_model_snapshot_bytes,
        )
        promoted = auto_classifier.consider(
            Candidate(
                outcome.trained_candidate,
                parent_generation=snapshot.generation,
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

    The runner owns database/archive access, agent invocation, isolated code
    execution, sealed evaluation, and atomic promotion. None of those concerns
    leak into the River-compatible classifier.
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
    *,
    once: bool = False,
    stop: threading.Event | None = None,
) -> None:
    """Periodically reflect for every task that opts into autonomous research."""
    configured = [task for task in tasks if isinstance(task.AUTO_RESEARCH, AutoResearchConfig)]
    if not configured:
        raise ValueError("no tasks define autonomous research")
    stop = stop or threading.Event()
    next_run = {task.TASK_NAME: 0.0 for task in configured}
    with Heartbeat(sessions, None, "auto-researcher"):
        while not stop.is_set():
            now = monotonic()
            due = configured if once else [task for task in configured if next_run[task.TASK_NAME] <= now]
            if not due:
                stop.wait(max(min(next_run.values()) - now, 0.0))
                continue
            for task in due:
                config = _config(task)
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
                next_run[task.TASK_NAME] = monotonic() + config.interval_seconds
            if once:
                return
