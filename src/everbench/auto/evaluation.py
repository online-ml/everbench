"""Compare River classifiers with the same delayed replay used by backtests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from river import base, metrics

from everbench.auto.research import ConstraintResult, Evaluation, Objective
from everbench.replay import ArchiveExample, replay


@dataclass(frozen=True, kw_only=True)
class EvaluationOutcome:
    """Comparison evidence plus the candidate state produced by causal replay."""

    evaluation: Evaluation
    trained_candidate: base.Classifier
    candidate_predict_seconds: float
    champion_predict_seconds: float


def _evaluate_model(
    *,
    model: base.Classifier,
    observations: Sequence[ArchiveExample],
    metric_definitions: tuple[metrics.base.ClassificationMetric, ...],
):
    trained = model.clone()
    metric = metrics.base.Metrics([definition.clone() for definition in metric_definitions])
    if not metric.works_with(trained):
        raise ValueError("evaluation metrics do not support this model")
    predict = trained.predict_one if metric.requires_labels else trained.predict_proba_one

    def score(*, target, prediction):
        if prediction is not None and prediction != {}:
            metric.update(y_true=target, y_pred=prediction)

    result = replay(
        observations=observations,
        predict=lambda *, event_id, event: predict(event),
        score=score,
        learn=lambda *, event_id, event, label: trained.learn_one(event, label),
    )
    return trained, list(metric), result


def progressive_validate(
    *,
    champion: base.Classifier,
    candidate: base.Classifier,
    observations: Sequence[ArchiveExample],
    objective: Objective,
    max_prediction_time_ratio: float | None = None,
) -> EvaluationOutcome:
    """Compare fresh definitions with shared delayed progressive validation.

    Both models see the same complete archive in event-availability order.
    Replay predicts when each event arrived, reveals its label after the recorded
    delay, updates every metric, and only then learns from that label.
    """
    if not observations:
        raise ValueError("evaluation contains no observations")

    metric_definitions = (objective.metric,) + tuple(constraint.metric for constraint in objective.metric_constraints)
    trained_champion, champion_metrics, champion_result = _evaluate_model(
        model=champion, observations=observations, metric_definitions=metric_definitions
    )
    trained_candidate, candidate_metrics, candidate_result = _evaluate_model(
        model=candidate, observations=observations, metric_definitions=metric_definitions
    )

    constraints = tuple(
        constraint.evaluate(champion_score=champion_metric.get(), candidate_score=candidate_metric.get())
        for constraint, champion_metric, candidate_metric in zip(
            objective.metric_constraints,
            champion_metrics[1:],
            candidate_metrics[1:],
            strict=True,
        )
    )
    if max_prediction_time_ratio is not None:
        ratio = candidate_result.predict_seconds / max(champion_result.predict_seconds, 1e-12)
        constraints += (
            ConstraintResult(
                name="prediction_time_ratio",
                passed=ratio <= max_prediction_time_ratio,
                detail=f"{ratio:.3f}x <= {max_prediction_time_ratio:.3f}x champion time",
            ),
        )

    return EvaluationOutcome(
        evaluation=Evaluation(
            champion_score=champion_metrics[0].get(),
            candidate_score=candidate_metrics[0].get(),
            observations=candidate_result.labels,
            constraints=constraints,
        ),
        trained_candidate=trained_candidate,
        candidate_predict_seconds=candidate_result.predict_seconds,
        champion_predict_seconds=champion_result.predict_seconds,
    )
