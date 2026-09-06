"""Portable primitives for autonomous research over River models."""

from everbench.auto.classifier import AutoClassifier
from everbench.auto.evaluation import (
    EvaluationOutcome,
    TemporalObservation,
    TemporalSplit,
    evaluate_temporally,
    temporal_split,
)
from everbench.auto.research import (
    Candidate,
    ConstraintResult,
    Evaluation,
    MetricConstraint,
    Objective,
    Observation,
    ResearchSnapshot,
)

__all__ = [
    "AutoClassifier",
    "Candidate",
    "ConstraintResult",
    "Evaluation",
    "EvaluationOutcome",
    "MetricConstraint",
    "Objective",
    "Observation",
    "ResearchSnapshot",
    "TemporalObservation",
    "TemporalSplit",
    "evaluate_temporally",
    "temporal_split",
]
