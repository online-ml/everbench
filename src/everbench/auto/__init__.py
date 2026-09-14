"""Portable primitives for autonomous research over River models."""

from everbench.auto.classifier import AutoClassifier
from everbench.auto.evaluation import (
    ArchiveExample,
    EvaluationOutcome,
    progressive_validate,
)
from everbench.auto.research import (
    Candidate,
    ConstraintResult,
    Evaluation,
    MetricConstraint,
    Objective,
)

__all__ = [
    "AutoClassifier",
    "ArchiveExample",
    "Candidate",
    "ConstraintResult",
    "Evaluation",
    "EvaluationOutcome",
    "MetricConstraint",
    "Objective",
    "progressive_validate",
]
