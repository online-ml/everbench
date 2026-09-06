"""Value objects exchanged with an external autonomous-research runner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from river import base, metrics


@dataclass(frozen=True)
class Observation:
    """One optional in-memory example captured before online learning."""

    sequence: int
    x: dict[Any, Any]
    y: Any
    prediction: Any
    learn_kwargs: dict[str, Any]


@dataclass(frozen=True)
class ConstraintResult:
    """The evaluator's result for one owner-defined promotion constraint."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class Evaluation:
    """Sealed evidence comparing a frozen candidate with its champion."""

    champion_score: float
    candidate_score: float
    observations: int
    constraints: tuple[ConstraintResult, ...] = ()


@dataclass(frozen=True)
class MetricConstraint:
    """Require a candidate not to regress too far on a secondary metric."""

    name: str
    metric: metrics.base.ClassificationMetric
    max_regression: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric constraint name must be non-empty")
        if self.max_regression < 0:
            raise ValueError("max_regression must be non-negative")

    def copy(self) -> MetricConstraint:
        return MetricConstraint(self.name, self.metric.clone(), self.max_regression)

    def evaluate(self, champion_score: float, candidate_score: float) -> ConstraintResult:
        if self.metric.bigger_is_better:
            regression = champion_score - candidate_score
        else:
            regression = candidate_score - champion_score
        passed = (
            math.isfinite(champion_score)
            and math.isfinite(candidate_score)
            and regression <= self.max_regression
        )
        return ConstraintResult(
            name=self.name,
            passed=passed,
            detail=(
                f"{type(self.metric).__name__}: champion={champion_score:.6f}, "
                f"candidate={candidate_score:.6f}, regression={regression:.6f} "
                f"<= {self.max_regression:.6f}"
            ),
        )


@dataclass(frozen=True)
class Objective:
    """Agent-immutable promotion criteria supplied by the model owner."""

    metric: metrics.base.ClassificationMetric
    min_improvement: float = 0.0
    min_observations: int = 1
    required_constraints: tuple[str, ...] = ()
    metric_constraints: tuple[MetricConstraint, ...] = ()

    def __post_init__(self) -> None:
        if self.min_improvement < 0:
            raise ValueError("min_improvement must be non-negative")
        if self.min_observations <= 0:
            raise ValueError("min_observations must be positive")
        if len(set(self.required_constraints)) != len(self.required_constraints):
            raise ValueError("required constraint names must be unique")
        constraint_names = [constraint.name for constraint in self.metric_constraints]
        if len(set(constraint_names)) != len(constraint_names):
            raise ValueError("metric constraint names must be unique")
        if set(constraint_names) & set(self.required_constraints):
            raise ValueError("metric and externally required constraint names must be distinct")

    def copy(self) -> Objective:
        """Return a detached copy whose stateful River metric is fresh."""
        return Objective(
            metric=self.metric.clone(),
            min_improvement=self.min_improvement,
            min_observations=self.min_observations,
            required_constraints=self.required_constraints,
            metric_constraints=tuple(constraint.copy() for constraint in self.metric_constraints),
        )

    def fresh_metric(self) -> metrics.base.ClassificationMetric:
        return self.metric.clone()

    def accepts(self, evaluation: Evaluation) -> bool:
        """Apply the immutable score threshold and required constraints."""
        if evaluation.observations < self.min_observations:
            return False
        if not math.isfinite(evaluation.champion_score) or not math.isfinite(evaluation.candidate_score):
            return False
        improvement = (
            evaluation.candidate_score - evaluation.champion_score
            if self.metric.bigger_is_better
            else evaluation.champion_score - evaluation.candidate_score
        )
        if improvement <= 0 or improvement < self.min_improvement:
            return False
        names = [result.name for result in evaluation.constraints]
        if len(set(names)) != len(names):
            return False
        results = {result.name: result.passed for result in evaluation.constraints}
        names_to_require = self.required_constraints + tuple(
            constraint.name for constraint in self.metric_constraints
        )
        return all(results.get(name, False) for name in names_to_require)


@dataclass(frozen=True)
class Candidate:
    """A frozen proposal produced against one champion generation."""

    model: base.Classifier
    parent_generation: int
    hypothesis: str


@dataclass(frozen=True)
class ResearchSnapshot:
    """Detached information made available for one reflection."""

    champion: base.Classifier
    generation: int
    observations_seen: int
    current_score: float
    objective: Objective
    history: tuple[Observation, ...]
    context: Any = None
