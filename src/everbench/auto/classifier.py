"""A River-compatible classifier with controlled champion replacement."""

from __future__ import annotations

import copy
from typing import Any

from river import base

from everbench.auto.research import Candidate, Evaluation, Objective


class AutoClassifier(base.Classifier):
    """Wrap a serving classifier with an objective and guarded promotion.

    This class never invokes an agent, starts background work, or retains a
    research dataset. The external runner gets research observations from
    immutable archives.

    Parameters
    ----------
    model
        The initial champion.
    objective
        Owner-defined promotion criteria. The contained metric is cloned before
        use so callers and research snapshots cannot alter the live objective.
    context
        Optional problem description or arbitrary structured context supplied
        to the research agent. It is copied at initialization and snapshot time.
    """

    def __init__(self, *, model: base.Classifier, objective: Objective, context: Any = None) -> None:
        if not objective.metric.works_with(model):
            raise ValueError(f"{type(objective.metric).__name__} does not work with {type(model).__name__}")
        incompatible = [
            type(constraint.metric).__name__
            for constraint in objective.metric_constraints
            if not constraint.metric.works_with(model)
        ]
        if incompatible:
            raise ValueError(f"secondary metrics do not work with {type(model).__name__}: {', '.join(incompatible)}")
        self.model = model
        self._objective = objective.copy()
        self._context = copy.deepcopy(context)
        self._metric = self._objective.fresh_metric()
        self._generation = 0

    @property
    def objective(self) -> Objective:
        """Return a detached copy of the owner-defined objective."""
        return self._objective.copy()

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def context(self) -> Any:
        """Return detached owner-supplied research context."""
        return copy.deepcopy(self._context)

    @property
    def score(self) -> float:
        return self._metric.get()

    @property
    def _multiclass(self) -> bool:
        return self.model._multiclass

    def predict_one(self, x: dict[Any, Any], **kwargs: Any) -> Any:  # noqa: PLR0917 -- external positional protocol
        return self.model.predict_one(x, **kwargs)

    def predict_proba_one(self, x: dict[Any, Any], **kwargs: Any) -> dict[Any, float]:  # noqa: PLR0917 -- external positional protocol
        return self.model.predict_proba_one(x, **kwargs)

    def _prediction_for_metric(self, *, x: dict[Any, Any]) -> Any:
        if self._metric.requires_labels:
            return self.model.predict_one(x)
        return self.model.predict_proba_one(x)

    def learn_one(self, x: dict[Any, Any], y: Any, **kwargs: Any) -> None:  # noqa: PLR0917 -- external positional protocol
        prediction = self._prediction_for_metric(x=x)
        self.model.learn_one(x, y, **kwargs)
        weight = kwargs.get("w", 1.0)
        self._metric.update(y, prediction, w=weight)

    def __getstate__(self) -> dict[str, Any]:
        """Persist only serving state; research observations belong to archives."""
        return {
            "model": self.model,
            "_objective": self._objective,
            "_context": self._context,
            "_metric": self._metric,
            "_generation": self._generation,
        }

    def consider(self, *, candidate: Candidate, evaluation: Evaluation) -> bool:
        """Promote a candidate when its weekly comparison satisfies the objective."""
        if candidate.parent_generation != self._generation:
            raise ValueError(
                f"candidate targets generation {candidate.parent_generation}, current generation is {self._generation}"
            )
        if not self._objective.metric.works_with(candidate.model):
            raise ValueError(
                f"{type(self._objective.metric).__name__} does not work with {type(candidate.model).__name__}"
            )
        if any(not constraint.metric.works_with(candidate.model) for constraint in self._objective.metric_constraints):
            raise ValueError(f"secondary metric does not work with {type(candidate.model).__name__}")
        if not self._objective.accepts(evaluation=evaluation):
            return False
        self.model = candidate.model.clone(include_attributes=True)
        self._generation += 1
        self._metric = self._objective.fresh_metric()
        return True
