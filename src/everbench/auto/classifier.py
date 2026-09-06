"""A River-compatible classifier with controlled champion replacement."""

from __future__ import annotations

import copy
from collections import deque
from typing import Any

from river import base

from everbench.auto.research import Candidate, Evaluation, Objective, Observation, ResearchSnapshot


class AutoClassifier(base.Classifier):
    """Wrap a River classifier with research snapshots and guarded promotion.

    This class never invokes an agent, starts background work, or retains raw
    observations unless a positive history capacity is explicitly supplied. An external
    runner is responsible for proposing candidates and collecting sealed
    evaluation evidence.

    Parameters
    ----------
    model
        The initial champion.
    objective
        Owner-defined promotion criteria. The contained metric is cloned before
        use so callers and research snapshots cannot alter the live objective.
    history_capacity
        Number of recent raw observations to retain in memory and in pickles.
        Zero, the default, retains nothing.
    context
        Optional problem description or arbitrary structured context supplied
        to the research agent. It is copied at initialization and snapshot time.
    """

    def __init__(
        self,
        model: base.Classifier,
        objective: Objective,
        history_capacity: int = 0,
        context: Any = None,
    ) -> None:
        if history_capacity < 0:
            raise ValueError("history_capacity cannot be negative")
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
        self.history_capacity = history_capacity
        self._history: deque[Observation] | None = deque(maxlen=history_capacity) if history_capacity else None
        self._context = copy.deepcopy(context)
        self._metric = self._objective.fresh_metric()
        self._generation = 0
        self._observations_seen = 0

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

    def predict_one(self, x: dict[Any, Any], **kwargs: Any) -> Any:
        return self.model.predict_one(x, **kwargs)

    def predict_proba_one(self, x: dict[Any, Any], **kwargs: Any) -> dict[Any, float]:
        return self.model.predict_proba_one(x, **kwargs)

    def _prediction_for_metric(self, x: dict[Any, Any]) -> Any:
        if self._metric.requires_labels:
            return self.model.predict_one(x)
        return self.model.predict_proba_one(x)

    def learn_one(self, x: dict[Any, Any], y: Any, **kwargs: Any) -> None:
        prediction = self._prediction_for_metric(x)
        self.model.learn_one(x, y, **kwargs)
        weight = kwargs.get("w", 1.0)
        self._metric.update(y, prediction, w=weight)
        self._observations_seen += 1
        if self._history is not None:
            self._history.append(
                copy.deepcopy(
                    Observation(
                        sequence=self._observations_seen,
                        x=x,
                        y=y,
                        prediction=prediction,
                        learn_kwargs=kwargs,
                    )
                )
            )

    def research_snapshot(self) -> ResearchSnapshot:
        """Return detached state that an external runner may expose to an agent."""
        history = copy.deepcopy(tuple(self._history)) if self._history is not None else ()
        return ResearchSnapshot(
            champion=self.model.clone(include_attributes=True),
            generation=self._generation,
            observations_seen=self._observations_seen,
            current_score=self.score,
            objective=self.objective,
            history=history,
            context=self.context,
        )

    def consider(self, candidate: Candidate, evaluation: Evaluation) -> bool:
        """Promote a candidate when fresh sealed evidence satisfies the objective."""
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
        if not self._objective.accepts(evaluation):
            return False
        self.model = candidate.model.clone(include_attributes=True)
        self._generation += 1
        self._metric = self._objective.fresh_metric()
        return True
