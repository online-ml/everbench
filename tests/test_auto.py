from __future__ import annotations

from typing import Any

import pytest
from river import base, metrics

from everbench.auto import (
    AutoClassifier,
    Candidate,
    ConstraintResult,
    Evaluation,
    MetricConstraint,
    Objective,
)


class CountingClassifier(base.Classifier):
    def __init__(self, positive_probability: float = 0.5) -> None:
        self.positive_probability = positive_probability
        self.examples = 0

    def predict_proba_one(
        self, x: dict[base.typing.FeatureName, Any], **kwargs: Any
    ) -> dict[base.typing.ClfTarget, float]:
        del x, kwargs
        return {0: 1.0 - self.positive_probability, 1: self.positive_probability}

    def learn_one(self, x: dict[Any, Any], y: Any, w: float = 1.0) -> None:
        del x, y, w
        self.examples += 1


class FailingClassifier(CountingClassifier):
    def learn_one(self, x: dict[Any, Any], y: Any, w: float = 1.0) -> None:
        del x, y, w
        raise RuntimeError("learning failed")


def test_auto_classifier_delegates_without_retaining_a_research_dataset() -> None:
    model = AutoClassifier(
        model=CountingClassifier(positive_probability=0.75),
        objective=Objective(metrics.LogLoss()),
    )
    assert model.predict_proba_one({"value": 1})[1] == 0.75
    model.learn_one({"value": 1}, 1)
    assert isinstance(model.model, CountingClassifier)
    assert model.model.examples == 1
    assert not hasattr(model, "_history")


def test_only_serving_state_is_serialized() -> None:
    model = AutoClassifier(CountingClassifier(), Objective(metrics.Accuracy()))
    model.__dict__["_history"] = [{"raw": "event"}]
    model.__dict__["history_capacity"] = 5_000

    restored = __import__("pickle").loads(__import__("pickle").dumps(model))

    assert not hasattr(restored, "_history")
    assert not hasattr(restored, "history_capacity")


def test_failed_learning_does_not_update_the_serving_metric() -> None:
    model = AutoClassifier(FailingClassifier(), Objective(metrics.LogLoss()))

    with pytest.raises(RuntimeError, match="learning failed"):
        model.learn_one({"value": 1}, 1)

    assert model.score == 0.0


def test_context_and_objective_are_detached_from_the_caller() -> None:
    source_metric = metrics.LogLoss()
    source_context = {"problem_description": "Predict a delayed outcome", "labels": [0, 1]}
    model = AutoClassifier(CountingClassifier(), Objective(source_metric), context=source_context)
    source_context["labels"].append(2)
    model.learn_one({"value": 1}, 1)

    objective = model.objective
    objective.metric.update(1, {0: 0.9, 1: 0.1})
    context = model.context
    context["labels"].append(3)
    source_metric.update(1, {0: 0.9, 1: 0.1})

    assert isinstance(model.model, CountingClassifier)
    assert model.model.examples == 1
    assert model.objective.metric.get() == 0.0
    assert model.context == {"problem_description": "Predict a delayed outcome", "labels": [0, 1]}


def test_consider_promotes_only_fresh_candidates_with_sufficient_evidence() -> None:
    objective = Objective(
        metrics.Accuracy(),
        min_improvement=0.1,
        min_observations=10,
        required_constraints=("latency",),
    )
    model = AutoClassifier(CountingClassifier(positive_probability=0.25), objective)
    candidate_model = CountingClassifier(positive_probability=0.8)
    candidate_model.learn_one({}, 1)
    candidate = Candidate(candidate_model, parent_generation=0, hypothesis="favor the positive class")

    assert not model.consider(
        candidate,
        Evaluation(
            champion_score=0.5,
            candidate_score=0.7,
            observations=9,
            constraints=(ConstraintResult("latency", True),),
        ),
    )
    assert not model.consider(
        candidate,
        Evaluation(champion_score=0.5, candidate_score=0.7, observations=10),
    )
    assert model.generation == 0

    assert model.consider(
        candidate,
        Evaluation(
            champion_score=0.5,
            candidate_score=0.7,
            observations=10,
            constraints=(ConstraintResult("latency", True),),
        ),
    )
    assert model.generation == 1
    assert isinstance(model.model, CountingClassifier)
    assert model.model.positive_probability == 0.8
    assert model.model.examples == 1
    assert model.score == 0.0

    candidate_model.learn_one({}, 1)
    assert model.model.examples == 1

    with pytest.raises(ValueError, match="targets generation 0, current generation is 1"):
        model.consider(
            candidate,
            Evaluation(
                champion_score=0.5,
                candidate_score=0.7,
                observations=10,
                constraints=(ConstraintResult("latency", True),),
            ),
        )


def test_objective_uses_the_metric_direction_and_rejects_non_finite_scores() -> None:
    minimize = Objective(metrics.LogLoss(), min_improvement=0.05)
    assert minimize.accepts(Evaluation(champion_score=0.5, candidate_score=0.4, observations=1))
    assert not minimize.accepts(Evaluation(champion_score=0.5, candidate_score=0.46, observations=1))
    assert not Objective(metrics.LogLoss()).accepts(Evaluation(champion_score=0.5, candidate_score=0.5, observations=1))
    assert not minimize.accepts(Evaluation(champion_score=float("nan"), candidate_score=0.4, observations=1))

    maximize = Objective(metrics.Accuracy(), min_improvement=0.05)
    assert maximize.accepts(Evaluation(champion_score=0.5, candidate_score=0.6, observations=1))
    assert not maximize.accepts(Evaluation(champion_score=0.5, candidate_score=0.54, observations=1))


def test_objective_rejects_ambiguous_constraint_results() -> None:
    objective = Objective(metrics.Accuracy(), required_constraints=("latency",))
    evaluation = Evaluation(
        champion_score=0.5,
        candidate_score=0.6,
        observations=1,
        constraints=(ConstraintResult("latency", False), ConstraintResult("latency", True)),
    )

    assert not objective.accepts(evaluation)


def test_secondary_metric_constraint_is_immutable_and_required() -> None:
    objective = Objective(
        metrics.LogLoss(),
        metric_constraints=(MetricConstraint("auc", metrics.ROCAUC(), max_regression=0.01),),
    )
    assert objective.accepts(
        Evaluation(
            champion_score=0.5,
            candidate_score=0.4,
            observations=1,
            constraints=(ConstraintResult("auc", True),),
        )
    )
    assert not objective.accepts(Evaluation(champion_score=0.5, candidate_score=0.4, observations=1))
