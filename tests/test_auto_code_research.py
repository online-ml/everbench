from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from river import compose, ensemble, metrics

from everbench.auto import Objective, TemporalObservation, temporal_split
from everbench.auto.code_execution import (
    build_candidate_model,
    evaluate_candidate_source,
    validate_candidate_source,
)
from everbench.auto.code_researcher import OpenAICodeResearcher, ResearchRequest
from everbench.auto.dataset import PreparedTemporalData

ENSEMBLE_SOURCE = """from river import compose, ensemble, linear_model

def features(event):
    return {"nested": float((event.get("complete") or {}).get("nested", 0))}

def build_model():
    return compose.FuncTransformer(features) | ensemble.BaggingClassifier(
        linear_model.LogisticRegression(), n_models=2, seed=42
    )
"""

STACKING_SOURCE = """from river import compose, ensemble, linear_model, naive_bayes

def features(event):
    return {"nested": float((event.get("complete") or {}).get("nested", 0))}

def build_model():
    return ensemble.StackingClassifier(
        models=[
            compose.FuncTransformer(features) | linear_model.LogisticRegression(),
            compose.FuncTransformer(features) | naive_bayes.GaussianNB(),
        ],
        meta_classifier=linear_model.LogisticRegression(),
    )
"""


def test_candidate_program_can_build_arbitrary_river_ensemble() -> None:
    model = build_candidate_model(
        ENSEMBLE_SOURCE,
        timeout_seconds=10,
        max_source_bytes=10_000,
        max_output_bytes=2_000_000,
    )

    assert isinstance(model, compose.Pipeline)
    assert len(model) == 2
    assert isinstance(model[1], ensemble.BaggingClassifier)
    assert len(model[1]) == 2


def test_candidate_program_can_build_a_stacking_meta_model() -> None:
    model = build_candidate_model(
        STACKING_SOURCE,
        timeout_seconds=10,
        max_source_bytes=10_000,
        max_output_bytes=2_000_000,
    )

    assert isinstance(model, ensemble.StackingClassifier)
    assert len(model.models) == 2


def test_wiki_seed_is_a_self_contained_raw_event_model() -> None:
    source = (Path(__file__).parents[1] / "tasks/wiki_liftwing/auto/candidate.py").read_text()
    model = build_candidate_model(
        source,
        timeout_seconds=10,
        max_source_bytes=100_000,
        max_output_bytes=2_000_000,
    )
    event = {
        "user": "~2026-123",
        "length": {"old": 1_000, "new": 10},
        "comment": "undid vandalism",
        "timestamp": 1_700_000_000,
        "unused_nested_payload": {"remains": "available"},
    }

    features = model[0].transform_one(event)
    assert features["anonymous"] == 1.0
    assert features["blanking"] == 1.0
    assert 0.0 <= model.predict_proba_one(event)[True] <= 1.0


def test_candidate_program_cannot_access_system_or_files() -> None:
    with pytest.raises(ValueError, match="import is not allowed"):
        validate_candidate_source("import os\ndef build_model(): pass", 10_000)
    with pytest.raises(ValueError, match="call is not allowed"):
        validate_candidate_source("def build_model():\n    return open('/etc/passwd')", 10_000)


def test_candidate_source_is_causally_evaluated_in_subprocess() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = tuple(
        TemporalObservation(
            str(index),
            index,
            {"complete": {"nested": index}},
            index % 2,
            start + timedelta(seconds=index),
            start + timedelta(seconds=index + 2),
        )
        for index in range(8)
    )
    outcome = evaluate_candidate_source(
        ENSEMBLE_SOURCE,
        build_candidate_model(
            ENSEMBLE_SOURCE,
            timeout_seconds=10,
            max_source_bytes=10_000,
            max_output_bytes=2_000_000,
        ),
        temporal_split(rows, promotion_observations=3, min_research_observations=5),
        Objective(metrics.ROCAUC(), min_observations=3),
        max_prediction_time_ratio=10.0,
        timeout_seconds=10,
        max_source_bytes=10_000,
        max_output_bytes=2_000_000,
    )

    assert outcome.evaluation.observations == 3
    assert outcome.checkpoint_event_sequence == 7


def test_candidate_subprocess_streams_a_prepared_temporal_split() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = tuple(
        TemporalObservation(
            str(index),
            index,
            {"complete": {"nested": index}},
            index % 2,
            start + timedelta(seconds=index),
            start + timedelta(seconds=index + 2),
        )
        for index in range(8)
    )
    champion = build_candidate_model(
        ENSEMBLE_SOURCE,
        timeout_seconds=10,
        max_source_bytes=10_000,
        max_output_bytes=2_000_000,
    )

    with PreparedTemporalData.from_observations(rows) as prepared:
        outcome = evaluate_candidate_source(
            ENSEMBLE_SOURCE,
            champion,
            prepared.split(promotion_observations=3, min_research_observations=5),
            Objective(metrics.ROCAUC(), min_observations=3),
            max_prediction_time_ratio=10.0,
            timeout_seconds=10,
            max_source_bytes=10_000,
            max_output_bytes=2_000_000,
        )

    assert outcome.evaluation.observations == 3
    assert outcome.checkpoint_event_sequence == 7


class FakeResponses:
    def __init__(self) -> None:
        self.calls = [
            SimpleNamespace(
                type="function_call",
                name="evaluate_candidate",
                call_id="evaluate",
                arguments=__import__("json").dumps(
                    {"source": ENSEMBLE_SOURCE, "hypothesis": "average two online linear models"}
                ),
            ),
            SimpleNamespace(type="function_call", name="finish", call_id="finish", arguments="{}"),
        ]
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(output=[self.calls.pop(0)])


def test_code_researcher_edits_evaluates_and_freezes_source() -> None:
    responses = FakeResponses()
    researcher = OpenAICodeResearcher(client=SimpleNamespace(responses=responses), max_experiments=2)
    request = ResearchRequest(
        context={"problem_description": "test"},
        research_summary={"observations": 100, "payload_schema": {"nested.value": ["int"]}},
        objective={"primary_metric": "ROCAUC"},
        current_source="from river import dummy\ndef build_model(): return dummy.PriorClassifier()",
    )
    evaluated: list[str] = []

    proposal = researcher.research(
        request,
        lambda source: evaluated.append(source) or {"candidate_score": 0.75},
    )

    assert proposal.source == ENSEMBLE_SOURCE
    assert proposal.hypothesis == "average two online linear models"
    assert evaluated == [ENSEMBLE_SOURCE]
    assert all(call["store"] is False for call in responses.requests)
    assert all(call["parallel_tool_calls"] is False for call in responses.requests)


def test_code_researcher_prefers_a_constraint_passing_candidate() -> None:
    responses = FakeResponses()
    responses.calls = [
        SimpleNamespace(
            type="function_call",
            name="evaluate_candidate",
            call_id="fast",
            arguments=__import__("json").dumps({"source": ENSEMBLE_SOURCE, "hypothesis": "faster primary gain"}),
        ),
        SimpleNamespace(
            type="function_call",
            name="evaluate_candidate",
            call_id="safe",
            arguments=__import__("json").dumps({"source": STACKING_SOURCE, "hypothesis": "respect the guard"}),
        ),
        SimpleNamespace(type="function_call", name="finish", call_id="finish", arguments="{}"),
    ]
    researcher = OpenAICodeResearcher(client=SimpleNamespace(responses=responses), max_experiments=3)
    request = ResearchRequest({}, {"primary_metric": "ROCAUC"}, {}, ENSEMBLE_SOURCE)

    proposal = researcher.research(
        request,
        lambda source: {
            "improvement": 0.2 if source == ENSEMBLE_SOURCE else 0.1,
            "constraints": [{"name": "latency", "passed": source == STACKING_SOURCE}],
        },
    )

    assert proposal.source == STACKING_SOURCE
    assert proposal.hypothesis == "respect the guard"
