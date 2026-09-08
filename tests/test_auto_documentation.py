from __future__ import annotations

import ast
from datetime import UTC, datetime

import pytest

from everbench.auto.documentation import documented_source
from everbench.schema import AutoExperiment


@pytest.mark.parametrize("original", ["", '"""Existing model explanation: café."""\n', '""""""\n'])
def test_research_docstring_preserves_valid_source_and_original_explanation(original: str) -> None:
    source = original + "from __future__ import annotations\n\ndef build_model():\n    return None\n"
    hypothesis = 'Try """quoted""" features and C:\\new weights.'

    result = documented_source(source, {"generation": 3, "hypothesis": hypothesis}, [], {"promoted": 3})

    compile(result, "candidate.py", "exec")
    doc = ast.get_docstring(ast.parse(result))
    assert doc is not None
    assert "generation 3" in doc
    assert hypothesis in doc
    assert "3 promoted" in doc
    assert "No research rounds have run yet" in doc
    if "Existing" in original:
        assert "Existing model explanation: café." in doc
    assert result.endswith(source[len(original) :])


def test_bootstrap_notes_do_not_claim_a_successful_experiment() -> None:
    result = documented_source("def build_model(): pass\n", {"generation": 0}, [], {})

    assert "generation 0" in result
    assert "no promoted changes yet" in result
    assert "0 promoted, 0 rejected, 0 failed, 0 running" in result


def test_research_notes_distinguish_active_rejected_failed_and_promoted_rounds() -> None:
    def experiment(status: str, **kwargs) -> AutoExperiment:
        return AutoExperiment(
            started_at=datetime(2026, 9, 8, 12, tzinfo=UTC),
            status=status,
            parent_generation=2,
            researcher="test-researcher",
            **kwargs,
        )

    evaluation = {
        "metric": "LogLoss",
        "champion_score": 0.4,
        "candidate_score": 0.3,
        "improvement": 0.1,
        "observations": 2000,
        "constraints": [{"name": "latency", "passed": False, "detail": "too slow"}],
    }
    result = documented_source(
        "def build_model(): pass\n",
        {"generation": 2, "hypothesis": "Use adaptive weights."},
        [
            experiment("running"),
            experiment("failed", error="ValueError: invalid candidate"),
            experiment("rejected", hypothesis="Try a larger ensemble.", evaluation=evaluation),
            experiment("promoted", hypothesis="Use adaptive weights."),
        ],
        {"promoted": 2, "rejected": 12, "failed": 1, "running": 1},
    )

    assert "2 promoted, 12 rejected, 1 failed, 1 running" in result
    assert "test-researcher is exploring candidates against generation 2" in result
    assert "Try a larger ensemble" in result
    assert "champion 0.400000, candidate 0.300000" in result
    assert "+0.100000 on 2,000 sealed observations" in " ".join(result.split())
    assert "Failed constraint latency: too slow" in result
    assert "ValueError: invalid candidate" in result
    assert "promoted, generation 2 → 3" in result
    assert "not live leaderboard scores" in result
