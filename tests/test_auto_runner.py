from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from river import base, metrics

from everbench.auto import (
    ArchiveExample,
    Objective,
    progressive_validate,
)
from everbench.auto.code_researcher import summarize_research
from everbench.auto.dataset import ArchiveWeek
from everbench.auto.service import _with_model_size_constraint
from everbench.schema import ArchiveManifest


class LabelCountClassifier(base.Classifier):
    def __init__(self) -> None:
        self.learned = 0

    def predict_proba_one(  # noqa: PLR0917 -- external positional protocol
        self, x: dict[base.typing.FeatureName, Any], **kwargs: Any
    ) -> dict[base.typing.ClfTarget, float]:
        del x, kwargs
        positive = 0.1 if self.learned == 0 else 0.9
        return {0: 1 - positive, 1: positive}

    def learn_one(self, x: dict[Any, Any], y: Any) -> None:  # noqa: PLR0917 -- external positional protocol
        del x, y
        self.learned += 1


def observations(*, count: int = 6) -> tuple[ArchiveExample, ...]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return tuple(
        ArchiveExample(
            event_id=str(index),
            sequence=index,
            payload={"value": index},
            target=1,
            available_at=start + timedelta(seconds=index),
            resolved_at=start + timedelta(seconds=index + 10),
        )
        for index in range(count)
    )


def _archive(*, path: Path, rows: list[dict[str, Any]], suffix: str) -> ArchiveManifest:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return ArchiveManifest(
        content_sha256=suffix * 64,
        task_name="test",
        event_date=date(2026, 1, 5),
        path=str(path),
        row_count=len(rows),
        byte_size=path.stat().st_size,
    )


def test_archive_week_clamps_labels_that_arrived_before_the_event(*, tmp_path: Path) -> None:
    later = {
        "event_id": "later",
        "event_sequence": 2,
        "event_available_at": "2026-01-05T00:00:02+00:00",
        "payload_json": '{"value":2}',
        "label": 0,
        "label_available_at": "2026-01-05T00:00:03+00:00",
    }
    earlier = {
        "event_id": "earlier",
        "event_sequence": 1,
        "event_available_at": "2026-01-05T00:00:01+00:00",
        "payload_json": '{"value":1}',
        "label": 1,
        # Inbox labels may precede their matching event.
        "label_available_at": "2026-01-05T00:00:00+00:00",
    }
    manifest = _archive(path=tmp_path / "week.parquet", rows=[earlier, later], suffix="a")

    with ArchiveWeek.open(manifest=manifest) as prepared:
        rows = tuple(prepared)

    assert [row.event_id for row in rows] == ["earlier", "later"]
    assert rows[0].resolved_at == rows[0].available_at


def test_research_summary_never_copies_raw_examples() -> None:
    rows = observations()

    summary = summarize_research(observations=rows)
    assert "raw_examples" not in summary
    assert summary["payload_schema"] == {"$": ["dict"], "value": ["int"]}


def test_evaluation_preserves_delayed_feedback() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = (
        ArchiveExample(
            event_id="research",
            sequence=1,
            payload={},
            target=1,
            available_at=start,
            resolved_at=start + timedelta(seconds=3),
        ),
        ArchiveExample(
            event_id="promotion",
            sequence=2,
            payload={},
            target=1,
            available_at=start + timedelta(seconds=1),
            resolved_at=start + timedelta(seconds=4),
        ),
    )
    outcome = progressive_validate(
        champion=LabelCountClassifier(),
        candidate=LabelCountClassifier(),
        observations=rows,
        objective=Objective(metric=metrics.LogLoss()),
    )

    assert outcome.evaluation.candidate_score == pytest.approx(-__import__("math").log(0.1))
    assert outcome.evaluation.champion_score == outcome.evaluation.candidate_score
    assert isinstance(outcome.trained_candidate, LabelCountClassifier)
    assert outcome.trained_candidate.learned == 2


def test_serialized_model_size_is_required_promotion_evidence() -> None:
    objective = Objective(
        metric=metrics.Accuracy(),
        required_constraints=("serialized_model_size",),
    )
    outcome = progressive_validate(
        champion=LabelCountClassifier(),
        candidate=LabelCountClassifier(),
        observations=observations(),
        objective=objective,
    )
    outcome = replace(
        outcome,
        evaluation=replace(outcome.evaluation, champion_score=0.0, candidate_score=1.0),
    )

    constrained = _with_model_size_constraint(outcome=outcome, max_bytes=1)
    within_limit = _with_model_size_constraint(outcome=outcome, max_bytes=1_000_000)

    size = constrained.evaluation.constraints[-1]
    assert size.name == "serialized_model_size"
    assert not size.passed
    assert not objective.accepts(evaluation=constrained.evaluation)
    assert objective.accepts(evaluation=within_limit.evaluation)
