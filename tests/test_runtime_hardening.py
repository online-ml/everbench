from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sqlalchemy.orm import Session

from everbench import archive, archive_store, artifacts, event_store, model_store
from everbench.api import task_source_url, validation_examples
from everbench.batching import TimedBatch
from everbench.collectors import CollectorBatch, StreamCursorState
from everbench.learner import _load_model
from everbench.models import PickledModel, prediction_for, validate_model
from everbench.records import LabelledExample
from everbench.tasks import TaskDefinition


class ConstantModel:
    def predict_one(self, *, event_id: str, event: dict[str, Any]) -> float:
        del event_id, event
        return 0.5


class EventAwareModel:
    def predict_proba_one(self, *, event_id: str, event: dict[str, Any]) -> dict[bool, float]:
        del event
        return {False: float(event_id != "event-1"), True: float(event_id == "event-1")}


class LegacyEventModel:
    def predict_event(self, *, event_id: str, event: dict[str, Any]) -> float:
        del event_id, event
        return 0.5


class MulticlassModel:
    def predict_proba_one(self, *, event_id: str, event: dict[str, Any]) -> dict[str, float]:
        del event_id, event
        return {"first": 0.2, "second": 0.8}


class ScoreOnlyModel:
    def score_one(self, *, event_id: str, event: dict[str, Any]) -> float:
        del event_id
        return float(event["score"])


class BrokenProbabilityModel:
    def predict_proba_one(self, *, event_id: str, event: dict[str, Any]) -> dict[bool, float]:
        del event_id, event
        raise AttributeError("internal typo")

    def predict_one(self, *, event_id: str, event: dict[str, Any]) -> float:
        del event_id, event
        return 0.25


@pytest.fixture
def signing_key(*, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVERBENCH_MODEL_SIGNING_KEY", "test-signing-key")


def test_model_load_uses_the_task_name_for_snapshots(*, signing_key: None, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = artifacts.dumps(model=ConstantModel())
    artifact = SimpleNamespace(payload=payload, signature=artifacts.sign(payload=payload))
    registration = SimpleNamespace(model_id="constant", artifact_id="artifact")
    task = SimpleNamespace(TASK_NAME="example")
    monkeypatch.setattr(model_store, "latest_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_store, "artifact", lambda *args, **kwargs: artifact)

    model, snapshot = _load_model(
        session=cast(Session, None), task=cast(TaskDefinition, task), registration=registration
    )

    assert snapshot is None
    assert model.predict_one(event_id="event", event={}) == 0.5


def test_task_source_url_links_to_the_checked_in_definition(*, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task = SimpleNamespace(__file__=Path("tasks/dummy/task.py").resolve())
    monkeypatch.chdir(tmp_path)

    assert task_source_url(task=task) == "https://github.com/online-ml/everbench/blob/main/tasks/dummy/task.py"


def test_idle_batch_flushes_without_another_source_item() -> None:
    flushed: list[list[str]] = []
    batch = TimedBatch(max_items=100, max_age_seconds=0.01, flush=lambda items: flushed.append(list(items)))
    batch.add(item="one")
    time.sleep(0.02)

    assert batch.flush_if_due()
    assert flushed == [["one"]]


def test_filtered_stream_messages_checkpoint_without_filling_the_ingest_batch() -> None:
    state = StreamCursorState(value="initial")
    flushed: list[tuple[list[str], str | None]] = []
    checkpoints: list[str] = []
    batch = CollectorBatch(
        cursor_state=state,
        flush=lambda items, cursor: flushed.append((list(items), cursor)),
        checkpoint=lambda cursor: checkpoints.append(cursor),
    )

    batch.observe(item=None, cursor="filtered-1")
    batch.observe(item=None, cursor="filtered-2")

    assert flushed == []
    assert checkpoints == []
    batch._checkpointed_at = 0
    batch.tick()
    assert checkpoints == ["filtered-2"]
    assert state.value == "filtered-2"

    batch.observe(item="accepted", cursor="accepted-3")
    assert batch.flush()
    assert flushed == [(["accepted"], "accepted-3")]
    assert state.value == "accepted-3"


def test_upload_validation_uses_recent_postgres_labels_before_archives(*, monkeypatch: pytest.MonkeyPatch) -> None:
    recent = [LabelledExample(event_id="one", payload={"x": 1.0}, target=1)] * 5
    archived = False

    def read_archives(**kwargs):
        nonlocal archived
        archived = True
        return []

    monkeypatch.setattr(event_store, "latest_labelled_examples", lambda *args, **kwargs: recent)
    monkeypatch.setattr(archive, "latest_labelled_examples", read_archives)

    assert validation_examples(session=cast(Session, SimpleNamespace()), task_name="task") == recent
    assert not archived


def test_upload_validation_combines_archived_and_recent_labels(*, monkeypatch: pytest.MonkeyPatch) -> None:
    recent = [LabelledExample(event_id="recent", payload={"x": 1.0}, target=1)] * 2
    archived_rows = [LabelledExample(event_id="old", payload={"x": 0.0}, target=0)] * 3
    monkeypatch.setattr(event_store, "latest_labelled_examples", lambda *args, **kwargs: recent)
    monkeypatch.setattr(archive_store, "task_archives", lambda *args, **kwargs: ["manifest"])
    monkeypatch.setattr(archive, "latest_labelled_examples", lambda *args, **kwargs: archived_rows)

    examples = validation_examples(session=cast(Session, SimpleNamespace()), task_name="task")

    assert examples == archived_rows + recent


def test_upload_validation_accepts_a_task_without_labels() -> None:
    task = SimpleNamespace(PROBLEM_TYPE="binary_classification", METRICS=())

    assert (
        validate_model(
            task=cast(TaskDefinition, task),
            candidate=PickledModel(model_id="constant", model=ConstantModel()),
            examples=[],
        )
        == 0
    )


def test_prediction_passes_event_id_to_standard_prediction_method() -> None:
    task = cast(TaskDefinition, SimpleNamespace(PROBLEM_TYPE="binary_classification"))
    model = PickledModel(model_id="event-aware", model=EventAwareModel())

    assert prediction_for(task=task, model=model, event_id="event-1", event={"x": 1.0}) == 1.0
    assert prediction_for(task=task, model=model, event_id="event-2", event={"x": 1.0}) == 0.0


def test_multiclass_prediction_keeps_probability_mapping() -> None:
    task = cast(TaskDefinition, SimpleNamespace(PROBLEM_TYPE="multiclass_classification"))
    model = PickledModel(model_id="multiclass", model=MulticlassModel())

    assert prediction_for(task=task, model=model, event_id="event-1", event={"x": 1.0}) == {"first": 0.2, "second": 0.8}


def test_legacy_predict_event_protocol_is_rejected() -> None:
    with pytest.raises(TypeError, match="predict_one"):
        PickledModel(model_id="legacy", model=LegacyEventModel())


def test_score_only_anomaly_model_is_accepted() -> None:
    task = cast(TaskDefinition, SimpleNamespace(PROBLEM_TYPE="anomaly_detection"))
    model = PickledModel(model_id="anomaly", model=ScoreOnlyModel())

    assert prediction_for(task=task, model=model, event_id="event-1", event={"score": 0.3}) == 0.3


def test_internal_attribute_error_does_not_trigger_prediction_fallback() -> None:
    task = cast(TaskDefinition, SimpleNamespace(PROBLEM_TYPE="binary_classification"))
    model = PickledModel(model_id="broken-probability", model=BrokenProbabilityModel())

    with pytest.raises(AttributeError, match="internal typo"):
        prediction_for(task=task, model=model, event_id="event-1", event={})
