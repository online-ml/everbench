"""Tests for task discovery used by the all-task worker."""

from __future__ import annotations

from pathlib import Path

import pytest

from everbench.sources import SSESource
from everbench.tasks import discover_tasks, load_task

TASK = """
from river import metrics
from everbench.sources import SSESource
from everbench.tasks import TaskDefinition

def decode(*, event): return ()
TASK = TaskDefinition(
    TASK_NAME="{name}", PROBLEM_TYPE="binary_classification",
    METRICS=(metrics.ROCAUC(),), DESCRIPTION_HTML="test",
    sources=(SSESource(name="events", url="https://example.test/events", decode=decode),),
)
"""


def write_task(*, root: Path, directory_name: str, task_name: str) -> None:
    task_directory = root / directory_name
    task_directory.mkdir()
    (task_directory / "task.py").write_text(TASK.format(name=task_name))


def test_loads_task_directories_in_path_order(*, tmp_path: Path) -> None:
    for name in ("beta", "alpha"):
        write_task(root=tmp_path, directory_name=name, task_name=name)
    examples = tmp_path / "alpha" / "examples"
    examples.mkdir()
    (examples / "not_a_task.py").write_text("raise AssertionError('must not load')")

    tasks = discover_tasks(directory=tmp_path)

    assert [task.TASK_NAME for task in tasks] == ["alpha", "beta"]


def test_rejects_duplicate_task_names(*, tmp_path: Path) -> None:
    for directory_name in ("one", "two"):
        write_task(root=tmp_path, directory_name=directory_name, task_name="same")

    with pytest.raises(ValueError, match="unique"):
        discover_tasks(directory=tmp_path)


def test_discovers_only_explicitly_selected_tasks(*, tmp_path: Path) -> None:
    for name in ("production", "dummy"):
        write_task(root=tmp_path, directory_name=name, task_name=name)

    tasks = discover_tasks(directory=tmp_path, task_names=["production"])

    assert [task.TASK_NAME for task in tasks] == ["production"]


def test_rejects_unknown_selected_tasks(*, tmp_path: Path) -> None:
    write_task(root=tmp_path, directory_name="production", task_name="production")

    with pytest.raises(LookupError, match="missing"):
        discover_tasks(directory=tmp_path, task_names=["missing"])


def test_wiki_task_accepts_a_live_reverted_tag_event() -> None:
    task = load_task(path="tasks/wiki_liftwing/task.py")
    event = {
        "database": "enwiki",
        "rev_id": 123,
        "tags": ["visualeditor", "mw-reverted"],
        "prior_state": {"tags": ["visualeditor"]},
        "meta": {"dt": "2026-09-03T12:34:56Z"},
    }

    assert task.label_policy is not None
    assert task.label_policy.delay_seconds == 48 * 60 * 60
    source = task.sources[1]
    assert isinstance(source, SSESource)
    label = next(iter(source.decode(event=event)))
    from everbench.records import LabelInput

    assert isinstance(label, LabelInput)
    assert (label.event_id, label.y, label.reason) == ("enwiki:123", 1, "mw-reverted")
    assert label.available_at.timestamp() == 1_788_438_896.0
