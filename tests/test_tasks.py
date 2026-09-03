"""Tests for task discovery used by the all-task worker."""

from __future__ import annotations

from pathlib import Path

import pytest

from everbench.tasks import discover_tasks, load_task

TASK = """
from river import metrics

TASK_NAME = "{name}"
EVENT_STREAM_URL = "https://example.test/events"
LABEL_STREAM_URL = "https://example.test/labels"
PROBLEM_TYPE = "binary_classification"
METRICS = [metrics.ROCAUC()]
DESCRIPTION_HTML = "test"
def accepts_event(payload): return True
def event_id(payload): return payload["id"]
def label_for(payload): return None
"""


def write_task(root: Path, directory_name: str, task_name: str) -> None:
    task_directory = root / directory_name
    task_directory.mkdir()
    (task_directory / "task.py").write_text(TASK.format(name=task_name))


def test_loads_task_directories_in_path_order(tmp_path: Path) -> None:
    for name in ("beta", "alpha"):
        write_task(tmp_path, name, name)
    examples = tmp_path / "alpha" / "examples"
    examples.mkdir()
    (examples / "not_a_task.py").write_text("raise AssertionError('must not load')")

    tasks = discover_tasks(tmp_path)

    assert [task.TASK_NAME for task in tasks] == ["alpha", "beta"]


def test_rejects_duplicate_task_names(tmp_path: Path) -> None:
    for directory_name in ("one", "two"):
        write_task(tmp_path, directory_name, "same")

    with pytest.raises(ValueError, match="unique"):
        discover_tasks(tmp_path)


def test_wiki_task_accepts_a_live_reverted_tag_event() -> None:
    task = load_task("tasks/wiki_liftwing/task.py")
    event = {
        "database": "enwiki",
        "rev_id": 123,
        "tags": ["visualeditor", "mw-reverted"],
        "prior_state": {"tags": ["visualeditor"]},
        "meta": {"dt": "2026-09-03T12:34:56Z"},
    }

    assert task.NEGATIVE_LABEL_DELAY_SECONDS == 48 * 60 * 60
    assert task.label_for(event) == ("enwiki:123", 1, "mw-reverted")
    assert task.label_timestamp is not None
    assert task.label_timestamp(event) == 1_788_438_896.0
