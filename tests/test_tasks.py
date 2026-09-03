"""Tests for task discovery used by the all-task worker."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from everbench.tasks import discover_tasks

TASK = """
from river import metrics

TASK_NAME = "{name}"
EVENT_STREAM_URL = None
LABEL_STREAM_URL = None
PROBLEM_TYPE = "binary_classification"
METRICS = [metrics.ROCAUC()]
DESCRIPTION_HTML = "test"
def accepts_event(payload): return True
def event_id(payload): return payload["id"]
def label_for(payload): return None
"""


class TaskDiscoveryTest(unittest.TestCase):
    def test_loads_task_directories_in_path_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("beta", "alpha"):
                task_directory = root / name
                task_directory.mkdir()
                (task_directory / "task.py").write_text(TASK.format(name=name))
            examples = root / "alpha" / "examples"
            examples.mkdir()
            (examples / "not_a_task.py").write_text("raise AssertionError('must not load')")

            tasks = discover_tasks(root)

        self.assertEqual([task.TASK_NAME for task in tasks], ["alpha", "beta"])

    def test_rejects_duplicate_task_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for directory_name in ("one", "two"):
                task_directory = root / directory_name
                task_directory.mkdir()
                (task_directory / "task.py").write_text(TASK.format(name="same"))

            with self.assertRaisesRegex(ValueError, "unique"):
                discover_tasks(root)
