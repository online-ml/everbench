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
def features_for(payload): return payload
def label_for(payload): return None
"""


class TaskDiscoveryTest(unittest.TestCase):
    def test_loads_top_level_task_files_in_path_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "beta.py").write_text(TASK.format(name="beta"))
            (root / "alpha.py").write_text(TASK.format(name="alpha"))
            nested = root / "support"
            nested.mkdir()
            (nested / "not_a_task.py").write_text("raise AssertionError('must not load')")

            tasks = discover_tasks(root)

        self.assertEqual([task.TASK_NAME for task in tasks], ["alpha", "beta"])

    def test_rejects_duplicate_task_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.py").write_text(TASK.format(name="same"))
            (root / "two.py").write_text(TASK.format(name="same"))

            with self.assertRaisesRegex(ValueError, "unique"):
                discover_tasks(root)
