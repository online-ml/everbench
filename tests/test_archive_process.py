"""The production archiver runs to completion in a separate interpreter."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from everbench import event_store
from everbench.db import make_session_factory
from everbench.records import Observation
from everbench.schema import Base, BenchmarkEvent


def test_archive_once_command_publishes_and_purges_in_child(*, tmp_path: Path) -> None:
    database = tmp_path / "everbench.db"
    sessions = make_session_factory(url=f"sqlite:///{database}")
    Base.metadata.create_all(sessions.kw["bind"])
    old = datetime.now(UTC) - timedelta(days=15)
    with sessions.begin() as session:
        event_store.add_events(
            session=session,
            task_name="dummy",
            events=[Observation(event_id="old", timestamp=old.timestamp(), payload={"value": 1})],
        )
        event = session.get(BenchmarkEvent, {"task_name": "dummy", "event_id": "old"})
        assert event is not None
        event.inserted_at = old
        event_store.add_labels(
            session=session,
            task_name="dummy",
            labels=[event_store.LabelInput(event_id="old", y=1, reason="test")],
            policy=None,
        )

    task_file = Path(__file__).resolve().parents[1] / "tasks/dummy/task.py"
    command = [sys.executable, "-m", "everbench.cli", "archive-once", str(task_file)]
    environment = os.environ | {
        "DATABASE_URL": f"sqlite:///{database}",
        "EVERBENCH_ARCHIVE_ROOT": str(tmp_path / "archives"),
        # Keep a local .env from selecting the production object store.
        "S3_BUCKET_NAME": "",
    }
    assert subprocess.check_output(command, text=True, env=environment).strip() == "1"
    assert subprocess.check_output(command, text=True, env=environment).strip() == "0"

    with sessions() as session:
        assert session.get(BenchmarkEvent, {"task_name": "dummy", "event_id": "old"}) is None
    assert len(list((tmp_path / "archives").rglob("*.parquet"))) == 1
    sessions.kw["bind"].dispose()
