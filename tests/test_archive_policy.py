from datetime import UTC, date, datetime
from pathlib import Path

from everbench.archive import archive_cutoff, archive_week_closed
from everbench.tasks import load_task


def test_archive_waits_for_the_complete_week() -> None:
    week_start = date(2026, 8, 31)

    assert not archive_week_closed(
        week_start=week_start,
        cutoff=datetime(2026, 9, 6, 23, 59, 59, tzinfo=UTC),
    )
    assert archive_week_closed(
        week_start=week_start,
        cutoff=datetime(2026, 9, 7, tzinfo=UTC),
    )


def test_september_seventh_week_waits_until_label_delay_plus_one_day() -> None:
    week_start = date(2026, 9, 7)
    task = load_task(path=Path("tasks/wiki_liftwing/task.py"))

    assert not archive_week_closed(
        week_start=week_start,
        cutoff=archive_cutoff(task=task, now=datetime(2026, 9, 16, 23, 59, 59, tzinfo=UTC), minimum_days=1),
    )
    assert archive_week_closed(
        week_start=week_start,
        cutoff=archive_cutoff(task=task, now=datetime(2026, 9, 17, 0, 1, tzinfo=UTC), minimum_days=1),
    )


def test_archive_minimum_can_extend_task_label_delay() -> None:
    task = load_task(path=Path("tasks/wiki_liftwing/task.py"))

    assert archive_cutoff(task=task, now=datetime(2026, 9, 21, tzinfo=UTC), minimum_days=7) == datetime(
        2026, 9, 14, tzinfo=UTC
    )


def test_short_label_horizon_is_included_in_archive_cutoff() -> None:
    task = load_task(path=Path("tasks/dummy/task.py"))

    assert archive_cutoff(task=task, now=datetime(2026, 9, 15, tzinfo=UTC), minimum_days=1) == datetime(
        2026, 9, 13, 23, 58, 57, tzinfo=UTC
    )
