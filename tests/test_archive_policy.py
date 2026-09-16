from datetime import UTC, date, datetime
from pathlib import Path

from everbench.archive import archive_cutoff, archive_week_closed
from everbench.tasks import load_task


def test_archive_waits_for_the_complete_week() -> None:
    week_start = date(2026, 8, 31)

    assert not archive_week_closed(
        week_start,
        datetime(2026, 9, 6, 23, 59, 59, tzinfo=UTC),
    )
    assert archive_week_closed(
        week_start,
        datetime(2026, 9, 7, tzinfo=UTC),
    )


def test_september_seventh_week_waits_until_label_delay_plus_one_day() -> None:
    week_start = date(2026, 9, 7)
    task = load_task(Path("tasks/wiki_liftwing/task.py"))

    assert not archive_week_closed(week_start, archive_cutoff(task, datetime(2026, 9, 16, 23, 59, 59, tzinfo=UTC), 1))
    assert archive_week_closed(week_start, archive_cutoff(task, datetime(2026, 9, 17, tzinfo=UTC), 1))


def test_archive_minimum_can_extend_task_label_delay() -> None:
    task = load_task(Path("tasks/wiki_liftwing/task.py"))

    assert archive_cutoff(task, datetime(2026, 9, 21, tzinfo=UTC), 7) == datetime(2026, 9, 14, tzinfo=UTC)


def test_task_without_delayed_labels_still_waits_one_day() -> None:
    task = load_task(Path("tasks/dummy/task.py"))

    assert archive_cutoff(task, datetime(2026, 9, 15, tzinfo=UTC), 1) == datetime(2026, 9, 14, tzinfo=UTC)
