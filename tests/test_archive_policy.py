from datetime import UTC, date, datetime, timedelta

from everbench.archive import archive_week_closed


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


def test_september_seventh_week_waits_until_september_twenty_first_with_seven_day_retention() -> None:
    week_start = date(2026, 9, 7)
    retention = timedelta(days=7)

    assert not archive_week_closed(week_start, datetime(2026, 9, 20, 23, 59, 59, tzinfo=UTC) - retention)
    assert archive_week_closed(week_start, datetime(2026, 9, 21, tzinfo=UTC) - retention)
