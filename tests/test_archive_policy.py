from datetime import UTC, date, datetime

from everbench.archive import archive_batch_ready


def test_open_week_waits_for_a_full_archive_batch() -> None:
    cutoff = datetime(2026, 9, 3, tzinfo=UTC)

    assert not archive_batch_ready(date(2026, 8, 31), cutoff, row_count=99_999, batch_size=100_000)
    assert archive_batch_ready(date(2026, 8, 31), cutoff, row_count=100_000, batch_size=100_000)


def test_closed_week_flushes_a_partial_archive_batch() -> None:
    week_start = date(2026, 8, 31)

    assert not archive_batch_ready(
        week_start,
        datetime(2026, 9, 6, 23, 59, 59, tzinfo=UTC),
        row_count=1,
        batch_size=100_000,
    )
    assert archive_batch_ready(
        week_start,
        datetime(2026, 9, 7, tzinfo=UTC),
        row_count=1,
        batch_size=100_000,
    )
