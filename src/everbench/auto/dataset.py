"""Disk-backed temporal cohorts for memory-bounded autonomous evaluation."""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self, overload

import cloudpickle

from everbench.auto.evaluation import TemporalObservation


def _time_key(value: datetime) -> int:
    if value.tzinfo is not None:
        value = value.astimezone(UTC).replace(tzinfo=None)
    epoch = datetime(1970, 1, 1)
    delta = value - epoch
    return ((delta.days * 86_400 + delta.seconds) * 1_000_000) + delta.microseconds


def _load_observation(payload: bytes) -> TemporalObservation:
    row = cloudpickle.loads(payload)
    if not isinstance(row, TemporalObservation):
        raise TypeError("prepared temporal dataset contains an invalid observation")
    return row


class PreparedTemporalView(Sequence[TemporalObservation]):
    """A slice of a prepared cohort that loads only requested rows."""

    def __init__(self, path: Path, start: int, stop: int) -> None:
        self.path = path
        self.start = start
        self.stop = stop

    def __len__(self) -> int:
        return self.stop - self.start

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA query_only = ON")
        return connection

    def _at_positions(self, positions: Sequence[int]) -> tuple[TemporalObservation, ...]:
        if not positions:
            return ()
        loaded: dict[int, TemporalObservation] = {}
        with self._connect() as connection:
            for offset in range(0, len(positions), 500):
                chunk = positions[offset : offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT position, payload FROM observations WHERE position IN ({placeholders})",  # noqa: S608
                    chunk,
                )
                loaded.update((position, _load_observation(payload)) for position, payload in rows)
        return tuple(loaded[position] for position in positions)

    @overload
    def __getitem__(self, index: int) -> TemporalObservation: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TemporalObservation, ...]: ...

    def __getitem__(self, index: int | slice) -> TemporalObservation | tuple[TemporalObservation, ...]:
        if isinstance(index, slice):
            positions = [self.start + relative for relative in range(*index.indices(len(self)))]
            return self._at_positions(positions)
        relative = index + len(self) if index < 0 else index
        if relative < 0 or relative >= len(self):
            raise IndexError(index)
        return self._at_positions((self.start + relative,))[0]

    def __iter__(self) -> Iterator[TemporalObservation]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM observations WHERE position >= ? AND position < ? ORDER BY position",
                (self.start, self.stop),
            )
            for (payload,) in rows:
                yield _load_observation(payload)

    def positive_count(self) -> int:
        with self._connect() as connection:
            value = connection.execute(
                "SELECT coalesce(sum(positive), 0) FROM observations WHERE position >= ? AND position < ?",
                (self.start, self.stop),
            ).fetchone()
        assert value is not None
        return int(value[0])

    def sequence_bounds(self) -> tuple[int, int]:
        with self._connect() as connection:
            value = connection.execute(
                "SELECT min(sequence), max(sequence) FROM observations WHERE position >= ? AND position < ?",
                (self.start, self.stop),
            ).fetchone()
        if value is None or value[0] is None or value[1] is None:
            raise ValueError("cannot find sequence bounds for an empty prepared view")
        return int(value[0]), int(value[1])


@dataclass(frozen=True)
class PreparedTemporalSplit:
    """File-backed research and promotion boundaries passed to evaluators."""

    path: Path
    stop: int
    promotion_start: int

    @property
    def research(self) -> PreparedTemporalView:
        return PreparedTemporalView(self.path, 0, self.promotion_start)

    @property
    def promotion(self) -> PreparedTemporalView:
        return PreparedTemporalView(self.path, self.promotion_start, self.stop)


class PreparedTemporalData:
    """Own a compact SQLite cohort whose rows can be replayed repeatedly."""

    def __init__(self, directory: tempfile.TemporaryDirectory[str], path: Path, count: int) -> None:
        self._directory = directory
        self.path = path
        self.count = count

    @classmethod
    def from_observations(cls, observations: Iterable[TemporalObservation]) -> Self:
        directory = tempfile.TemporaryDirectory(prefix="everbench-auto-data-")
        path = Path(directory.name) / "observations.sqlite3"
        try:
            with sqlite3.connect(path) as connection:
                connection.execute("PRAGMA journal_mode = OFF")
                connection.execute("PRAGMA synchronous = OFF")
                connection.execute(
                    """CREATE TABLE observations (
                        position INTEGER PRIMARY KEY,
                        observation_id TEXT NOT NULL UNIQUE,
                        sequence INTEGER NOT NULL,
                        available_at INTEGER NOT NULL,
                        label_available_at INTEGER NOT NULL,
                        positive INTEGER NOT NULL,
                        payload BLOB NOT NULL
                    )"""
                )
                batch: list[tuple[Any, ...]] = []
                count = 0
                for position, row in enumerate(observations):
                    batch.append(
                        (
                            position,
                            row.observation_id,
                            row.sequence,
                            _time_key(row.available_at),
                            _time_key(row.label_available_at),
                            int(bool(row.y)),
                            cloudpickle.dumps(row),
                        )
                    )
                    count = position + 1
                    if len(batch) >= 500:
                        connection.executemany("INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                        batch.clear()
                if batch:
                    connection.executemany("INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                connection.execute(
                    "CREATE INDEX observations_label_order ON observations (label_available_at, sequence)"
                )
        except sqlite3.IntegrityError as error:
            directory.cleanup()
            raise ValueError("observation IDs must be unique") from error
        except BaseException:
            directory.cleanup()
            raise
        return cls(directory, path, count)

    def split(
        self,
        promotion_observations: int,
        min_research_observations: int = 1,
        *,
        stop: int | None = None,
    ) -> PreparedTemporalSplit:
        effective_stop = self.count if stop is None else stop
        if promotion_observations <= 0:
            raise ValueError("promotion_observations must be positive")
        if min_research_observations <= 0:
            raise ValueError("min_research_observations must be positive")
        if effective_stop < 0 or effective_stop > self.count:
            raise ValueError("prepared split boundary is outside the dataset")
        required = promotion_observations + min_research_observations
        if effective_stop < required:
            raise ValueError(f"evaluation needs at least {required} complete observations; received {effective_stop}")
        return PreparedTemporalSplit(self.path, effective_stop, effective_stop - promotion_observations)

    def close(self) -> None:
        self._directory.cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


def prepared_temporal_actions(
    split: PreparedTemporalSplit,
) -> Iterator[tuple[int, TemporalObservation, bool]]:
    """Stream a prepared split in causal event/label order."""
    connection = sqlite3.connect(split.path)
    connection.execute("PRAGMA query_only = ON")
    try:
        events = iter(
            connection.execute(
                """SELECT position, available_at, sequence, payload
                   FROM observations WHERE position < ? ORDER BY position""",
                (split.stop,),
            )
        )
        labels = iter(
            connection.execute(
                """SELECT position, label_available_at, sequence, payload
                   FROM observations WHERE position < ? ORDER BY label_available_at, sequence""",
                (split.stop,),
            )
        )
        event = next(events, None)
        label = next(labels, None)
        while event is not None or label is not None:
            event_key = (event[1], 0, event[2]) if event is not None else None
            label_key = (label[1], 1, label[2]) if label is not None else None
            if label_key is None or (event_key is not None and event_key < label_key):
                assert event is not None
                yield 0, _load_observation(event[3]), event[0] >= split.promotion_start
                event = next(events, None)
            else:
                assert label is not None
                yield 1, _load_observation(label[3]), label[0] >= split.promotion_start
                label = next(labels, None)
    finally:
        connection.close()
