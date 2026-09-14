"""A locally staged weekly Parquet archive for repeated candidate evaluation."""

from __future__ import annotations

import itertools
import json
import tempfile
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self, overload

import pyarrow.parquet as pq

from everbench.auto.evaluation import ArchiveExample
from everbench.schema import ArchiveManifest


class ArchiveWeek(Sequence[ArchiveExample]):
    """One archive file staged locally so every candidate can replay it."""

    def __init__(self, path: Path, row_count: int, directory: tempfile.TemporaryDirectory[str] | None = None) -> None:
        self.path = path
        self.row_count = row_count
        self._directory = directory

    @classmethod
    def open(cls, manifest: ArchiveManifest) -> Self:
        from everbench.archive import read_archive

        directory = tempfile.TemporaryDirectory(prefix="everbench-archive-week-")
        path = Path(directory.name) / "week.parquet"
        try:
            path.write_bytes(read_archive(manifest.path))
        except BaseException:
            directory.cleanup()
            raise
        return cls(path, manifest.row_count, directory)

    def __len__(self) -> int:
        return self.row_count

    @overload
    def __getitem__(self, index: int) -> ArchiveExample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ArchiveExample, ...]: ...

    def __getitem__(self, index: int | slice) -> ArchiveExample | tuple[ArchiveExample, ...]:
        if isinstance(index, slice):
            return tuple(itertools.islice(self, *index.indices(len(self))))
        position = index + len(self) if index < 0 else index
        if position < 0 or position >= len(self):
            raise IndexError(index)
        return next(itertools.islice(self, position, position + 1))

    def __iter__(self) -> Iterator[ArchiveExample]:
        parquet = pq.ParquetFile(self.path)
        columns = [
            "event_id",
            "event_sequence",
            "payload_json",
            "label",
            "event_available_at",
            "label_available_at",
        ]
        for batch in parquet.iter_batches(columns=columns):
            for row in batch.to_pylist():
                available_at = datetime.fromisoformat(row["event_available_at"])
                label_available_at = max(datetime.fromisoformat(row["label_available_at"]), available_at)
                sequence = int(row["event_sequence"])
                event = json.loads(row["payload_json"])
                if not isinstance(event, dict):
                    raise TypeError(f"archived event {row['event_id']!r} is not a mapping")
                yield row["event_id"], sequence, event, row["label"], available_at, label_available_at

    def positive_count(self) -> int:
        return sum(int(bool(row[3])) for row in self)

    def sequence_bounds(self) -> tuple[int, int]:
        sequences = (row[1] for row in self)
        try:
            first = next(sequences)
        except StopIteration as error:
            raise ValueError("cannot find sequence bounds for an empty archive week") from error
        minimum = maximum = first
        for sequence in sequences:
            minimum = min(minimum, sequence)
            maximum = max(maximum, sequence)
        return minimum, maximum

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path, "row_count": self.row_count, "_directory": None}

    def close(self) -> None:
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None

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
