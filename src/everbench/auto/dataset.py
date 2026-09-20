"""A locally staged weekly Parquet archive for repeated candidate evaluation."""

from __future__ import annotations

import itertools
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self, overload

from everbench.replay import ArchiveExample, read_examples
from everbench.schema import ArchiveManifest


class ArchiveWeek(Sequence[ArchiveExample]):
    """One archive file staged locally so every candidate can replay it."""

    def __init__(
        self, *, path: Path, row_count: int, directory: tempfile.TemporaryDirectory[str] | None = None
    ) -> None:
        self.path = path
        self.row_count = row_count
        self._directory = directory

    @classmethod
    def open(cls, *, manifest: ArchiveManifest) -> Self:
        from everbench.archive import read_archive

        directory = tempfile.TemporaryDirectory(prefix="everbench-archive-week-")
        path = Path(directory.name) / "week.parquet"
        try:
            path.write_bytes(read_archive(location=manifest.path))
        except BaseException:
            directory.cleanup()
            raise
        return cls(path=path, row_count=manifest.row_count, directory=directory)

    def __len__(self) -> int:
        return self.row_count

    @overload
    def __getitem__(self, index: int) -> ArchiveExample: ...  # noqa: PLR0917 -- external positional protocol

    @overload
    def __getitem__(self, index: slice) -> tuple[ArchiveExample, ...]: ...  # noqa: PLR0917 -- external positional protocol

    def __getitem__(self, index: int | slice) -> ArchiveExample | tuple[ArchiveExample, ...]:  # noqa: PLR0917 -- external positional protocol
        if isinstance(index, slice):
            return tuple(itertools.islice(self, *index.indices(len(self))))
        position = index + len(self) if index < 0 else index
        if position < 0 or position >= len(self):
            raise IndexError(index)
        return next(itertools.islice(self, position, position + 1))

    def __iter__(self) -> Iterator[ArchiveExample]:
        yield from read_examples(path=self.path)

    def sequence_bounds(self) -> tuple[int, int]:
        sequences = (row.sequence for row in self)
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

    def __exit__(  # noqa: PLR0917 -- external positional protocol
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()
