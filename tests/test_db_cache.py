import os
from pathlib import Path

from everbench import db


def test_reclaim_covers_database_and_write_ahead_log(*, tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "everbench.db"
    database.write_bytes(b"database")
    wal = tmp_path / "everbench.db-wal"
    wal.write_bytes(b"wal")
    engine = db.make_engine(url=f"sqlite:///{database}")
    advised: list[bytes] = []

    def record_advice(fd: int, offset: int, length: int, advice: int) -> None:  # noqa: PLR0917 -- OS callback
        del offset, length, advice
        advised.append(os.read(fd, 16))

    monkeypatch.setattr(db.os, "posix_fadvise", record_advice, raising=False)
    monkeypatch.setattr(db.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    try:
        db.release_sqlite_file_cache(engine=engine)
        assert advised == [b"database", b"wal"]
        wal.unlink()
        db.release_sqlite_file_cache(engine=engine)
        assert advised[-1] == b"database"
    finally:
        engine.dispose()
