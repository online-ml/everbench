"""Source transports yield normalized observations and label updates."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from threading import Event
from typing import Protocol

import httpx

from everbench.records import LabelInput, Observation
from everbench.sse import subscribe

Input = Observation | LabelInput


@dataclass(frozen=True, kw_only=True)
class SourceMessage:
    records: tuple[Input, ...]
    cursor: str | None = None


class Source(Protocol):
    @property
    def name(self) -> str: ...

    def read(self, *, stop: Event, cursor: Callable[[], str | None]) -> Iterable[SourceMessage]: ...


@dataclass(frozen=True, kw_only=True)
class SSESource:
    name: str
    url: str
    decode: Callable[..., Iterable[Input]]

    def read(self, *, stop: Event, cursor: Callable[[], str | None]) -> Iterator[SourceMessage]:
        for message in subscribe(name=self.name, url=self.url, stop=stop, last_event_id=cursor):
            yield SourceMessage(records=tuple(self.decode(event=message.payload)), cursor=message.event_id)


@dataclass(frozen=True, kw_only=True)
class PollingSource:
    name: str
    interval_seconds: float
    poll: Callable[..., Iterable[Input]]

    def read(self, *, stop: Event, cursor: Callable[[], str | None]) -> Iterator[SourceMessage]:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            while not stop.is_set():
                try:
                    yield SourceMessage(records=tuple(self.poll(client=client)))
                except (httpx.HTTPError, KeyError, TypeError, ValueError):
                    logging.exception("%s poll failed; retrying next interval", self.name)
                stop.wait(timeout=self.interval_seconds)
