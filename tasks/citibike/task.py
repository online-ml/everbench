"""Forecast NYC Citi Bike station availability with delayed, observed labels."""

from __future__ import annotations

import time
from collections.abc import Iterator

import httpx
from river import metrics

from everbench.records import Observation
from everbench.sources import PollingSource
from everbench.tasks import LabelPolicy, TaskDefinition

HORIZON_SECONDS = 30 * 60
POLL_SECONDS = 15 * 60
MAX_LATENESS_SECONDS = 120
MAX_STALENESS_SECONDS = 120
DESCRIPTION_HTML = """
<p>Predict the number of available bikes at each NYC Citi Bike station 30 minutes ahead.
Each regressor is one shared model trained across all stations.</p>
<p>We poll the <a href="https://citibikenyc.com/system-data">official GBFS feed</a> every 15 minutes.
The target is the first fresh station reading at or after +30 minutes, within a two-minute tolerance.
Unavailable targets are excluded, never filled with zero. MAE and RMSE are measured in bikes.</p>
"""


def _get(*, client: httpx.Client, url: str) -> dict:
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def snapshot_events(*, information: dict, status: dict, observed_at: float) -> Iterator[dict]:
    """Keep raw station fields and collection time; do not backdate observations."""
    updated = float(status["last_updated"])
    if not 0 <= observed_at - updated <= MAX_STALENESS_SECONDS:
        return
    stations = {str(row["station_id"]): row for row in information["data"]["stations"]}
    for row in status["data"]["stations"]:
        station_id = str(row["station_id"])
        info = stations.get(station_id)
        # Region IDs come from this feed's system_regions.json: NYC and Bronx.
        if info is None or str(info.get("region_id")) not in {"71", "185"}:
            continue
        if not all(row.get(field) == 1 for field in ("is_installed", "is_renting", "is_returning")):
            continue
        if not 0 <= observed_at - float(row.get("last_reported", 0)) <= MAX_STALENESS_SECONDS:
            continue
        count = row.get("num_bikes_available")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            continue
        yield {
            "id": f"{station_id}:{int(observed_at // 60)}",
            "timestamp": observed_at,
            "station_id": station_id,
            "station_information": info,
            "station_status": row,
            "feed_last_updated": updated,
            "target_timestamp": observed_at + HORIZON_SECONDS,
        }


class CitiBikeFeed:
    """Keep only the station metadata cache between polls."""

    def __init__(self) -> None:
        self.feeds: dict[str, str] = {}
        self.information: dict = {}
        self.refreshed_at = 0.0

    def __call__(self, *, client: httpx.Client) -> Iterator[Observation]:
        if not self.feeds or time.time() - self.refreshed_at >= 3600:
            discovery = _get(client=client, url=GBFS_URL)
            self.feeds = {feed["name"]: feed["url"] for feed in discovery["data"]["en"]["feeds"]}
            self.information = _get(client=client, url=self.feeds["station_information"])
            self.refreshed_at = time.time()
        status = _get(client=client, url=self.feeds["station_status"])
        for event in snapshot_events(information=self.information, status=status, observed_at=time.time()):
            yield Observation(
                event_id=event["id"],
                timestamp=event["timestamp"],
                payload=event,
                entity_key=event["station_id"],
                value=event["station_status"]["num_bikes_available"],
            )


GBFS_URL = "https://gbfs.citibikenyc.com/gbfs/2.3/gbfs.json"
TASK = TaskDefinition(
    TASK_NAME="citibike",
    PROBLEM_TYPE="regression",
    METRICS=(metrics.MAE(), metrics.RMSE()),
    LEADERBOARD_PRIMARY_METRIC="MAE",
    DESCRIPTION_HTML=DESCRIPTION_HTML,
    sources=(PollingSource(name="stations", interval_seconds=POLL_SECONDS, poll=CitiBikeFeed()),),
    label_policy=LabelPolicy(delay_seconds=HORIZON_SECONDS, tolerance_seconds=MAX_LATENESS_SECONDS),
)
