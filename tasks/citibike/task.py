"""Forecast NYC Citi Bike station availability with delayed, observed labels."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator

import httpx
from river import metrics

from everbench.records import Observation
from everbench.sources import PollingSource
from everbench.tasks import LabelPolicy, TaskDefinition

HORIZON_SECONDS = 60 * 60
POLL_SECONDS = 15 * 60
TARGET_TOLERANCE_SECONDS = 5 * 60
MAX_FEED_AGE_SECONDS = 5 * 60
DESCRIPTION_HTML = """
<p>Predict the number of available bikes at each NYC Citi Bike station one hour ahead.
Each regressor is one shared model trained across all stations.
The <a href="https://citibikenyc.com/system-data">official GBFS feed</a> is polled every 15 minutes.
The ground truth is the first station snapshot collected between +55 and +65 minutes.</p>
"""


def _get(*, client: httpx.Client, url: str) -> dict:
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def snapshot_events(*, information: dict, status: dict, observed_at: float) -> Iterator[dict]:
    """Keep raw station fields and collection time; do not backdate observations."""
    updated = float(status["last_updated"])
    if not 0 <= observed_at - updated <= MAX_FEED_AGE_SECONDS:
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
    """Cache metadata and four compact snapshots for causal lag features."""

    def __init__(self) -> None:
        self.feeds: dict[str, str] = {}
        self.information: dict = {}
        self.refreshed_at = 0.0
        self.history: deque[dict[str, dict]] = deque(maxlen=4)

    def observations(self, *, status: dict, observed_at: float) -> Iterator[Observation]:
        snapshot = {}
        for event in snapshot_events(information=self.information, status=status, observed_at=observed_at):
            station_id = event["station_id"]
            event["history"] = [
                previous[station_id]
                for previous in self.history
                if station_id in previous and 0 < observed_at - previous[station_id]["timestamp"] <= 65 * 60
            ]
            snapshot[station_id] = {
                "timestamp": observed_at,
                "bikes": event["station_status"]["num_bikes_available"],
                "capacity": event["station_information"].get("capacity", 0),
            }
            yield Observation(
                event_id=event["id"],
                timestamp=event["timestamp"],
                payload=event,
                entity_key=station_id,
                value=event["station_status"]["num_bikes_available"],
            )
        if snapshot:
            self.history.append(snapshot)

    def __call__(self, *, client: httpx.Client) -> Iterator[Observation]:
        if not self.feeds or time.time() - self.refreshed_at >= 3600:
            discovery = _get(client=client, url=GBFS_URL)
            self.feeds = {feed["name"]: feed["url"] for feed in discovery["data"]["en"]["feeds"]}
            self.information = _get(client=client, url=self.feeds["station_information"])
            self.refreshed_at = time.time()
        status = _get(client=client, url=self.feeds["station_status"])
        yield from self.observations(status=status, observed_at=time.time())


GBFS_URL = "https://gbfs.citibikenyc.com/gbfs/2.3/gbfs.json"
TASK = TaskDefinition(
    TASK_NAME="citibike",
    PROBLEM_TYPE="regression",
    METRICS=(metrics.MAE(), metrics.RMSE()),
    LEADERBOARD_PRIMARY_METRIC="MAE",
    DESCRIPTION_HTML=DESCRIPTION_HTML,
    sources=(PollingSource(name="stations", interval_seconds=POLL_SECONDS, poll=CitiBikeFeed()),),
    label_policy=LabelPolicy(
        delay_seconds=HORIZON_SECONDS - TARGET_TOLERANCE_SECONDS,
        tolerance_seconds=2 * TARGET_TOLERANCE_SECONDS,
    ),
)
