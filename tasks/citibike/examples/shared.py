"""Causal features and capacity normalization shared by the Citi Bike examples."""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from river import base


def features(*, event: dict) -> dict[Hashable, float]:
    info = event["station_information"]
    status = event["station_status"]
    target = datetime.fromtimestamp(event["target_timestamp"], ZoneInfo("America/New_York"))
    hour = target.hour + target.minute / 60
    capacity = max(float(info.get("capacity", 0)), 1.0)
    occupancy = status["num_bikes_available"] / capacity
    result: dict[Hashable, float] = {
        "occupancy": occupancy,
        "ebikes": float(status.get("num_ebikes_available", 0)) / capacity,
        "docks": float(status.get("num_docks_available", 0)) / capacity,
        "disabled_bikes": float(status.get("num_bikes_disabled", 0)) / capacity,
        "disabled_docks": float(status.get("num_docks_disabled", 0)) / capacity,
        "log_capacity": math.log1p(capacity),
        "latitude": float(info["lat"]) - 40.75,
        "longitude": float(info["lon"]) + 73.95,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "weekday_sin": math.sin(2 * math.pi * target.weekday() / 7),
        "weekday_cos": math.cos(2 * math.pi * target.weekday() / 7),
        "weekend": float(target.weekday() >= 5),
    }
    history = [row for row in event.get("history", []) if 0 < event["timestamp"] - row["timestamp"] <= 65 * 60]
    for minutes in (15, 30, 60):
        matches = [row for row in history if abs(event["timestamp"] - row["timestamp"] - minutes * 60) <= 5 * 60]
        previous = min(matches, key=lambda row: abs(event["timestamp"] - row["timestamp"] - minutes * 60), default=None)
        lag = previous["bikes"] / max(float(previous["capacity"]), 1.0) if previous else occupancy
        result[f"occupancy_lag_{minutes}m"] = lag
        result[f"change_{minutes}m"] = occupancy - lag
        result[f"has_lag_{minutes}m"] = float(previous is not None)
    occupancies = [row["bikes"] / max(float(row["capacity"]), 1.0) for row in history]
    result["occupancy_mean_60m"] = sum([occupancy, *occupancies]) / (len(occupancies) + 1)
    result["history_count"] = float(len(history))
    averages = event.get("station_averages", {})
    for statistic in ("mean", "ewm"):
        station = averages.get(f"occupancy_{statistic}_by_station", occupancy)
        hourly = (
            averages[f"occupancy_{statistic}_by_station_hour"]
            if averages.get("occupancy_count_by_station_hour", 0)
            else station
        )
        for group, average in (("station", station), ("station_hour", hourly)):
            result[f"occupancy_{statistic}_by_{group}"] = float(average)
            result[f"occupancy_{statistic}_by_{group}_minus_current"] = average - occupancy
    for group in ("station", "station_hour"):
        result[f"log_occupancy_count_by_{group}"] = math.log1p(averages.get(f"occupancy_count_by_{group}", 0))
    return result


@dataclass(kw_only=True)
class SharedRegressor:
    """Learn a capacity-normalized change, then return a forecast in bikes."""

    model: base.Regressor

    def predict_one(self, *, event_id: str, event: dict) -> float:
        current = float(event["station_status"]["num_bikes_available"])
        capacity = max(float(event["station_information"].get("capacity", 0)), 1.0)
        return max(0.0, current + capacity * self.model.predict_one(features(event=event)))

    def learn_one(self, *, event_id: str, event: dict, label: float) -> None:
        change = float(label) - float(event["station_status"]["num_bikes_available"])
        capacity = max(float(event["station_information"].get("capacity", 0)), 1.0)
        self.model.learn_one(features(event=event), change / capacity)
