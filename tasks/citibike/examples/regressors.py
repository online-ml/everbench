"""Shared online regressors: one fitted estimator for the entire station network."""

from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

from river import forest, linear_model, optim, preprocessing, tree


def features(*, event: dict) -> dict[str, float]:
    info = event["station_information"]
    status = event["station_status"]
    target = datetime.fromtimestamp(event["target_timestamp"], ZoneInfo("America/New_York"))
    hour = target.hour + target.minute / 60
    capacity = max(float(info.get("capacity", 0)), 1.0)
    return {
        "bikes": float(status["num_bikes_available"]),
        "ebikes": float(status.get("num_ebikes_available", 0)),
        "docks": float(status["num_docks_available"]),
        "disabled_bikes": float(status.get("num_bikes_disabled", 0)),
        "disabled_docks": float(status.get("num_docks_disabled", 0)),
        "capacity": capacity,
        "occupancy": status["num_bikes_available"] / capacity,
        "latitude": float(info["lat"]),
        "longitude": float(info["lon"]),
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "weekday_sin": math.sin(2 * math.pi * target.weekday() / 7),
        "weekday_cos": math.cos(2 * math.pi * target.weekday() / 7),
        "weekend": float(target.weekday() >= 5),
    }


class PersistenceRegressor:
    """Predict that the current available-bike count persists for 30 minutes."""

    def predict_one(self, *, event_id: str, event: dict) -> float:
        return float(event["station_status"]["num_bikes_available"])


class SharedRegressor:
    """Learn a shared correction to persistence, pooling every station's labels."""

    def __init__(self, *, model) -> None:
        self.model = model

    def predict_one(self, *, event_id: str, event: dict) -> float:
        current = float(event["station_status"]["num_bikes_available"])
        return max(0.0, current + self.model.predict_one(features(event=event)))

    def learn_one(self, *, event_id: str, event: dict, label: float) -> None:
        change = float(label) - float(event["station_status"]["num_bikes_available"])
        self.model.learn_one(features(event=event), change)


def models() -> dict:
    return {
        "persistence": PersistenceRegressor(),
        "linear-regression": SharedRegressor(
            model=preprocessing.StandardScaler()
            | linear_model.LinearRegression(optimizer=optim.SGD(0.005), l2=0.001, clip_gradient=100)
        ),
        "hoeffding-tree": SharedRegressor(
            model=tree.HoeffdingTreeRegressor(
                leaf_prediction="mean", max_depth=12, max_size=8, memory_estimate_period=10_000
            )
        ),
        "adaptive-forest": SharedRegressor(
            model=forest.ARFRegressor(
                n_models=5, leaf_prediction="mean", max_depth=12, max_size=2, memory_estimate_period=10_000, seed=42
            )
        ),
    }


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    import cloudpickle
    import httpx

    from everbench import artifacts

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", help="Also upload to this Everbench server")
    parser.add_argument("--owner", help="Required when uploading")
    args = parser.parse_args()
    if args.api_url and not args.owner:
        parser.error("--owner is required with --api-url")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_id, model in models().items():
        payload = cloudpickle.dumps(model)
        path = args.output_dir / f"{model_id}.pkl"
        path.write_bytes(payload)
        print(path)
        if args.api_url:
            response = httpx.post(
                f"{args.api_url.rstrip('/')}/api/tasks/citibike/models",
                headers={
                    "X-API-Key": os.environ["EVERBENCH_API_KEY"],
                    "X-Everbench-Artifact-Signature": artifacts.sign(payload=payload),
                },
                data={"model_id": model_id, "owner": args.owner, "class_definition": Path(__file__).read_text()},
                files={"model": (path.name, payload, "application/octet-stream")},
                timeout=60,
            )
            response.raise_for_status()
            print(f"Registered {model_id}")
