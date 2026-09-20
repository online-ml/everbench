from __future__ import annotations

import importlib.util
import math
from datetime import UTC, datetime
from pathlib import Path

import cloudpickle
import httpx
import pytest

from everbench.metrics import MetricTracker
from everbench.models import PickledModel, prediction_for
from everbench.tasks import discover_tasks, load_task

ROOT = Path(__file__).parents[1]


def module_at(*, relative_path: str):
    spec = importlib.util.spec_from_file_location("citibike_test_module", ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


task = module_at(relative_path="tasks/citibike/task.py")
regressors = module_at(relative_path="tasks/citibike/examples/regressors.py")
NOW = datetime(2026, 9, 19, 12, tzinfo=UTC).timestamp()


def observation(*, timestamp=NOW, station_id="nyc", bikes=12):
    info = {
        "data": {"stations": [{"station_id": station_id, "region_id": "71", "capacity": 30, "lat": 40.7, "lon": -74.0}]}
    }
    status = {
        "last_updated": timestamp,
        "data": {
            "stations": [
                {
                    "station_id": station_id,
                    "num_bikes_available": bikes,
                    "num_docks_available": 30 - bikes,
                    "num_ebikes_available": 3,
                    "is_installed": 1,
                    "is_renting": 1,
                    "is_returning": 1,
                    "last_reported": timestamp,
                }
            ]
        },
    }
    return info, status


def event(*, timestamp=NOW, station_id="nyc", bikes=12):
    info, status = observation(timestamp=timestamp, station_id=station_id, bikes=bikes)
    return next(task.snapshot_events(information=info, status=status, observed_at=timestamp))


def test_task_is_discovered_as_regression():
    definition = load_task(path=ROOT / "tasks/citibike/task.py")
    assert definition.TASK_NAME in {item.TASK_NAME for item in discover_tasks(directory=ROOT / "tasks")}
    assert definition.PROBLEM_TYPE == "regression"
    assert definition.LEADERBOARD_PRIMARY_METRIC == "MAE"
    assert [type(metric).__name__ for metric in definition.METRICS] == ["MAE", "RMSE"]
    assert definition.label_policy is not None
    assert definition.label_policy.delay_seconds == 1800
    assert definition.label_policy.default_label is None


def test_snapshot_preserves_raw_fields_zero_counts_and_actual_origin():
    info, status = observation(bikes=0)
    row = next(task.snapshot_events(information=info, status=status, observed_at=NOW + 12.5))
    assert row["station_status"] == status["data"]["stations"][0]
    assert row["station_information"] == info["data"]["stations"][0]
    assert row["timestamp"] == NOW + 12.5
    assert row["target_timestamp"] == NOW + 1812.5
    assert row["id"] == event(timestamp=NOW + 30)["id"]


@pytest.mark.parametrize("change", ["stale_feed", "stale_station", "future", "closed", "nj", "missing", "negative"])
def test_invalid_snapshots_are_not_events(*, change):
    info, status = observation()
    station = status["data"]["stations"][0]
    if change == "stale_feed":
        status["last_updated"] = NOW - 121
    elif change == "stale_station":
        station["last_reported"] = NOW - 121
    elif change == "future":
        station["last_reported"] = NOW + 1
    elif change == "closed":
        station["is_renting"] = 0
    elif change == "nj":
        info["data"]["stations"][0]["region_id"] = "70"
    elif change == "missing":
        info["data"]["stations"] = []
    else:
        station["num_bikes_available"] = -1
    assert list(task.snapshot_events(information=info, status=status, observed_at=NOW)) == []


def test_poll_retries_http_failure_and_uses_discovery(*, monkeypatch):
    calls = []
    info, status = observation()

    def response(*, request):
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx.Response(503)
        if request.url.path.endswith("gbfs.json"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "en": {
                            "feeds": [
                                {"name": "station_information", "url": "https://test/info"},
                                {"name": "station_status", "url": "https://test/status"},
                            ]
                        }
                    }
                },
            )
        return httpx.Response(200, json=info if request.url.path == "/info" else status)

    client = httpx.Client(transport=httpx.MockTransport(lambda request: response(request=request)))
    monkeypatch.setattr(task.httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr(task.time, "time", lambda: NOW)

    class Stop:
        waits = 0

        def is_set(self):
            return self.waits >= 2

        def wait(self, *, timeout):
            assert timeout == 15 * 60
            self.waits += 1

    messages = list(task.TASK.sources[0].read(stop=Stop(), cursor=lambda: None))
    assert [record.payload for message in messages for record in message.records] == [event()]
    assert calls[-2:] == ["https://test/info", "https://test/status"]


@pytest.mark.parametrize("model_id", ["persistence", "linear-regression", "hoeffding-tree", "adaptive-forest"])
def test_models_pool_stations_and_survive_serialization(*, model_id):
    definition = load_task(path=ROOT / "tasks/citibike/task.py")
    model = regressors.models()[model_id]
    first = event()
    other = event(station_id="unseen")
    assert model.predict_one(event_id=first["id"], event=first) == 12
    tracker = MetricTracker.fresh(problem_type=definition.PROBLEM_TYPE, prototypes=definition.METRICS)
    for _ in range(50):
        prediction = prediction_for(
            task=definition, model=PickledModel(model_id=model_id, model=model), event_id=first["id"], event=first
        )
        tracker.update(y_true=17, prediction=prediction)
        if hasattr(model, "learn_one"):
            model.learn_one(event_id=first["id"], event=first, label=17)
    assert all(value is not None and math.isfinite(value) for value in tracker.values().values())
    prediction = model.predict_one(event_id=other["id"], event=other)
    if model_id != "persistence":
        assert prediction > 12  # Training one station updates predictions for another.
    assert model.predict_one(event_id=first["id"], event=first) == prediction
    restored = cloudpickle.loads(cloudpickle.dumps(model))
    assert restored.predict_one(event_id=other["id"], event=other) == prediction


def test_features_do_not_include_future_truth_or_raw_identifiers():
    payload = event()
    expected = regressors.features(event=payload)
    payload.update(id="new", station_id="different", y=999)
    assert regressors.features(event=payload) == expected
    assert all(isinstance(value, float) for value in expected.values())
