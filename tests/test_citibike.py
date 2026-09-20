from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import cloudpickle
import httpx
import pytest

from everbench.metrics import MetricTracker
from everbench.models import PickledModel, prediction_for
from everbench.sources import PollingSource
from everbench.tasks import discover_tasks, load_task

ROOT = Path(__file__).parents[1]


def module_at(*, relative_path: str):
    spec = importlib.util.spec_from_file_location(f"citibike_test_{Path(relative_path).stem}", ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


task = module_at(relative_path="tasks/citibike/task.py")
shared = module_at(relative_path="tasks/citibike/examples/shared.py")
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
    assert definition.label_policy.delay_seconds == 55 * 60
    assert definition.label_policy.tolerance_seconds == 10 * 60
    assert definition.label_policy.default_label is None


def test_snapshot_preserves_raw_fields_zero_counts_and_actual_origin():
    info, status = observation(bikes=0)
    row = next(task.snapshot_events(information=info, status=status, observed_at=NOW + 12.5))
    assert row["station_status"] == status["data"]["stations"][0]
    assert row["station_information"] == info["data"]["stations"][0]
    assert row["timestamp"] == NOW + 12.5
    assert row["target_timestamp"] == NOW + 3612.5
    assert row["id"] == event(timestamp=NOW + 30)["id"]


@pytest.mark.parametrize("change", ["stale_feed", "future_feed", "closed", "nj", "missing", "negative"])
def test_invalid_snapshots_are_not_events(*, change):
    info, status = observation()
    station = status["data"]["stations"][0]
    if change == "stale_feed":
        status["last_updated"] = NOW - 301
    elif change == "future_feed":
        status["last_updated"] = NOW + 1
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
    monkeypatch.setattr(task.time, "monotonic", lambda: NOW)

    class Stop:
        waits = 0

        def is_set(self):
            return self.waits >= 2

        def wait(self, *, timeout):
            assert timeout == 15 * 60
            self.waits += 1

    messages = list(task.TASK.sources[0].read(stop=Stop(), cursor=lambda: None))
    assert [record.payload for message in messages for record in message.records] == [{**event(), "history": []}]
    assert calls[-2:] == ["https://test/info", "https://test/status"]


def test_poll_interval_accounts_for_fetch_and_ingestion_time(*, monkeypatch):
    ticks = iter([0.0, 30.0, 900.0, 970.0])
    monkeypatch.setattr(task.time, "monotonic", lambda: next(ticks))
    waits = []

    stop = Event()
    monkeypatch.setattr(stop, "is_set", lambda: len(waits) == 2)
    monkeypatch.setattr(stop, "wait", lambda *, timeout: waits.append(timeout))
    source = PollingSource(name="test", interval_seconds=900, poll=lambda **kwargs: ())
    assert len(list(source.read(stop=stop, cursor=lambda: None))) == 2
    assert waits == [870, 830]


@pytest.mark.parametrize("model_id", ["persistence", "linear-regression", "hoeffding-tree", "adaptive-forest"])
def test_models_pool_stations_and_survive_serialization(*, model_id):
    definition = load_task(path=ROOT / "tasks/citibike/task.py")
    model = module_at(relative_path=f"tasks/citibike/examples/{model_id.replace('-', '_')}.py").build_model()
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
    expected = shared.features(event=payload)
    payload.update(id="new", station_id="different", y=999)
    assert shared.features(event=payload) == expected
    assert all(isinstance(value, float) for value in expected.values())


def test_station_report_age_does_not_filter_a_current_snapshot():
    info, status = observation()
    status["data"]["stations"][0]["last_reported"] = NOW - 900
    assert len(list(task.snapshot_events(information=info, status=status, observed_at=NOW))) == 1


def test_lags_are_station_specific_bounded_and_frozen_at_observation_time():
    feed = task.CitiBikeFeed()
    retained = None
    for index in range(6):
        timestamp = NOW + index * 900
        info, status = observation(timestamp=timestamp, bikes=10 + index)
        other_info, other_status = observation(timestamp=timestamp, station_id="other", bikes=25)
        info["data"]["stations"].extend(other_info["data"]["stations"])
        status["data"]["stations"].extend(other_status["data"]["stations"])
        feed.information = info
        records = list(feed.observations(status=status, observed_at=timestamp))
        if index == 4:
            retained = records[0].payload
            assert shared.features(event=retained)["change_15m"] == pytest.approx(1 / 30)
            assert shared.features(event=retained)["change_30m"] == pytest.approx(2 / 30)
            assert shared.features(event=retained)["change_60m"] == pytest.approx(4 / 30)
            assert shared.features(event=retained)["occupancy_mean_60m"] == pytest.approx(12 / 30)
    assert len(feed.history) == 4
    assert retained is not None
    assert [row["bikes"] for row in retained["history"]] == [10, 11, 12, 13]
    assert all(row["timestamp"] < retained["timestamp"] for row in retained["history"])


def test_missing_lags_and_restart_use_current_occupancy_without_fake_history():
    info, status = observation(timestamp=NOW + 7200)
    feed = task.CitiBikeFeed()
    feed.information = info
    feed.history.append({"nyc": {"timestamp": NOW, "bikes": 29, "capacity": 30}})
    payload = list(feed.observations(status=status, observed_at=NOW + 7200))[0].payload
    assert payload["history"] == []
    values = shared.features(event=payload)
    assert values["occupancy_lag_15m"] == values["occupancy"]
    assert values["has_lag_15m"] == values["history_count"] == 0


def test_capacity_normalization_transfers_proportional_changes_between_stations():
    model = module_at(relative_path="tasks/citibike/examples/hoeffding_tree.py").build_model()
    small = event(bikes=12)
    large = event(station_id="large", bikes=24)
    large["station_information"]["capacity"] = 60
    model.learn_one(event_id=small["id"], event=small, label=18)
    assert model.predict_one(event_id=small["id"], event=small) == pytest.approx(18)
    assert model.predict_one(event_id=large["id"], event=large) == pytest.approx(36)


def test_exported_model_can_run_without_the_examples_on_python_path(*, tmp_path):
    model = module_at(relative_path="tasks/citibike/examples/linear_regression.py").build_model()
    model_module = sys.modules[type(model).__module__]
    cloudpickle.register_pickle_by_value(model_module)
    try:
        payload = cloudpickle.dumps((model, event()))
    finally:
        cloudpickle.unregister_pickle_by_value(model_module)
    path = tmp_path / "model.pkl"
    path.write_bytes(payload)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import cloudpickle, sys; model, event = cloudpickle.load(open(sys.argv[1], 'rb')); "
            "assert model.predict_one(event_id=event['id'], event=event) == 12; "
            "model.learn_one(event_id=event['id'], event=event, label=17)",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
