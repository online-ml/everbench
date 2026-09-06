from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from types import SimpleNamespace

import pytest
from flask.testing import FlaskClient
from river import metrics

from everbench import api, archive_store, reporting


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[FlaskClient]:
    monkeypatch.setattr(api, "_session", lambda: SimpleNamespace())
    monkeypatch.setattr(reporting, "worker_health", lambda session: [])
    monkeypatch.setattr(reporting, "task_stats", lambda session, task_name: {"events": 0, "labels": 0})
    monkeypatch.setattr(reporting, "task_leaderboard", lambda session, task_name: [])
    monkeypatch.setattr(reporting, "task_names", lambda session: ["dummy"])
    monkeypatch.setattr(archive_store, "task_archives", lambda session, task_name: [])
    with api.create_app().test_client() as test_client:
        yield test_client


def test_dashboard_lists_known_tasks(client: FlaskClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert 'href="/tasks/dummy"' in response.text


def test_task_dashboard_loads_refresh_behavior_and_configured_metrics(client: FlaskClient) -> None:
    response = client.get("/tasks/dummy")

    assert response.status_code == 200
    assert 'hx-trigger="every 5s"' in response.text
    assert 'src="/static/task.js"' in response.text
    assert "Accuracy" in response.text


def test_task_dashboard_identifies_archive_downloads_by_filename(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = "s3://everbench/task=dummy/week=2026-08-31/events-abc123.parquet"
    monkeypatch.setattr(
        archive_store,
        "task_archives",
        lambda session, task_name: [
            SimpleNamespace(
                path=path,
                event_date=date(2026, 8, 31),
                content_sha256="abc123def456789",
                row_count=42,
                byte_size=2048,
            )
        ],
    )

    response = client.get("/tasks/dummy")

    assert response.status_code == 200
    assert ">dummy-2026-08-31-abc123def456.parquet · 42 records · 2 KB</a>" in response.text
    assert path not in response.text
    assert "week of" not in response.text


def test_task_panel_is_not_cached(client: FlaskClient) -> None:
    response = client.get("/tasks/dummy/panel")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert 'id="task-panel"' in response.text


def test_leaderboard_medals_and_default_metric_order(client: FlaskClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def row(model_id: str, accuracy: float | None, log_loss: float) -> dict:
        return {
            "model_id": model_id,
            "owner": "test",
            "active": True,
            "failure_count": 0,
            "last_error": None,
            "failed_at": None,
            "disabled_until": None,
            "error_count": 0,
            "skipped": 0,
            "created_at": None,
            "predictions": 10,
            "labels": 10,
            "metrics": {"Accuracy": accuracy, "LogLoss": log_loss},
            "model_bytes": 100,
        }

    monkeypatch.setattr(
        reporting,
        "task_leaderboard",
        lambda session, task_name: [
            row("alpha", 0.8, 0.1),
            row("beta", 0.9, 0.3),
            row("gamma", 0.7, 0.2),
            row("unscored", None, 0.05),
        ],
    )

    response = client.get("/tasks/dummy/panel")

    assert response.status_code == 200
    assert response.text.index('data-model-dialog-heading="beta"') < response.text.index(
        'data-model-dialog-heading="alpha"'
    )
    assert response.text.index('data-model-dialog-heading="gamma"') < response.text.index(
        'data-model-dialog-heading="unscored"'
    )
    assert 'data-sort-default-direction="descending">Accuracy</button>' in response.text
    assert '<span class="metric-medal" role="img" aria-label="gold medal">🥇</span> 0.900' in response.text
    assert '<span class="metric-medal" role="img" aria-label="silver medal">🥈</span> 0.800' in response.text
    assert '<span class="metric-medal" role="img" aria-label="bronze medal">🥉</span> 0.700' in response.text
    assert '<span class="metric-medal" role="img" aria-label="gold medal">🥇</span> 0.050' in response.text
    assert (
        'data-tooltip="No finite score yet because this metric needs more suitable labels." '
        'aria-label="No finite score yet because this metric needs more suitable labels.">Ø' in response.text
    )


def test_lower_is_better_first_metric_sorts_ascending() -> None:
    view = api.leaderboard_view(
        [
            {"model_id": "high", "metrics": {"LogLoss": 0.8}},
            {"model_id": "missing", "metrics": {}},
            {"model_id": "low", "metrics": {"LogLoss": 0.2}},
        ],
        (metrics.LogLoss(),),
    )

    assert [row["model_id"] for row in view["leaderboard"]] == ["low", "high", "missing"]
    assert view["leaderboard_metrics"] == [{"name": "LogLoss", "bigger_is_better": False}]


def test_unknown_task_panel_is_not_found(client: FlaskClient) -> None:
    response = client.get("/tasks/unknown/panel")

    assert response.status_code == 404


def test_status_requires_an_api_key(client: FlaskClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVERBENCH_API_KEY", raising=False)

    response = client.get("/api/status")

    assert response.status_code == 503
    assert response.get_json() == {"error": "EVERBENCH_API_KEY is not configured"}
