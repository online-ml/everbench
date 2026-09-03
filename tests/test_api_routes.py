from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from flask.testing import FlaskClient

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


def test_task_panel_is_not_cached(client: FlaskClient) -> None:
    response = client.get("/tasks/dummy/panel")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert 'id="task-panel"' in response.text


def test_unknown_task_panel_is_not_found(client: FlaskClient) -> None:
    response = client.get("/tasks/unknown/panel")

    assert response.status_code == 404


def test_status_requires_an_api_key(client: FlaskClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVERBENCH_API_KEY", raising=False)

    response = client.get("/api/status")

    assert response.status_code == 503
    assert response.get_json() == {"error": "EVERBENCH_API_KEY is not configured"}
