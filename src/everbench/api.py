"""Flask control plane and operational-health endpoints."""

from __future__ import annotations

import ast
import io
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache, wraps
from importlib.metadata import distributions
from typing import Any

from flask import Flask, Response, abort, g, jsonify, make_response, render_template, request, send_file
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from everbench import archive, artifacts, store
from everbench.config import CONFIG
from everbench.db import make_session_factory
from everbench.metrics import metric_definition
from everbench.models import validate_uploaded_model
from everbench.tasks import load_task_named


@lru_cache
def sessions() -> sessionmaker[Session]:
    return make_session_factory()


def _session() -> Session:
    if "db_session" not in g:
        g.db_session = sessions()()
    return g.db_session


def _close_session(_: BaseException | None = None) -> None:
    session = g.pop("db_session", None)
    if session is not None:
        session.close()


def require_api_key(view: Callable) -> Callable:
    @wraps(view)
    def wrapped(*args, **kwargs):
        expected = os.getenv("EVERBENCH_API_KEY")
        if not expected:
            return jsonify(error="EVERBENCH_API_KEY is not configured"), 503
        if request.headers.get("X-API-Key") != expected:
            return jsonify(error="invalid API key"), 401
        return view(*args, **kwargs)

    return wrapped


def registration_response(registration) -> dict[str, Any]:
    return {
        "task_name": registration.task_name,
        "model_id": registration.model_id,
        "owner": registration.owner,
        "artifact_id": registration.artifact_id,
        "active": registration.active,
        "start_sequence": registration.start_sequence,
        "created_at": registration.created_at.isoformat(),
    }


def archive_bytes(manifest) -> bytes:
    try:
        return archive.read_archive(manifest.path)
    except (FileNotFoundError, OSError, ValueError):
        abort(404)


def multipart_json(name: str, default: dict[str, Any] | None = None) -> dict[str, Any] | None:
    value = request.form.get(name)
    if value is None:
        return default
    try:
        parsed = json.loads(value)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def displayed_class_definition(source: str, class_name: str) -> str:
    """Keep the dashboard focused on the uploaded model's class.

    A source file is accepted so callers can simply use curl's ``<file``
    syntax. When it is valid Python, retain only the matching top-level class;
    this is presentation metadata and never used to reconstruct the model.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_source_segment(source, node) or source
    return source


def validation_examples(session: Session, task_name: str) -> list[tuple[str, dict[str, float], Any]]:
    """Prefer fresh Postgres labels; archives make a durable fallback."""
    examples = store.latest_labelled_examples(session, task_name, limit=5)
    if len(examples) < 5:
        examples = (
            archive.latest_labelled_examples(store.task_archives(session, task_name), limit=5 - len(examples))
            + examples
        )
    return examples


def task_or_404(task_name: str):
    try:
        return load_task_named(task_name)
    except LookupError:
        abort(404, description=f"task definition not found: {task_name}")


def task_snapshot(session: Session, task_name: str) -> dict[str, Any]:
    heartbeats = [heartbeat for heartbeat in store.worker_health(session) if heartbeat.task_name == task_name]
    runtime = next((heartbeat for heartbeat in heartbeats if heartbeat.role == "task-runtime"), None)
    hot_store: dict[str, int] | None = None
    if runtime and runtime.detail:
        try:
            detail = json.loads(runtime.detail)
            candidate = detail.get("hot_store") if isinstance(detail, dict) else None
            hot_store = candidate if isinstance(candidate, dict) else None
        except ValueError:
            hot_store = None
    return {
        "stats": store.task_stats(session, task_name),
        "leaderboard": store.task_leaderboard(session, task_name),
        "metric_names": store.task_metric_names(session, task_name),
        "hot_store": hot_store,
    }


def create_app() -> Flask:
    app = Flask(__name__)
    app.teardown_appcontext(_close_session)

    @app.template_filter("number")
    def format_number(value: int) -> str:
        return f"{int(value):,}"

    @app.template_filter("file_size")
    def format_file_size(value: int | str | None) -> str:
        try:
            size = float(value if isinstance(value, int) else archive.archive_size(value or ""))
        except OSError:
            return "unavailable"
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{int(size):,} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024
        raise AssertionError("unreachable")

    @app.get("/")
    def dashboard() -> str:
        return render_template("tasks.html", tasks=store.task_names(_session()))

    @app.get("/api")
    def api_documentation() -> str:
        return render_template("api.html")

    @app.get("/tasks/<task_name>")
    def task_dashboard(task_name: str) -> str:
        task = task_or_404(task_name)
        snapshot = task_snapshot(_session(), task_name)
        return render_template(
            "task.html",
            task_name=task_name,
            task_type=task.PROBLEM_TYPE,
            task_description=task.DESCRIPTION_HTML,
            archives=store.task_archives(_session(), task_name),
            **snapshot,
        )

    @app.get("/tasks/<task_name>/panel")
    def task_panel(task_name: str) -> Response:
        """HTML fragment polled by HTMX on the task dashboard."""
        response = make_response(
            render_template("_task_panel.html", task_name=task_name, **task_snapshot(_session(), task_name))
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/tasks/<task_name>/archives/<content_sha256>")
    def download_archive(task_name: str, content_sha256: str) -> Response:
        manifest = store.task_archive(_session(), task_name, content_sha256)
        if manifest is None:
            abort(404)
        return send_file(
            io.BytesIO(archive_bytes(manifest)),
            as_attachment=True,
            download_name=f"{task_name}-{manifest.event_date.isoformat()}-{content_sha256[:12]}.parquet",
        )

    @app.get("/api/health")
    def health() -> Response:
        _session().execute(text("SELECT 1"))
        return jsonify(status="ok")

    @app.get("/api/status")
    @require_api_key
    def status() -> Response:
        now = datetime.now(UTC)
        return jsonify(
            [
                {
                    "worker_id": heartbeat.worker_id,
                    "task_name": heartbeat.task_name,
                    "role": heartbeat.role,
                    "status": heartbeat.status,
                    "last_seen_at": heartbeat.last_seen_at.isoformat(),
                    "stale": (now - heartbeat.last_seen_at).total_seconds() > 45,
                }
                for heartbeat in store.worker_health(_session())
            ]
        )

    @app.get("/api/environment")
    @require_api_key
    def environment() -> Response:
        packages = sorted(
            (
                {"name": distribution.metadata["Name"] or distribution.name, "version": distribution.version}
                for distribution in distributions()
            ),
            key=lambda package: package["name"].lower(),
        )
        return jsonify(python=sys.version, packages=packages)

    @app.post("/api/tasks/<task_name>/models")
    @require_api_key
    def upload_model(task_name: str) -> Response | tuple[Response, int]:
        """Validate and register one signed model artifact."""
        task = task_or_404(task_name)
        uploaded = request.files.get("model")
        model_id = request.form.get("model_id", "")
        owner = request.form.get("owner", "")
        class_definition = request.form.get("class_definition", "")
        metadata = multipart_json("metadata", {})
        if uploaded is None:
            return jsonify(error="multipart form field 'model' is required"), 400
        if not model_id.strip() or not owner.strip():
            return jsonify(error="model_id and owner are required form fields"), 400
        if not class_definition.strip():
            return jsonify(error="class_definition is required so the dashboard can show the model implementation"), 400
        if len(class_definition.encode()) > CONFIG.max_class_definition_bytes:
            return jsonify(error=f"class_definition must be at most {CONFIG.max_class_definition_bytes} bytes"), 413
        if metadata is None:
            return jsonify(error="metadata must be a JSON object"), 400
        payload = uploaded.read(CONFIG.max_model_bytes + 1)
        signature = request.headers.get("X-Everbench-Artifact-Signature", "")
        if not payload or len(payload) > CONFIG.max_model_bytes:
            return jsonify(error=f"model must be between 1 and {CONFIG.max_model_bytes} bytes"), 413
        try:
            is_trusted = artifacts.verify(payload, signature)
        except RuntimeError as error:
            return jsonify(error=str(error)), 503
        if not is_trusted:
            return jsonify(error="invalid model signature"), 400
        session = _session()
        try:
            examples = validation_examples(session, task_name)
            example_count = validate_uploaded_model(task, payload, signature, examples)
            class_name = type(artifacts.loads(payload, signature)).__name__
            metadata = {
                **metadata,
                "class_definition": displayed_class_definition(class_definition, class_name),
                "class_name": class_name,
            }
            artifact_record = store.store_artifact(session, payload, signature, metadata)
            definition = metric_definition(task.PROBLEM_TYPE, task.METRICS)
            store.record_artifact_validation(artifact_record, task_name, definition, example_count)
            existing = store.model_registration(session, task_name, model_id.strip())
            if (existing is None or not existing.active) and store.active_model_count(
                session, task_name
            ) >= CONFIG.max_active_models_per_task:
                raise ValueError(f"a task may have at most {CONFIG.max_active_models_per_task} active models")
            registration, created = store.register_model(
                session, task_name, model_id.strip(), owner.strip(), artifact_record.artifact_id
            )
            session.commit()
        except ValueError as error:
            session.rollback()
            return jsonify(error=str(error)), 409
        except Exception as error:
            session.rollback()
            return jsonify(error=f"model validation failed: {error}"), 422
        response = registration_response(registration)
        response["created"] = created
        response["validation_examples"] = example_count
        response["sha256"] = artifact_record.sha256
        return jsonify(response), 201 if created else 200

    @app.delete("/api/tasks/<task_name>/models/<model_id>")
    @require_api_key
    def remove_model(task_name: str, model_id: str) -> Response | tuple[Response, int]:
        session = _session()
        if not store.deactivate_model(session, task_name, model_id):
            return jsonify(error="no active model with that ID"), 404
        session.commit()
        return jsonify(task_name=task_name, model_id=model_id, active=False)

    @app.post("/api/tasks/<task_name>/backtest")
    @require_api_key
    def backtest_model(task_name: str) -> Response | tuple[Response, int]:
        """Run an uploaded signed model without registering or persisting it."""
        task = task_or_404(task_name)
        uploaded = request.files.get("model")
        archive_sha256 = request.form.get("archive_sha256")
        if uploaded is None:
            return jsonify(error="multipart form field 'model' is required"), 400
        if not isinstance(archive_sha256, str) or not archive_sha256:
            return jsonify(error="archive_sha256 is required"), 400
        payload = uploaded.read(CONFIG.max_model_bytes + 1)
        signature = request.headers.get("X-Everbench-Artifact-Signature", "")
        if not payload or len(payload) > CONFIG.max_model_bytes:
            return jsonify(error=f"model must be between 1 and {CONFIG.max_model_bytes} bytes"), 413
        try:
            uploaded_model = artifacts.loads(payload, signature)
        except RuntimeError as error:
            return jsonify(error=str(error)), 503
        except ValueError:
            return jsonify(error="invalid model signature"), 400
        session = _session()
        manifest = store.task_archive(session, task_name, archive_sha256)
        if manifest is None:
            return jsonify(error="archive not found"), 404
        try:
            archive_bytes_count = (
                manifest.byte_size if manifest.byte_size is not None else archive.archive_size(manifest.path)
            )
        except OSError:
            return jsonify(error="archive is unavailable"), 404
        if manifest.row_count > CONFIG.max_backtest_rows:
            return jsonify(error=f"archive exceeds the {CONFIG.max_backtest_rows:,}-row backtest limit"), 413
        if archive_bytes_count > CONFIG.max_backtest_bytes:
            return jsonify(error=f"archive exceeds the {CONFIG.max_backtest_bytes:,}-byte backtest limit"), 413
        try:
            result = archive.replay_archive(task, uploaded_model, archive_bytes(manifest))
        except Exception as error:
            return jsonify(error=f"backtest failed: {error}"), 422
        return jsonify(archive_sha256=manifest.content_sha256, **result)

    return app


app = create_app()
