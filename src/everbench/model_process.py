"""Killable subprocess boundary for uploaded model code."""

from __future__ import annotations

import multiprocessing
import traceback
from multiprocessing.connection import Connection
from typing import Any

from everbench import artifacts
from everbench.models import PickledModel


def _serve_model(connection: Connection, model_id: str, payload: bytes, signature: str) -> None:
    try:
        model = PickledModel(model_id, artifacts.loads(payload, signature))
        connection.send(
            (
                True,
                {
                    "supports_learning": model.supports_learning,
                    "supports_probabilities": model.supports_probabilities,
                    "supports_scoring": model.supports_scoring,
                    "class_name": type(model.model).__name__,
                },
            )
        )
        while True:
            method, args = connection.recv()
            if method == "close":
                return
            try:
                result = getattr(model, method)(*args)
                connection.send((True, result))
            except BaseException as error:
                connection.send(
                    (
                        False,
                        f"{type(error).__name__}: {error}",
                        "".join(traceback.format_exception(error))[-8_000:],
                    )
                )
    except BaseException as error:
        connection.send(
            (
                False,
                f"{type(error).__name__}: {error}",
                "".join(traceback.format_exception(error))[-8_000:],
            )
        )
    finally:
        connection.close()


class IsolatedModel:
    """A stateful model proxy whose operations have a real wall-clock deadline."""

    def __init__(self, model_id: str, payload: bytes, signature: str, timeout_seconds: float):
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        context = multiprocessing.get_context("spawn")
        self._connection, child_connection = context.Pipe()
        self._process = context.Process(
            target=_serve_model,
            args=(child_connection, model_id, payload, signature),
            name=f"everbench-model-{model_id}",
            daemon=True,
        )
        self._process.start()
        child_connection.close()
        try:
            capabilities = self._receive("load")
        except BaseException:
            self._terminate()
            raise
        self.supports_learning = bool(capabilities["supports_learning"])
        self.supports_probabilities = bool(capabilities["supports_probabilities"])
        self.supports_scoring = bool(capabilities["supports_scoring"])
        self.class_name = str(capabilities["class_name"])

    def _receive(self, operation: str) -> Any:
        if not self._connection.poll(self.timeout_seconds):
            self._terminate()
            raise TimeoutError(f"{self.model_id} {operation} exceeded the {self.timeout_seconds:.1f}s operation limit")
        try:
            message = self._connection.recv()
        except EOFError as error:
            self._terminate()
            raise RuntimeError(f"{self.model_id} process exited during {operation}") from error
        if message[0]:
            return message[1]
        _, summary, remote_traceback = message
        raise RuntimeError(f"{self.model_id} {operation} failed: {summary}\n{remote_traceback}")

    def _call(self, method: str, *args: Any) -> Any:
        if not self._process.is_alive():
            raise RuntimeError(f"{self.model_id} process is not running")
        try:
            self._connection.send((method, args))
        except (BrokenPipeError, EOFError, OSError) as error:
            self._terminate()
            raise RuntimeError(f"{self.model_id} process exited before {method}") from error
        return self._receive(method.removesuffix("_one"))

    def predict_one(self, event_id: str, event: dict[str, Any]) -> Any:
        return self._call("predict_one", event_id, event)

    def predict_proba_one(self, event_id: str, event: dict[str, Any]) -> Any:
        return self._call("predict_proba_one", event_id, event)

    def score_one(self, event_id: str, event: dict[str, Any]) -> Any:
        return self._call("score_one", event_id, event)

    def learn_one(self, event_id: str, event: dict[str, Any], label: Any) -> None:
        self._call("learn_one", event_id, event, label)

    def payload(self) -> bytes:
        return self._call("payload")

    def _terminate(self) -> None:
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=1)
        if not self._connection.closed:
            self._connection.close()

    def close(self) -> None:
        if not self._process.is_alive():
            self._terminate()
            return
        try:
            self._connection.send(("close", ()))
            self._process.join(timeout=1)
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            self._terminate()

    def __enter__(self) -> IsolatedModel:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
