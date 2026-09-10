"""Validation and subprocess execution for editable candidate programs."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cloudpickle
from river import base

from everbench.auto.dataset import PreparedTemporalSplit
from everbench.auto.evaluation import EvaluationOutcome, TemporalSplit
from everbench.auto.research import Objective

ALLOWED_IMPORTS = {
    "__future__",
    "collections",
    "dataclasses",
    "datetime",
    "functools",
    "hashlib",
    "heapq",
    "ipaddress",
    "itertools",
    "json",
    "math",
    "operator",
    "random",
    "re",
    "river",
    "statistics",
    "typing",
}
BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}


def validate_candidate_source(source: str, max_bytes: int) -> None:
    """Reject system access while leaving model construction open-ended."""
    if len(source.encode()) > max_bytes:
        raise ValueError(f"candidate source exceeds the {max_bytes:,}-byte limit")
    try:
        tree = ast.parse(source, filename="candidate.py")
    except SyntaxError as error:
        raise ValueError(f"candidate source does not parse: {error}") from error
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                if not any(name == allowed or name.startswith(f"{allowed}.") for allowed in ALLOWED_IMPORTS):
                    raise ValueError(f"candidate import is not allowed: {name!r}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            raise ValueError(f"candidate call is not allowed: {node.func.id}()")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"candidate dunder access is not allowed: {node.attr}")
    builders = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build_model"
    ]
    if len(builders) != 1 or isinstance(builders[0], ast.AsyncFunctionDef):
        raise ValueError("candidate source must define exactly one synchronous build_model()")


def _run_runtime(payload: dict[str, Any], timeout_seconds: float, max_output_bytes: int) -> Any:
    with tempfile.TemporaryDirectory(prefix="everbench-auto-") as directory:
        root = Path(directory)
        request_path = root / "request.pkl"
        result_path = root / "result.pkl"
        with request_path.open("wb") as request_file:
            cloudpickle.dump(payload, request_file)

        def limits() -> None:
            try:
                import resource

                cpu_seconds = max(int(timeout_seconds), 1)
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
                resource.setrlimit(resource.RLIMIT_FSIZE, (max_output_bytes, max_output_bytes))
                resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            except (ImportError, OSError, ValueError):
                pass

        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
            "TMPDIR": directory,
        }
        completed = subprocess.run(
            [sys.executable, "-m", "everbench.auto.candidate_runtime", str(request_path), str(result_path)],
            cwd=directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            check=False,
            preexec_fn=limits if os.name == "posix" else None,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"candidate subprocess exited with status {completed.returncode}")
        if not result_path.is_file():
            raise RuntimeError("candidate subprocess produced no result")
        if result_path.stat().st_size > max_output_bytes:
            raise ValueError("candidate subprocess result exceeds the artifact limit")
        with result_path.open("rb") as result_file:
            result = cloudpickle.load(result_file)
        if not isinstance(result, dict):
            raise RuntimeError("candidate subprocess returned an invalid result")
        if error := result.get("error"):
            raise RuntimeError(str(error))
        return result.get("value")


def build_candidate_model(
    source: str,
    *,
    timeout_seconds: float,
    max_source_bytes: int,
    max_output_bytes: int,
) -> Any:
    validate_candidate_source(source, max_source_bytes)
    model = _run_runtime({"operation": "build", "source": source}, timeout_seconds, max_output_bytes)
    if not callable(getattr(model, "learn_one", None)) or not (
        callable(getattr(model, "predict_one", None)) or callable(getattr(model, "predict_proba_one", None))
    ):
        raise TypeError("build_model() must return a River-compatible online classifier")
    return model


def evaluate_candidate_source(
    source: str,
    champion: base.Classifier,
    split: TemporalSplit | PreparedTemporalSplit,
    objective: Objective,
    *,
    max_prediction_time_ratio: float,
    timeout_seconds: float,
    max_source_bytes: int,
    max_output_bytes: int,
) -> EvaluationOutcome:
    validate_candidate_source(source, max_source_bytes)
    outcome = _run_runtime(
        {
            "operation": "evaluate",
            "source": source,
            "champion": champion,
            "split": split,
            "objective": objective,
            "max_prediction_time_ratio": max_prediction_time_ratio,
        },
        timeout_seconds,
        max_output_bytes,
    )
    if not isinstance(outcome, EvaluationOutcome):
        raise TypeError("candidate evaluator returned an invalid outcome")
    return outcome
