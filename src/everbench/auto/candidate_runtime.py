"""Private subprocess entry point for candidate construction and evaluation."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import cloudpickle

from everbench.auto.dataset import PreparedTemporalSplit, prepared_temporal_actions
from everbench.auto.evaluation import evaluate_temporal_actions, evaluate_temporally


def _build(source: str):
    name = f"everbench_auto_candidate_{uuid4().hex}"
    module = ModuleType(name)
    module.__file__ = "candidate.py"
    sys.modules[name] = module
    cloudpickle.register_pickle_by_value(module)
    exec(compile(source, "candidate.py", "exec"), module.__dict__)
    builder = getattr(module, "build_model", None)
    if not callable(builder):
        raise TypeError("candidate does not define callable build_model()")
    model = builder()
    if not callable(getattr(model, "learn_one", None)) or not (
        callable(getattr(model, "predict_one", None)) or callable(getattr(model, "predict_proba_one", None))
    ):
        raise TypeError("build_model() must return a River-compatible online classifier")
    return model


def main(request_path: Path, result_path: Path) -> None:
    try:
        with request_path.open("rb") as request_file:
            request = cloudpickle.load(request_file)
        model = _build(request["source"])
        if request["operation"] == "build":
            value = model
        elif request["operation"] == "evaluate":
            split = request["split"]
            if isinstance(split, PreparedTemporalSplit):
                value = evaluate_temporal_actions(
                    request["champion"],
                    model,
                    prepared_temporal_actions(split),
                    request["objective"],
                    promotion_observations=len(split.promotion),
                    max_prediction_time_ratio=request["max_prediction_time_ratio"],
                )
            else:
                value = evaluate_temporally(
                    request["champion"],
                    model,
                    split,
                    request["objective"],
                    max_prediction_time_ratio=request["max_prediction_time_ratio"],
                )
        else:
            raise ValueError(f"unknown candidate operation: {request['operation']!r}")
        result = {"value": value}
    except BaseException as error:
        result = {"error": f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=8)}"}
    with result_path.open("wb") as result_file:
        cloudpickle.dump(result, result_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: candidate_runtime REQUEST_PATH RESULT_PATH")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
