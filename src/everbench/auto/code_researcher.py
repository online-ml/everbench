"""OpenAI tool loop for Karpathy-style single-file model research."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from everbench.auto.evaluation import TemporalObservation

INSTRUCTIONS = """You are an autonomous online-machine-learning researcher.
You may replace the complete candidate.py program. Its build_model() function
must return a River Classifier that receives the full raw event mapping in
predict_one/predict_proba_one and learn_one. Feature extraction, models,
hyperparameters, drift handling, ensembles, voting, and stacking are all in
scope. Do not change the interface or evaluation harness and do not attempt
system, filesystem, network, environment, or secret access. Dependencies are
fixed to the imports admitted by the candidate validator.

Submit each complete program and hypothesis to evaluate_candidate. Iterate from
the measured research results, then call finish. The best constraint-passing
program you evaluated is frozen automatically. Promotion is decided later on a
separate sealed cohort that you never see. Prefer causal, bounded-memory online
designs.
"""


@dataclass(frozen=True)
class ResearchRequest:
    """Everything the coding agent may inspect during one reflection."""

    context: Any
    objective: dict[str, Any]
    research_summary: dict[str, Any]
    current_source: str
    previous_experiments: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class CodeProposal:
    hypothesis: str
    source: str


def describe_objective(objective: Any) -> dict[str, Any]:
    return {
        "primary_metric": type(objective.metric).__name__,
        "bigger_is_better": objective.metric.bigger_is_better,
        "minimum_improvement": objective.min_improvement,
        "minimum_promotion_observations": objective.min_observations,
        "required_constraints": list(objective.required_constraints),
        "secondary_metric_constraints": [
            {
                "name": constraint.name,
                "metric": type(constraint.metric).__name__,
                "max_regression": constraint.max_regression,
            }
            for constraint in objective.metric_constraints
        ],
    }


def summarize_research(
    observations: tuple[TemporalObservation, ...],
    *,
    include_raw_examples: bool = False,
    max_examples: int = 12,
) -> dict[str, Any]:
    """Describe agent-visible history, optionally including bounded raw rows."""
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    first = observations[0]
    last = observations[-1]
    positives = sum(int(bool(row.y)) for row in observations)
    feature_types: dict[str, set[str]] = {}

    def record(value: Any, path: str, depth: int = 0) -> None:
        feature_types.setdefault(path or "$", set()).add(type(value).__name__)
        if depth >= 4:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                record(child, f"{path}.{key}" if path else str(key), depth + 1)
        elif isinstance(value, list):
            for child in value[:5]:
                record(child, f"{path}[]", depth + 1)

    for row in observations[: min(len(observations), 1_000)]:
        record(row.x, "")
    summary: dict[str, Any] = {
        "observations": len(observations),
        "positive_rate": positives / len(observations),
        "available_from": first.available_at.isoformat(),
        "available_to": last.available_at.isoformat(),
        "payload_schema": {key: sorted(values) for key, values in sorted(feature_types.items())},
    }
    if include_raw_examples:
        step = max(len(observations) // max_examples, 1)
        summary["raw_examples"] = [
            {
                "sequence": row.sequence,
                "event": row.x,
                "label": row.y,
                "event_available_at": row.available_at.isoformat(),
                "label_available_at": row.label_available_at.isoformat(),
            }
            for row in observations[::step][:max_examples]
        ]
    return summary


class OpenAICodeResearcher:
    """Let a model edit and repeatedly measure one candidate program."""

    def __init__(
        self,
        model: str = "gpt-5.6-terra",
        reasoning_effort: str = "high",
        max_experiments: int = 6,
        response_timeout_seconds: float = 300.0,
        client: Any = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_experiments = max_experiments
        self.response_timeout_seconds = response_timeout_seconds
        self.client: Any = client

    def research(
        self,
        request: ResearchRequest,
        evaluate: Callable[[str], dict[str, Any]],
    ) -> CodeProposal:
        experiments = 0
        attempts = 0
        best: CodeProposal | None = None
        best_rank = (False, float("-inf"))
        tools = [
            {
                "type": "function",
                "name": "evaluate_candidate",
                "description": "Evaluate a complete candidate.py program on research-only causal data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "minLength": 1},
                        "hypothesis": {"type": "string", "minLength": 1},
                    },
                    "required": ["source", "hypothesis"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "finish",
                "description": "Finish research and freeze the best constraint-passing candidate evaluated so far.",
                "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                "strict": True,
            },
        ]
        prompt = {
            "problem_context": request.context,
            "objective": request.objective,
            "research_summary": request.research_summary,
            "previous_experiments": request.previous_experiments,
            "current_candidate_source": request.current_source,
        }
        conversation: list[Any] = [{"role": "user", "content": json.dumps(prompt, default=str, sort_keys=True)}]
        max_attempts = self.max_experiments * 2 + 2
        max_rounds = max_attempts + self.max_experiments + 6
        for _ in range(max_rounds):
            response = self.client.responses.create(
                model=self.model,
                instructions=INSTRUCTIONS,
                input=conversation,
                reasoning={"effort": self.reasoning_effort},
                max_output_tokens=16_000,
                tools=tools,
                tool_choice="required",
                parallel_tool_calls=False,
                store=False,
                timeout=self.response_timeout_seconds,
            )
            conversation.extend(response.output)
            calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            if not calls:
                raise RuntimeError("coding researcher returned no tool call")
            outputs = []
            for call in calls:
                try:
                    arguments = json.loads(call.arguments)
                    if call.name == "evaluate_candidate":
                        if attempts >= max_attempts:
                            result: dict[str, Any] = {
                                "ok": False,
                                "error": "research attempt budget exhausted; call finish",
                            }
                            outputs.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": call.call_id,
                                    "output": json.dumps(result, sort_keys=True),
                                }
                            )
                            continue
                        attempts += 1
                        source = arguments.get("source")
                        hypothesis = arguments.get("hypothesis")
                        if not isinstance(source, str) or not source.strip():
                            raise ValueError("source must be a non-empty string")
                        if not isinstance(hypothesis, str) or not hypothesis.strip():
                            raise ValueError("hypothesis must be a non-empty string")
                        result = {"ok": True, **evaluate(source)}
                        experiments += 1
                        constraints_value = result.get("constraints")
                        constraints = constraints_value if isinstance(constraints_value, list) else []
                        feasible = all(
                            isinstance(constraint, dict) and constraint.get("passed") is True
                            for constraint in constraints
                        )
                        rank = (feasible, float(result.get("improvement", 0.0)))
                        if rank > best_rank:
                            best_rank = rank
                            best = CodeProposal(hypothesis=hypothesis.strip(), source=source)
                    elif call.name == "finish":
                        if best is None:
                            result = {"ok": False, "error": "evaluate at least one valid candidate first"}
                        else:
                            return best
                    else:
                        result = {"ok": False, "error": f"unknown tool: {call.name}"}
                except Exception as error:
                    result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, default=str, sort_keys=True),
                    }
                )
            conversation.extend(outputs)
            if experiments >= self.max_experiments and best is not None:
                return best
        if best is not None:
            return best
        raise RuntimeError("coding researcher exhausted its tool budget without a valid candidate")
