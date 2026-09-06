"""Owner-controlled policy for autonomous model research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from everbench.auto.research import Objective


@dataclass(frozen=True)
class AutoResearchConfig:
    """Immutable task policy consumed by Everbench's external research worker."""

    model_id: str
    owner: str
    candidate_path: Path
    objective: Objective
    context: Any = None
    history_limit: int = 50_000
    promotion_observations: int = 10_000
    min_research_observations: int = 20_000
    interval_seconds: float = 6 * 60 * 60
    maturity_margin_seconds: float = 10 * 60
    min_research_span_seconds: float = 0.0
    max_prediction_time_ratio: float = 3.0
    research_evaluation_observations: int = 5_000
    max_research_experiments: int = 6
    max_candidate_source_bytes: int = 100 * 1024
    candidate_timeout_seconds: float = 180.0
    retain_raw_examples: bool = False

    def __post_init__(self) -> None:
        if not self.model_id or not self.owner:
            raise ValueError("auto research model_id and owner must be non-empty")
        positive = {
            "history_limit": self.history_limit,
            "promotion_observations": self.promotion_observations,
            "min_research_observations": self.min_research_observations,
            "interval_seconds": self.interval_seconds,
            "max_prediction_time_ratio": self.max_prediction_time_ratio,
            "research_evaluation_observations": self.research_evaluation_observations,
            "max_research_experiments": self.max_research_experiments,
            "max_candidate_source_bytes": self.max_candidate_source_bytes,
            "candidate_timeout_seconds": self.candidate_timeout_seconds,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"auto research settings must be positive: {', '.join(invalid)}")
        if self.history_limit < self.promotion_observations + self.min_research_observations:
            raise ValueError("history_limit must contain the research and promotion cohorts")
        if self.maturity_margin_seconds < 0 or self.min_research_span_seconds < 0:
            raise ValueError("maturity_margin_seconds and min_research_span_seconds cannot be negative")
        if not self.candidate_path.is_file():
            raise ValueError(f"auto research candidate file does not exist: {self.candidate_path}")
