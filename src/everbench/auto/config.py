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
    seed_path: Path
    objective: Objective
    context: Any = None
    min_archive_observations: int = 10_000
    max_prediction_time_ratio: float = 3.0
    candidate_budget_per_week: int = 6
    max_candidate_source_bytes: int = 100 * 1024
    max_candidate_model_bytes: int = 16 * 1024 * 1024
    candidate_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if not self.model_id or not self.owner:
            raise ValueError("auto research model_id and owner must be non-empty")
        positive = {
            "min_archive_observations": self.min_archive_observations,
            "max_prediction_time_ratio": self.max_prediction_time_ratio,
            "candidate_budget_per_week": self.candidate_budget_per_week,
            "max_candidate_source_bytes": self.max_candidate_source_bytes,
            "max_candidate_model_bytes": self.max_candidate_model_bytes,
            "candidate_timeout_seconds": self.candidate_timeout_seconds,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"auto research settings must be positive: {', '.join(invalid)}")
        if not self.seed_path.is_file():
            raise ValueError(f"auto research seed file does not exist: {self.seed_path}")
