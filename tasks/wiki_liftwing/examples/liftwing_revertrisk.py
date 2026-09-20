"""A scoring-only Lift Wing predictor for the wiki-liftwing task."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LiftWingRevertRisk:
    """Return Lift Wing's probability that a Wikipedia revision is reverted."""

    endpoint = "https://api.wikimedia.org/service/lw/inference/v1/models/revertrisk-language-agnostic:predict"

    def __init__(
        self, *, user_agent: str, timeout_seconds: float = 1.5, max_attempts: int = 2, backoff_seconds: float = 0.25
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds cannot be negative")
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    def predict_proba_one(self, *, event_id: str, event: dict[str, Any]) -> dict[bool, float]:
        del event
        wiki, separator, revision_id = event_id.partition(":")
        if not separator or not wiki.endswith("wiki") or not revision_id.isdigit():
            raise ValueError("Lift Wing requires an event ID in the form '<language>wiki:<revision_id>'")
        request = Request(
            self.endpoint,
            data=json.dumps({"rev_id": int(revision_id), "lang": wiki.removesuffix("wiki")}).encode(),
            headers={"Content-Type": "application/json", "User-Agent": self.user_agent},
            method="POST",
        )
        for attempt in range(self.max_attempts):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 -- fixed Wikimedia endpoint
                    payload = json.load(response)
                break
            except HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == self.max_attempts - 1:
                    raise
            except (TimeoutError, URLError):
                if attempt == self.max_attempts - 1:
                    raise
            time.sleep(self.backoff_seconds * 2**attempt)
        try:
            risk = float(payload["output"]["probabilities"]["true"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Lift Wing returned an unexpected revert-risk response") from error
        return {False: 1.0 - risk, True: risk}
