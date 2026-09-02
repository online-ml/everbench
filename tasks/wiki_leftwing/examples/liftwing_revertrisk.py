"""Build a scoring-only Lift Wing predictor for the wiki-leftwing task.

This file is deliberately outside Everbench's runtime. It is an ordinary
user-owned model definition which cloudpickle embeds in ``liftwing.pkl``.

Example:
    uv run python tasks/wiki_leftwing/examples/liftwing_revertrisk.py --user-agent 'name (email)'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import cloudpickle


class LiftWingRevertRisk:
    """Return Lift Wing's probability that a Wikipedia revision is reverted."""

    uses_event_context = True
    endpoint = "https://api.wikimedia.org/service/lw/inference/v1/models/revertrisk-language-agnostic:predict"

    def __init__(self, user_agent: str, timeout_seconds: float = 10.0):
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def predict_event(self, event_id: str, features: dict[str, float]) -> float:
        del features
        wiki, separator, revision_id = event_id.partition(":")
        if not separator or not wiki.endswith("wiki") or not revision_id.isdigit():
            raise ValueError("Lift Wing requires an event ID in the form '<language>wiki:<revision_id>'")
        request = Request(
            self.endpoint,
            data=json.dumps({"rev_id": int(revision_id), "lang": wiki.removesuffix("wiki")}).encode(),
            headers={"Content-Type": "application/json", "User-Agent": self.user_agent},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 -- fixed Wikimedia endpoint
            payload = json.load(response)
        try:
            return float(payload["output"]["probabilities"]["true"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Lift Wing returned an unexpected revert-risk response") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-agent", required=True, help="Descriptive contact User-Agent for Wikimedia")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("liftwing.pkl"))
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    args.output.write_bytes(cloudpickle.dumps(LiftWingRevertRisk(args.user_agent, args.timeout_seconds)))
    print(args.output)


if __name__ == "__main__":
    main()
