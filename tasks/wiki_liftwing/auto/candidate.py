"""Generation-zero candidate for wiki-liftwing autonomous research.

The researcher may replace this entire program. The only contract is that
``build_model()`` returns a River classifier that consumes the raw event.
"""

from __future__ import annotations

import ipaddress
import math
from datetime import UTC, datetime
from typing import Any

from river import compose, linear_model, optim, preprocessing


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _anonymous(user: Any) -> float:
    if not isinstance(user, str):
        return 0.0
    if user.startswith("~"):
        return 1.0
    try:
        ipaddress.ip_address(user)
    except ValueError:
        return 0.0
    return 1.0


def features(event: dict[Any, Any]) -> dict[str, float]:
    """Extract a small, legible baseline from the complete raw event."""
    lengths = event.get("length")
    lengths = lengths if isinstance(lengths, dict) else {}
    old_length = _number(lengths.get("old"))
    new_length = _number(lengths.get("new"))
    change = new_length - old_length
    comment = str(event.get("comment") or "")
    lower_comment = comment.casefold()
    user = event.get("user")
    anonymous = _anonymous(user)
    timestamp = _number(event.get("timestamp"))
    moment = datetime.fromtimestamp(timestamp, UTC) if timestamp > 0 else datetime(1970, 1, 1, tzinfo=UTC)
    return {
        "anonymous": anonymous,
        "bot": float(bool(event.get("bot"))),
        "minor": float(bool(event.get("minor"))),
        "log_old_length": math.log1p(max(old_length, 0.0)),
        "log_new_length": math.log1p(max(new_length, 0.0)),
        "signed_log_change": math.copysign(math.log1p(abs(change)), change),
        "log_abs_change": math.log1p(abs(change)),
        "large_change": float(abs(change) >= 500),
        "blanking": float(old_length > 0 and new_length / old_length < 0.1),
        "log_comment_length": math.log1p(len(comment)),
        "empty_comment": float(not comment.strip()),
        "revert_language": float(any(word in lower_comment for word in ("revert", "undo", "undid"))),
        "anonymous_large_change": anonymous * float(abs(change) >= 500),
        "anonymous_empty_comment": anonymous * float(not comment.strip()),
        "hour_sin": math.sin(2 * math.pi * moment.hour / 24),
        "hour_cos": math.cos(2 * math.pi * moment.hour / 24),
        "weekend": float(moment.weekday() >= 5),
    }


def build_model():
    return (
        compose.FuncTransformer(features)
        | preprocessing.StandardScaler()
        | linear_model.LogisticRegression(optimizer=optim.SGD(0.003))
    )
