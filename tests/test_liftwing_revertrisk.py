from __future__ import annotations

import json
from io import BytesIO
from typing import Any, cast
from urllib.error import HTTPError

import pytest

from everbench.models import PredictionUnavailable
from tasks.wiki_liftwing.examples import liftwing_revertrisk


def http_error(*, detail: str) -> HTTPError:
    body = BytesIO(json.dumps({"detail": detail}).encode())
    return HTTPError("https://example.test", 422, "Unprocessable Entity", cast(Any, {}), cast(Any, body))


def test_expected_422_is_a_prediction_abstention(*, monkeypatch: pytest.MonkeyPatch) -> None:
    detail = "Could not make prediction for revisions dict_keys([(123, 'en')]). Reason: ['parent_revision_missing']"

    def reject(*_args, **_kwargs):
        raise http_error(detail=detail)

    monkeypatch.setattr(liftwing_revertrisk, "urlopen", reject)
    model = liftwing_revertrisk.LiftWingRevertRisk(user_agent="everbench-test", max_attempts=1)

    with pytest.raises(PredictionUnavailable, match=r"enwiki:123.*parent_revision_missing"):
        model.predict_proba_one(event_id="enwiki:123", event={})


def test_unexpected_422_keeps_event_and_response_context(*, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*_args, **_kwargs):
        raise http_error(detail="request schema changed")

    monkeypatch.setattr(liftwing_revertrisk, "urlopen", reject)
    model = liftwing_revertrisk.LiftWingRevertRisk(user_agent="everbench-test", max_attempts=1)

    with pytest.raises(RuntimeError, match=r"enwiki:456.*HTTP 422.*request schema changed"):
        model.predict_proba_one(event_id="enwiki:456", event={})
