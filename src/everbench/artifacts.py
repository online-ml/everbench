"""Trusted pickle artifact handling.

Pickle is code execution, not a safe data format. This module never attempts
to inspect or validate a pickle. It verifies provenance with an HMAC before a
worker unpickles an artifact; the HTTP API only stores opaque bytes.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

import cloudpickle


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def signing_key() -> bytes:
    key = os.getenv("EVERBENCH_MODEL_SIGNING_KEY")
    if not key:
        raise RuntimeError("EVERBENCH_MODEL_SIGNING_KEY is required for pickle artifacts")
    return key.encode()


def sign(payload: bytes) -> str:
    return hmac.new(signing_key(), sha256(payload).encode(), hashlib.sha256).hexdigest()


def verify(payload: bytes, signature: str) -> bool:
    return hmac.compare_digest(sign(payload), signature)


def dumps(model: Any) -> bytes:
    """Serialize a model together with classes defined in its upload module."""
    return cloudpickle.dumps(model)


def loads(payload: bytes, signature: str) -> Any:
    if not verify(payload, signature):
        raise ValueError("pickle artifact signature does not verify")
    # Only call after an authenticated upload or a worker-created snapshot.
    # cloudpickle also reads ordinary pickle payloads, keeping existing
    # artifacts compatible after the serializer switch.
    return cloudpickle.loads(payload)
