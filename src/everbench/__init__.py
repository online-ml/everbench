"""Everbench: durable, live predict-then-learn benchmarks."""

from pathlib import Path

from dotenv import load_dotenv

# A local .env is a development convenience. Existing environment variables
# (including Railway variables) always win.
load_dotenv(Path.cwd() / ".env", override=False)
