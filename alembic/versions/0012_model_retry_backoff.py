"""Add retry backoff and durable model error counters.

Revision ID: 0012_model_retry_backoff
Revises: 0011_normalize_owners
Create Date: 2026-09-03
"""

import sqlalchemy as sa

from alembic import op

revision = "0012_model_retry_backoff"
down_revision = "0011_normalize_owners"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("benchmark_models", sa.Column("disabled_until", sa.DateTime(timezone=True)))
    op.add_column(
        "benchmark_models", sa.Column("prediction_errors", sa.BigInteger(), nullable=False, server_default="0")
    )
    op.add_column("benchmark_models", sa.Column("label_errors", sa.BigInteger(), nullable=False, server_default="0"))
    op.alter_column("benchmark_models", "prediction_errors", server_default=None)
    op.alter_column("benchmark_models", "label_errors", server_default=None)


def downgrade() -> None:
    op.drop_column("benchmark_models", "label_errors")
    op.drop_column("benchmark_models", "prediction_errors")
    op.drop_column("benchmark_models", "disabled_until")
