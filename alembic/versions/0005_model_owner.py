"""Add a required owner to every model registration.

Revision ID: 0005_model_owner
Revises: 0004_task_metrics
Create Date: 2026-09-02
"""

import sqlalchemy as sa

from alembic import op

revision = "0005_model_owner"
down_revision = "0004_task_metrics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("benchmark_models", sa.Column("owner", sa.String(), nullable=False, server_default="unassigned"))
    op.alter_column("benchmark_models", "owner", server_default=None)


def downgrade() -> None:
    op.drop_column("benchmark_models", "owner")
