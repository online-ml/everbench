"""Add trusted model artifacts and archive tracking.

Revision ID: 0002_artifacts_and_archives
Revises: 0001_initial
Create Date: 2026-09-02
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0002_artifacts_and_archives"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("benchmark_events", sa.Column("archived_at", sa.DateTime(timezone=True)))
    op.add_column("benchmark_models", sa.Column("artifact_id", sa.String()))
    op.create_table(
        "model_artifacts",
        sa.Column("artifact_id", sa.String(), primary_key=True),
        sa.Column("sha256", sa.String(64), unique=True, nullable=False),
        sa.Column("serializer", sa.String(), nullable=False),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        sa.Column("signature", sa.String(64), nullable=False),
        sa.Column("trusted", sa.Boolean(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_table(
        "model_snapshots",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("model_id", sa.String(), primary_key=True),
        sa.Column("version", sa.Integer(), primary_key=True),
        sa.Column("artifact_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_table(
        "archive_manifest",
        sa.Column("content_sha256", sa.String(64), primary_key=True),
        sa.Column("task_name", sa.String(), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("archive_manifest")
    op.drop_table("model_snapshots")
    op.drop_table("model_artifacts")
    op.drop_column("benchmark_models", "artifact_id")
    op.drop_column("benchmark_events", "archived_at")
