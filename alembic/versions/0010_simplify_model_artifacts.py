"""Remove unused artifact metadata and snapshot versioning.

Revision ID: 0010_simplify_artifacts
Revises: 0009_remove_dead_storage
Create Date: 2026-09-03
"""

import sqlalchemy as sa

from alembic import op

revision = "0010_simplify_artifacts"
down_revision = "0009_remove_dead_storage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("model_artifacts", "trusted")
    op.drop_column("model_artifacts", "serializer")
    op.drop_constraint("model_snapshots_pkey", "model_snapshots", type_="primary")
    op.drop_column("model_snapshots", "version")
    op.create_primary_key("model_snapshots_pkey", "model_snapshots", ["task_name", "model_id"])


def downgrade() -> None:
    op.drop_constraint("model_snapshots_pkey", "model_snapshots", type_="primary")
    op.add_column("model_snapshots", sa.Column("version", sa.Integer(), nullable=False, server_default="1"))
    op.alter_column("model_snapshots", "version", server_default=None)
    op.create_primary_key("model_snapshots_pkey", "model_snapshots", ["task_name", "model_id", "version"])
    op.add_column("model_artifacts", sa.Column("serializer", sa.String(), nullable=False, server_default="cloudpickle"))
    op.add_column("model_artifacts", sa.Column("trusted", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.alter_column("model_artifacts", "serializer", server_default=None)
    op.alter_column("model_artifacts", "trusted", server_default=None)
