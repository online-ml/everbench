"""Store archive sizes alongside their manifests.

Revision ID: 0007_archive_byte_size
Revises: 0006_runtime_hardening
Create Date: 2026-09-02
"""

import sqlalchemy as sa

from alembic import op

revision = "0007_archive_byte_size"
down_revision = "0006_runtime_hardening"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("archive_manifest", sa.Column("byte_size", sa.BigInteger()))


def downgrade() -> None:
    op.drop_column("archive_manifest", "byte_size")
