"""Normalize existing model owner labels.

Revision ID: 0011_normalize_owners
Revises: 0010_simplify_artifacts
Create Date: 2026-09-03
"""

from alembic import op

revision = "0011_normalize_owners"
down_revision = "0010_simplify_artifacts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE benchmark_models SET owner = 'max' WHERE owner <> 'max'")


def downgrade() -> None:
    # Existing owner labels are deliberately not recoverable.
    pass
