"""Materialize legacy checkpoints once, then keep only the ready sequence.

Revision ID: 0022_sequence_checkpoints
Revises: 0021_observation_resolutions
"""

from uuid import uuid4

import sqlalchemy as sa

from alembic import op

revision = "0022_sequence_checkpoints"
down_revision = "0021_observation_resolutions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Run with workers stopped. Only legacy snapshots require loading a signed
    # model and replaying committed training. Modern sequence snapshots are untouched.
    connection = op.get_bind()
    legacy = connection.execute(
        sa.text("""
        SELECT snapshot.*, model.label_cursor_sequence, artifact.payload, artifact.signature
        FROM model_snapshots AS snapshot
        JOIN benchmark_models AS model USING (task_name, model_id)
        JOIN model_artifacts AS artifact ON artifact.artifact_id = snapshot.artifact_id
        WHERE snapshot.checkpoint_ready_sequence IS NULL
    """),
        execution_options={"yield_per": 1},
    ).mappings()
    for snapshot in legacy:
        if snapshot["checkpoint_label_available_at"] is not None:
            from everbench import artifacts

            model = artifacts.loads(payload=snapshot["payload"], signature=snapshot["signature"])
            learner = getattr(model, "learn_one", None)
            if callable(learner):
                rows = connection.execute(
                    sa.text("""
                    SELECT event.event_id, event.event, label.y
                    FROM benchmark_model_events AS state
                    JOIN benchmark_events AS event USING (task_name, event_id)
                    JOIN benchmark_labels AS label USING (task_name, event_id)
                    WHERE state.task_name = :task AND state.model_id = :model AND state.trained_at IS NOT NULL
                      AND NOT state.training_skipped AND label.y <> 'null'::jsonb
                      AND (label.available_at, event.sequence) > (:available_at, :sequence)
                    ORDER BY label.available_at, event.sequence
                """),
                    {
                        "task": snapshot["task_name"],
                        "model": snapshot["model_id"],
                        "available_at": snapshot["checkpoint_label_available_at"],
                        "sequence": snapshot["checkpoint_event_sequence"],
                    },
                    execution_options={"yield_per": 500},
                )
                for event_id, event, target in rows:
                    learner(event_id=event_id, event=event, label=target)
            payload = artifacts.dumps(model=model)
            checksum = artifacts.sha256(payload=payload)
            identifier = str(uuid4())
            connection.execute(
                sa.text("""
                INSERT INTO model_artifacts (artifact_id, sha256, payload, signature, metadata)
                VALUES (:id, :sha, :payload, :signature, '{"source":"checkpoint-migration"}'::jsonb)
                ON CONFLICT (sha256) DO NOTHING
            """),
                {"id": identifier, "sha": checksum, "payload": payload, "signature": artifacts.sign(payload=payload)},
            )
            connection.execute(
                sa.text("""
                UPDATE model_snapshots SET artifact_id = (SELECT artifact_id FROM model_artifacts WHERE sha256 = :sha)
                WHERE task_name = :task AND model_id = :model
            """),
                {"sha": checksum, "task": snapshot["task_name"], "model": snapshot["model_id"]},
            )
        connection.execute(
            sa.text("""
            UPDATE model_snapshots SET checkpoint_ready_sequence = :sequence
            WHERE task_name = :task AND model_id = :model
        """),
            {
                "sequence": snapshot["label_cursor_sequence"],
                "task": snapshot["task_name"],
                "model": snapshot["model_id"],
            },
        )
    op.alter_column("model_snapshots", "checkpoint_ready_sequence", nullable=False)
    op.drop_column("model_snapshots", "checkpoint_label_available_at")
    op.drop_column("model_snapshots", "checkpoint_event_sequence")


def downgrade() -> None:
    op.add_column("model_snapshots", sa.Column("checkpoint_label_available_at", sa.DateTime(timezone=True)))
    op.add_column("model_snapshots", sa.Column("checkpoint_event_sequence", sa.BigInteger()))
    op.alter_column("model_snapshots", "checkpoint_ready_sequence", nullable=True)
