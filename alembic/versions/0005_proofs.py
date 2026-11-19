"""Proofs and proof batches (ticket C25).

Revision ID: 0005
Revises: 0004
"""
from alembic import op
import sqlalchemy as sa

revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "proof_batches",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("root", sa.String(64), nullable=False, unique=True),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("tx_hash", sa.String(66), nullable=True),
        sa.Column("onchain_id", sa.Integer(), nullable=True),
    )
    op.create_table(
        "proofs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("job_id", sa.String(36), nullable=False, unique=True),
        sa.Column("terms_hash", sa.String(64), nullable=False),
        sa.Column("verdict_hash", sa.String(64), nullable=False),
        sa.Column("payouts", sa.JSON(), nullable=False),
        sa.Column("batch_id", sa.String(36), sa.ForeignKey("proof_batches.id"), nullable=True),
        sa.Column("leaf_index", sa.Integer(), nullable=True),
    )
    op.create_index("ix_proofs_batch_id", "proofs", ["batch_id"])


def downgrade() -> None:
    op.drop_index("ix_proofs_batch_id", table_name="proofs")
    op.drop_table("proofs")
    op.drop_table("proof_batches")
