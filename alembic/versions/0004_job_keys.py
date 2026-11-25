"""The job_keys table: each job's age recipient and its KMS-wrapped identity (ticket A12).

The identity is never stored unwrapped. `wrapped_identity` is set to NULL and `destroyed_at` is stamped when a customer
deletes their data (A14), which makes every ciphertext made for that job unreadable.

Revision ID: 0004
Revises: 0003
"""
from alembic import op
import sqlalchemy as sa

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "job_keys",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("jobs.id"), nullable=False, unique=True),
        sa.Column("recipient", sa.String(80), nullable=False),
        sa.Column("wrapped_identity", sa.LargeBinary(), nullable=True),
        sa.Column("destroyed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("job_keys")
