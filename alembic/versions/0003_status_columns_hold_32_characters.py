"""Status, state and similar columns hold 32 characters instead of 16 (or 8).

Values such as AWAITING_SIGNATURES (19) were written into String(16) columns. SQLite never enforces a length, so it
went unnoticed; Postgres refuses it (StringDataRightTruncation), which stopped the live server's data copy on
16 Nov. Widening a varchar is metadata-only in Postgres and loses nothing. On SQLite the batch mode rebuilds each
table. The downgrade narrows them back and fails if a longer value has been stored since.

Revision ID: 0003
Revises: 0002
"""
from alembic import op
import sqlalchemy as sa

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None

# (table, column, old length)
COLUMNS = [
    ("outbox", "status", 16),
    ("wallets", "role", 16),
    ("jobs", "state", 16),
    ("providers", "status", 16),
    ("providers", "kind", 16),
    ("adaptation_manifests", "verdict", 16),
    ("escrow_refs", "state", 16),
    ("job_assignments", "state", 16),
    ("job_transitions", "from_state", 16),
    ("job_transitions", "to_state", 16),
    ("settlements", "status", 16),
    ("workers", "state", 16),
    ("telemetry_samples", "source", 16),
    ("training_runs", "status", 16),
    ("checkpoints", "state", 16),
    ("quality_evaluations", "verdict", 8),
]


def _alter(to_old: bool) -> None:
    tables: dict[str, list[tuple[str, int]]] = {}
    for table, column, old in COLUMNS:
        tables.setdefault(table, []).append((column, old))
    for table, columns in tables.items():
        with op.batch_alter_table(table) as batch:
            for column, old in columns:
                new, existing = (old, 32) if to_old else (32, old)
                batch.alter_column(column, type_=sa.String(new), existing_type=sa.String(existing), existing_nullable=False)


def upgrade() -> None:
    _alter(to_old=False)


def downgrade() -> None:
    _alter(to_old=True)
