"""Example-generation requests (ticket #154, contract distill-gen/v1).

Revision ID: 0006
Revises: 0005
"""
from alembic import op
import sqlalchemy as sa

revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'distillation_jobs',
        sa.Column('job_id', sa.String(36), primary_key=True),
        sa.Column('distillation_id', sa.String(64), nullable=False),
        sa.Column('attempt', sa.Integer, nullable=False),
        sa.Column('state', sa.String(16), nullable=False),
        sa.Column('spec', sa.JSON, nullable=False),
        sa.Column('spec_sha256', sa.String(64), nullable=False),
        sa.Column('seeds_jsonl', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('distillation_id', 'attempt', name='uq_distillation_attempt'),
    )
    op.create_index('ix_distillation_jobs_distillation_id', 'distillation_jobs', ['distillation_id'])
    op.create_index('ix_distillation_jobs_state', 'distillation_jobs', ['state'])


def downgrade() -> None:
    op.drop_index('ix_distillation_jobs_state', 'distillation_jobs')
    op.drop_index('ix_distillation_jobs_distillation_id', 'distillation_jobs')
    op.drop_table('distillation_jobs')
