"""account tables move to the platform service

Revision ID: 0002
Revises: 0001
Created: 2026-11-16 15:46:09.094307
"""

from alembic import op
import sqlalchemy as sa

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Sign-in, teams and API keys now live in the Platform API, in its own schema. Children first, so foreign keys
    # never point at a dropped table; a table's indexes go with it.
    for table in ("org_members", "api_keys", "job_owners", "consents", "web_sessions", "user_emails", "login_tokens", "orgs"):
        op.drop_table(table)


def downgrade() -> None:
    # Recreate the tables (empty); orgs first, since the others refer to it.
    op.create_table('orgs',
    sa.Column('id', sa.VARCHAR(length=32), nullable=False),
    sa.Column('name', sa.VARCHAR(length=80), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('org_members',
    sa.Column('org_id', sa.VARCHAR(length=32), nullable=False),
    sa.Column('user_id', sa.VARCHAR(length=36), nullable=False),
    sa.Column('role', sa.VARCHAR(length=16), nullable=False),
    sa.Column('id', sa.VARCHAR(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['org_id'], ['orgs.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('org_id', 'user_id')
    )
    with op.batch_alter_table('org_members', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_org_members_user_id'), ['user_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_org_members_org_id'), ['org_id'], unique=False)

    op.create_table('consents',
    sa.Column('user_id', sa.VARCHAR(length=36), nullable=False),
    sa.Column('purpose', sa.VARCHAR(length=40), nullable=False),
    sa.Column('granted', sa.BOOLEAN(), nullable=False),
    sa.Column('policy_version', sa.VARCHAR(length=40), nullable=False),
    sa.Column('id', sa.VARCHAR(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('consents', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_consents_user_id'), ['user_id'], unique=False)

    op.create_table('job_owners',
    sa.Column('job_id', sa.VARCHAR(length=36), nullable=False),
    sa.Column('org_id', sa.VARCHAR(length=32), nullable=False),
    sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ),
    sa.ForeignKeyConstraint(['org_id'], ['orgs.id'], ),
    sa.PrimaryKeyConstraint('job_id')
    )
    with op.batch_alter_table('job_owners', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_job_owners_org_id'), ['org_id'], unique=False)

    op.create_table('web_sessions',
    sa.Column('user_id', sa.VARCHAR(length=36), nullable=False),
    sa.Column('token_hash', sa.VARCHAR(length=64), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('id', sa.VARCHAR(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('token_hash')
    )
    with op.batch_alter_table('web_sessions', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_web_sessions_user_id'), ['user_id'], unique=False)

    op.create_table('login_tokens',
    sa.Column('email', sa.VARCHAR(length=320), nullable=False),
    sa.Column('token_hash', sa.VARCHAR(length=64), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('used_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('id', sa.VARCHAR(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('token_hash')
    )
    with op.batch_alter_table('login_tokens', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_login_tokens_email'), ['email'], unique=False)

    op.create_table('api_keys',
    sa.Column('id', sa.VARCHAR(length=32), nullable=False),
    sa.Column('org_id', sa.VARCHAR(length=32), nullable=False),
    sa.Column('name', sa.VARCHAR(length=60), nullable=False),
    sa.Column('mode', sa.VARCHAR(length=8), nullable=False),
    sa.Column('prefix', sa.VARCHAR(length=16), nullable=False),
    sa.Column('key_hash', sa.VARCHAR(length=64), nullable=False),
    sa.Column('scopes', sa.JSON(), nullable=False),
    sa.Column('created_by', sa.VARCHAR(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['org_id'], ['orgs.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('key_hash')
    )
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_api_keys_org_id'), ['org_id'], unique=False)

    op.create_table('user_emails',
    sa.Column('user_id', sa.VARCHAR(length=36), nullable=False),
    sa.Column('email', sa.VARCHAR(length=320), nullable=False),
    sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('id', sa.VARCHAR(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email')
    )
    with op.batch_alter_table('user_emails', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_user_emails_user_id'), ['user_id'], unique=False)
