"""Initial database schema.

Creates the core tables for the Indus Kernel:
- tenants, users, sessions, api_keys
- roles, user_roles
- models, prompts
- memory_metadata, memory_conflicts
- plans, plan_runs, tasks
- tools, plugins
- workflow_defs, automations
- eval_runs, benchmark_runs
- audit_log, llm_calls, webhooks

See ARCHITECTURE.md Section 7.1 for the full DDL.
"""
from __future__ import annotations

"""Alembic migration template — populated by `alembic revision --autogenerate`."""

# revision identifiers, used by Alembic.
revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply the migration."""
    # See ARCHITECTURE.md Section 7.1 for the full DDL.
    # In M1 this will be filled in by `alembic revision --autogenerate`.
    pass


def downgrade() -> None:
    """Reverse the migration."""
    pass
