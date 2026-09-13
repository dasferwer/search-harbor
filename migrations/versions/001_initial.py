from pathlib import Path

from alembic import op

revision = "001"
down_revision = None


def upgrade():
    sql = Path("infra/schema.sql").read_text()
    op.get_bind().connection.run_async(lambda conn: conn.execute(sql))


def downgrade():
    raise RuntimeError("Use a database backup to restore catalog history")
