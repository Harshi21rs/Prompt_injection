import sys
from pathlib import Path

# Ensure project root is on sys.path so package imports work when running as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    dialect = engine.dialect.name
    print("dialect=" + dialect)
    rows = []
    if dialect == "mysql":
        conn.execute(text("USE behavior_anomaly_detector"))
        res = conn.execute(text("SHOW TABLES"))
        rows = [tuple(r)[0] for r in res]
    elif dialect == "sqlite":
        res = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        )
        rows = [r[0] for r in res]
    else:
        print("Unsupported dialect for SHOW TABLES: " + dialect)

    print("tables:")
    for r in rows:
        print(r)
