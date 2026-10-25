"""Load raw Parquet events into DuckDB and run sql/ files in order."""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "leicester_xg.duckdb"
RAW_DIR = ROOT / "data" / "raw"
SQL_DIR = ROOT / "sql"


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"""
        CREATE OR REPLACE TABLE events AS
        SELECT * FROM read_parquet('{RAW_DIR / "events" / "*" / "*.parquet"}', union_by_name = true)
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE matches AS
        SELECT * FROM read_parquet('{RAW_DIR / "matches.parquet"}')
    """)
    print(f"events: {con.execute('SELECT count(*) FROM events').fetchone()[0]:,} rows")

    for sql_file in sorted(SQL_DIR.glob("*.sql")):
        con.execute(sql_file.read_text())
        print(f"ran {sql_file.name}")

    con.close()


if __name__ == "__main__":
    main()
