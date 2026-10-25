"""Load raw Parquet events into DuckDB and run sql/ files in order."""

from pathlib import Path

import duckdb

from freeze_frame import build_freeze_frame_table

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
    # Some events use short team names ("Marseille", "Caen") for the same team_id.
    # Use the matches table's name for every team_id so each team has one name.
    con.execute("""
        CREATE OR REPLACE TABLE team_names AS
        SELECT DISTINCT home_team_id AS team_id, home_team AS team_name FROM matches
        UNION
        SELECT DISTINCT away_team_id, away_team FROM matches
    """)
    con.execute("""
        UPDATE events SET team = n.team_name
        FROM team_names n WHERE events.team_id = n.team_id AND events.team != n.team_name
    """)
    con.execute("""
        UPDATE events SET possession_team = n.team_name
        FROM team_names n WHERE events.possession_team_id = n.team_id AND events.possession_team != n.team_name
    """)
    print(f"events: {con.execute('SELECT count(*) FROM events').fetchone()[0]:,} rows")

    for sql_file in sorted(SQL_DIR.glob("*.sql")):
        con.execute(sql_file.read_text())
        print(f"ran {sql_file.name}")
        # Freeze frame features are geometry on player lists, so they're built in
        # Python once shots exist and before the final shot table joins them in.
        if sql_file.name == "01_shots.sql":
            build_freeze_frame_table(con)
            print("built freeze_frame_features")

    # Final scores, for expected points. Small enough to commit alongside shots_with_xg.
    con.execute(f"""
        COPY (SELECT match_id, competition, match_date, home_team, away_team, home_score, away_score FROM matches)
        TO '{ROOT / "data" / "processed" / "matches.parquet"}' (FORMAT parquet)
    """)
    # Minutes per formation and game state, for comparing formations per 90.
    con.execute(f"""
        COPY formation_minutes TO '{ROOT / "data" / "processed" / "formation_minutes.parquet"}' (FORMAT parquet)
    """)

    n = con.execute("SELECT count(*) FROM shot_table").fetchone()[0]
    print(f"shot_table: {n:,} shots")
    con.close()


if __name__ == "__main__":
    main()
