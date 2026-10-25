"""Shared fixtures.

Tests that need the DuckDB database (built from the raw download) are skipped when it
isn't there, e.g. in CI. Everything else runs on the committed data/processed/ files.
"""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "leicester_xg.duckdb"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"


@pytest.fixture(scope="session")
def db():
    if not DB_PATH.exists():
        pytest.skip("DuckDB not built (run src/download.py and src/build_db.py)")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    yield con
    con.close()


@pytest.fixture(scope="session")
def shots() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "shots_with_xg.parquet")


@pytest.fixture(scope="session")
def np_shots(shots) -> pd.DataFrame:
    return shots[~shots.is_penalty]


@pytest.fixture(scope="session")
def matches() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "matches.parquet")


@pytest.fixture(scope="session")
def formation_minutes() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "formation_minutes.parquet")
