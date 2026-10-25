"""Download 2015/16 StatsBomb event data for the top 5 leagues to Parquet.

One Parquet file per match in data/raw/events/<competition_id>/, plus
data/raw/matches.parquet. Re-running skips matches already downloaded.
"""

import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore", module="statsbombpy")

SEASON_ID = 27  # 2015/2016
COMPETITIONS = {
    2: "Premier League",
    11: "La Liga",
    9: "1. Bundesliga",
    12: "Serie A",
    7: "Ligue 1",
}

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def to_json(value):
    """Serialize lists and dicts so mixed nested columns fit in Parquet."""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def clean_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """JSON-encode nested values and stringify columns with mixed types."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(to_json)
            types = {type(v) for v in df[col].dropna()}
            if len(types) > 1:
                df[col] = df[col].map(lambda v: v if pd.isna(v) else str(v))
    return df


def download_match(match_id: int, out_path: Path) -> int:
    events = clean_for_parquet(sb.events(match_id=match_id))
    events["match_id"] = match_id
    events.to_parquet(out_path, index=False)
    return len(events)


def main(workers: int = 8) -> None:
    matches = clean_for_parquet(pd.concat(
        [sb.matches(competition_id=cid, season_id=SEASON_ID) for cid in COMPETITIONS],
        ignore_index=True,
    ))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(RAW_DIR / "matches.parquet", index=False)
    print(matches.groupby("competition").size().to_string())

    todo = []
    for _, m in matches.iterrows():
        comp_id = next(cid for cid, name in COMPETITIONS.items() if name in m.competition)
        out_path = RAW_DIR / "events" / str(comp_id) / f"{m.match_id}.parquet"
        if not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            todo.append((m.match_id, out_path))

    print(f"{len(matches) - len(todo)} already downloaded, {len(todo)} to go")
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_match, mid, path): mid for mid, path in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as exc:
                failed.append(futures[fut])
                print(f"match {futures[fut]} failed: {exc}")
            if i % 100 == 0:
                print(f"{i}/{len(todo)}")

    if failed:
        print(f"{len(failed)} failed, re-run to retry: {failed}")
    else:
        print("All matches downloaded")


if __name__ == "__main__":
    main()
