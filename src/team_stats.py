"""Team-level breakdowns of shots_with_xg, shared by the analysis scripts and the app.

Only needs pandas, so the app can import it without the training stack.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

PL = "England - Premier League"
LEICESTER = "Leicester City"
CHAMPIONS = {
    "England - Premier League": "Leicester City",
    "Spain - La Liga": "Barcelona",
    "Italy - Serie A": "Juventus",
    "France - Ligue 1": "Paris Saint-Germain",
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    shots = pd.read_parquet(PROCESSED_DIR / "shots_with_xg.parquet")
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    return shots, matches


def team_breakdown(shots: pd.DataFrame, matches: pd.DataFrame, competition: str) -> pd.DataFrame:
    """Volume, quality and finishing for and against, per team. Non-penalty shots."""
    s = shots[(shots.competition == competition) & ~shots.is_penalty]
    m = matches[matches.competition == competition]
    games = pd.concat([m.home_team, m.away_team]).value_counts().rename("matches")
    attack = s.groupby("team").agg(shots=("shot_id", "size"), xg=("xg", "sum"), goals=("is_goal", "sum"))
    defense = s.groupby("opponent").agg(shots_against=("shot_id", "size"), xga=("xg", "sum"),
                                        goals_against=("is_goal", "sum"))
    pens = shots[(shots.competition == competition) & shots.is_penalty].groupby("team").agg(
        penalties=("shot_id", "size"), penalty_goals=("is_goal", "sum"))
    t = pd.concat([games, attack, defense.rename_axis("team"), pens], axis=1).fillna(0)
    t["shots_per_match"] = t.shots / t.matches
    t["xg_per_shot"] = t.xg / t.shots
    t["finishing"] = t.goals - t.xg
    t["shots_against_per_match"] = t.shots_against / t.matches
    t["xga_per_shot"] = t.xga / t.shots_against
    t["goals_prevented"] = t.xga - t.goals_against
    t["xg_diff"] = t.xg - t.xga
    return t


def rank(t: pd.DataFrame, col: str, team: str = LEICESTER, ascending: bool = False) -> int:
    return int(t[col].rank(ascending=ascending, method="min")[team])


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def share_vs_league(shots: pd.DataFrame, column: str, team: str = LEICESTER, competition: str = PL,
                    side: str = "team") -> pd.DataFrame:
    """Share of a team's non-penalty xG from each category, vs the rest of its league.

    side="opponent" gives the xG the team conceded instead of created.
    """
    s = shots[(shots.competition == competition) & ~shots.is_penalty]
    is_team = s[side] == team
    out = pd.DataFrame({
        "team_xg": s[is_team].groupby(column).xg.sum(),
        "team_share": s[is_team].groupby(column).xg.sum() / s[is_team].xg.sum(),
        "rest_of_league_share": s[~is_team].groupby(column).xg.sum() / s[~is_team].xg.sum(),
    }).fillna(0)
    out["ratio"] = out.team_share / out.rest_of_league_share
    return out.sort_values("team_xg", ascending=False)


def champion_summary(shots: pd.DataFrame, matches: pd.DataFrame, n_sims: int = 10_000) -> pd.DataFrame:
    """Each champion vs its own league's average (leagues differ in scoring)."""
    from simulate import expected_points

    rows = []
    for competition, team in CHAMPIONS.items():
        t = team_breakdown(shots, matches, competition)
        avg, c = t.mean(), t.loc[team]
        league_shots = shots[shots.competition == competition]
        xpts = expected_points(league_shots, matches[matches.competition == competition], n_sims=n_sims)
        rows.append({
            "team": team,
            "league": competition.split(" - ")[1],
            "matches": int(c.matches),
            "shots_per_match": c.shots_per_match,
            "shots_vs_league": c.shots_per_match / avg.shots_per_match - 1,
            "xg_per_shot": c.xg_per_shot,
            "quality_vs_league": c.xg_per_shot / avg.xg_per_shot - 1,
            "goals": c.goals,
            "xg": c.xg,
            "finishing": c.finishing,
            "shots_against_vs_league": c.shots_against_per_match / avg.shots_against_per_match - 1,
            "quality_against_vs_league": c.xga_per_shot / avg.xga_per_shot - 1,
            "goals_against": c.goals_against,
            "xga": c.xga,
            "goals_prevented": c.goals_prevented,
            "xg_diff_per_match": c.xg_diff / c.matches,
            "points": xpts.loc[team, "points"],
            "xpts": xpts.loc[team, "xpts"],
            "xpts_rank": int(xpts.xpts.rank(ascending=False, method="min")[team]),
            "counter_xg_ratio": share_vs_league(shots, "play_pattern", team, competition).loc["From Counter", "ratio"],
        })
    return pd.DataFrame(rows).set_index("team")
