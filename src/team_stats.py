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


GAME_STATES = ["-2 or worse", "-1", "0", "+1", "+2 or better"]
FLIP_STATE = dict(zip(GAME_STATES, GAME_STATES[::-1], strict=True))
SET_PIECES = ["From Corner", "From Free Kick", "From Throw In"]
ADDITIVE = ["shots", "xg", "goals", "counter_xg", "set_piece_xg", "cross_xg", "through_ball_xg",
            "shots_against", "xga", "goals_against"]


def load_formation_minutes() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "formation_minutes.parquet")


def formation_cells(shots: pd.DataFrame, minutes: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Minutes plus shot totals for and against, per group + formation + game state.

    Shots against use the defending team's formation and its own view of the score.
    """
    s = shots[~shots.is_penalty]
    s = s.assign(counter_xg=s.xg.where(s.play_pattern == "From Counter", 0),
                 set_piece_xg=s.xg.where(s.play_pattern.isin(SET_PIECES), 0),
                 cross_xg=s.xg.where(s.assist_type == "Cross", 0),
                 through_ball_xg=s.xg.where(s.assist_type == "Through Ball", 0))
    keys = group + ["formation", "game_state"]
    attack = s.groupby(keys).agg(shots=("shot_id", "size"), xg=("xg", "sum"), goals=("is_goal", "sum"),
                                 counter_xg=("counter_xg", "sum"), set_piece_xg=("set_piece_xg", "sum"),
                                 cross_xg=("cross_xg", "sum"), through_ball_xg=("through_ball_xg", "sum"))
    defending = s.assign(team=s.opponent, formation=s.opponent_formation, game_state=s.game_state.map(FLIP_STATE))
    defense = defending.groupby(keys).agg(shots_against=("shot_id", "size"), xga=("xg", "sum"),
                                          goals_against=("is_goal", "sum"))
    mins = minutes.groupby(keys).minutes.sum()
    return pd.concat([mins, attack, defense], axis=1).fillna(0)


def per90(cells: pd.DataFrame, weights: pd.Series | None, min_state_minutes: float) -> pd.Series:
    """Per-90 rates for one formation from its game state cells.

    weights=None gives raw rates (totals / minutes). Otherwise each game state's rate is
    weighted by `weights`, so formations are compared at the same mix of scorelines.
    """
    c = cells[cells.minutes >= min_state_minutes]
    if c.empty:
        return pd.Series(dtype=float)
    if weights is None:
        rates = c[ADDITIVE].sum() / c.minutes.sum() * 90
    else:
        w = weights.reindex(c.index).fillna(0)
        if w.sum() == 0:
            return pd.Series(dtype=float)
        rates = (c[ADDITIVE].div(c.minutes, axis=0) * 90).mul(w / w.sum(), axis=0).sum()
    def ratio(a: float, b: float) -> float:
        return a / b if b else float("nan")    # e.g. no shots at all in a rare game state

    return pd.Series({
        "minutes": cells.minutes.sum(),
        "shots_per90": rates.shots,
        "xg_per90": rates.xg,
        "goals_per90": rates.goals,
        "xg_per_shot": ratio(rates.xg, rates.shots),
        "shots_against_per90": rates.shots_against,
        "xga_per90": rates.xga,
        "counter_share": ratio(rates.counter_xg, rates.xg),
        "set_piece_share": ratio(rates.set_piece_xg, rates.xg),
        "cross_share": ratio(rates.cross_xg, rates.xg),
        "through_ball_share": ratio(rates.through_ball_xg, rates.xg),
    })


def formation_profile(shots: pd.DataFrame, minutes: pd.DataFrame, *, competition: str | None = None,
                      game_state: str | None = None, adjust: bool = True, team: str | None = None,
                      exclude_team: str | None = None, min_state_minutes: float = 90) -> pd.DataFrame:
    """One row per formation: volume, quality and xG mix, per 90 minutes.

    game_state picks one scoreline. Otherwise adjust=True weights every formation by the
    same game state mix (all minutes in scope), so a formation used mostly while losing
    isn't credited with chasing-the-game shots.
    """
    if competition:
        shots, minutes = shots[shots.competition == competition], minutes[minutes.competition == competition]
    weights = minutes.groupby("game_state").minutes.sum()
    cells = formation_cells(shots, minutes, ["team"])
    teams = cells.index.get_level_values("team")
    if team:
        cells = cells[teams == team]
    if exclude_team:
        cells = cells[teams != exclude_team]
    cells = cells.groupby(["formation", "game_state"]).sum()
    if game_state:
        cells = cells[cells.index.get_level_values("game_state") == game_state]
        weights = None
    elif not adjust:
        weights = None
    rows = {f: per90(c.droplevel("formation"), weights, min_state_minutes) for f, c in cells.groupby("formation")}
    return pd.DataFrame(rows).T.dropna(how="all").sort_values("minutes", ascending=False)


def team_formation_profiles(shots: pd.DataFrame, minutes: pd.DataFrame, formation: str,
                            min_minutes: float = 900) -> pd.DataFrame:
    """Game-state-adjusted profile of every team that used `formation` for at least min_minutes."""
    weights = minutes.groupby("game_state").minutes.sum()
    cells = formation_cells(shots, minutes, ["competition", "team"])
    cells = cells[cells.index.get_level_values("formation") == formation].droplevel("formation")
    rows = {}
    for (competition, team), c in cells.groupby(["competition", "team"]):
        if c.minutes.sum() >= min_minutes:
            rows[(competition, team)] = per90(c.droplevel(["competition", "team"]), weights, 90)
    return pd.DataFrame(rows).T.rename_axis(["competition", "team"])


ON_TARGET = ["Goal", "Saved", "Saved to Post"]


def load_funnel() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "funnel.parquet")


def defense_funnel(shots: pd.DataFrame, funnel: pd.DataFrame, competition: str) -> pd.DataFrame:
    """Split each defending team's goals prevented (xG against - goals) into three stages.

    blocks:        xG against - expected goals given which shots were actually blocked
    missed_target: that - expected goals given which shots actually hit the target
    goalkeeping:   that - goals actually conceded
    The three add up exactly to goals prevented. Non-penalty shots.
    """
    s = shots[(shots.competition == competition) & ~shots.is_penalty].merge(funnel, on="shot_id")
    got_through = s.outcome != "Blocked"
    on_target = s.outcome.isin(ON_TARGET)
    s = s.assign(after_blocks=(s.p_on_target * s.p_goal_on_target).where(got_through, 0.0),
                 after_misses=s.p_goal_on_target.where(on_target, 0.0),
                 blocked=(~got_through).astype(int), on_target=on_target.astype(int))
    t = s.groupby("opponent").agg(shots_against=("shot_id", "size"), xga=("xg", "sum"),
                                  after_blocks=("after_blocks", "sum"), after_misses=("after_misses", "sum"),
                                  goals_against=("is_goal", "sum"), blocked=("blocked", "sum"),
                                  expected_blocked=("p_not_blocked", lambda p: (1 - p).sum()),
                                  on_target=("on_target", "sum")).rename_axis("team")
    t["blocks"] = t.xga - t.after_blocks
    t["missed_target"] = t.after_blocks - t.after_misses
    t["goalkeeping"] = t.after_misses - t.goals_against
    t["goals_prevented"] = t.xga - t.goals_against
    return t
