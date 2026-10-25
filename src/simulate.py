"""Monte Carlo simulations on shot xG: goal totals and expected points.

Each shot is a weighted coin flip that goes in with probability equal to its xG.
"""

import numpy as np
import pandas as pd

N_SIMS = 10_000
SEED = 42


def simulate_goals(xg, n_sims: int = N_SIMS, seed: int = SEED) -> np.ndarray:
    """Total goals in each of n_sims simulated replays of the given shots."""
    rng = np.random.default_rng(seed)
    xg = np.sort(np.asarray(xg, dtype=float))     # sorted, so the result doesn't depend on shot order
    return (rng.random((n_sims, len(xg))) < xg).sum(axis=1)


def p_at_least(actual: float, sims: np.ndarray) -> float:
    """Share of simulations scoring at least the actual total (one-sided p-value)."""
    return float((sims >= actual).mean())


def p_at_most(actual: float, sims: np.ndarray) -> float:
    """Share of simulations with at most the actual total (for goals conceded)."""
    return float((sims <= actual).mean())


def expected_points(shots: pd.DataFrame, matches: pd.DataFrame, n_sims: int = N_SIMS) -> pd.DataFrame:
    """Expected points per team from simulating every match's shots, plus actual points.

    Uses all shots including penalties. Own goals aren't shots, so they count in
    actual points but not expected points.
    """
    xg_by_team = shots.groupby(["match_id", "team"]).xg.apply(lambda s: np.sort(s.to_numpy())).to_dict()
    rows = []
    for m in matches.itertuples():
        # Seeded per match, so results don't depend on the order of rows in `matches`.
        rng = np.random.default_rng([SEED, int(m.match_id)])
        home_xg = xg_by_team.get((m.match_id, m.home_team), np.array([]))
        away_xg = xg_by_team.get((m.match_id, m.away_team), np.array([]))
        home = (rng.random((n_sims, len(home_xg))) < home_xg).sum(axis=1)
        away = (rng.random((n_sims, len(away_xg))) < away_xg).sum(axis=1)
        p_home, p_draw, p_away = (home > away).mean(), (home == away).mean(), (home < away).mean()
        home_pts = 3 if m.home_score > m.away_score else int(m.home_score == m.away_score)
        away_pts = 3 if m.away_score > m.home_score else int(m.home_score == m.away_score)
        rows.append({"team": m.home_team, "xpts": 3 * p_home + p_draw, "points": home_pts})
        rows.append({"team": m.away_team, "xpts": 3 * p_away + p_draw, "points": away_pts})
    table = pd.DataFrame(rows).groupby("team").sum()
    table["points_minus_xpts"] = table.points - table.xpts
    return table.sort_values("points", ascending=False)
