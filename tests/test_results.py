"""The README and results files must match what the code produces today.

Chain of checks: README numbers -> figures/*_results.md -> recomputed from data/processed/.
"""

import re

import numpy as np
import pytest

from conftest import FIGURES, ROOT
from simulate import expected_points, p_at_least, p_at_most, simulate_goals
from team_stats import (
    LEICESTER,
    PL,
    champion_summary,
    defense_funnel,
    formation_profile,
    load_funnel,
    team_breakdown,
)

README = (ROOT / "README.md").read_text()
DOCS = README + (ROOT / "docs" / "interview_prep.md").read_text()
RESULTS = " ".join(p.read_text() for p in FIGURES.glob("*.md"))
# Numbers that describe the project or illustrate a concept rather than report a result.
NOT_RESULTS = {"3.11", "3.13", "5.4", "0.2", "20%", "84%", "92%", "90%", "9.5%"}
RESULT_VALUES = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", RESULTS.replace(",", ""))]


def test_readme_links_exist():
    links = re.findall(r"\]\(((?!https?://)[^)#]+)\)", README) + re.findall(r'src="((?!https?://)[^"]+)"', README)
    assert links
    missing = [link for link in links if not (ROOT / link).exists()]
    assert not missing, missing


def test_every_figure_in_readme_is_generated():
    for image in re.findall(r"!\[[^\]]*\]\(((?!https?://)[^)]+)\)", README):    # local figures, not badges
        assert (ROOT / image).stat().st_size > 10_000, image


def readme_numbers():
    # The README and the interview prep both quote results, so both are checked.
    return sorted(set(re.findall(r"(?<![\w.,])[−+-]?\d+\.\d+%?|(?<![\w.])\d+%", DOCS)))


def matches_at_precision(number: str, values: list[float]) -> bool:
    """True if some value rounds to the README number at the precision the README shows it."""
    text = number.replace("−", "-").lstrip("+")
    percent = text.endswith("%")
    text = text.rstrip("%")
    decimals = len(text.split(".")[1]) if "." in text else 0
    target = abs(float(text))
    for v in values:
        for scaled in ([v * 100, v] if percent else [v]):
            if round(abs(scaled), decimals) == pytest.approx(target, abs=1e-9):
                return True
    return False


@pytest.mark.parametrize("number", readme_numbers())
def test_readme_number_comes_from_results(number, shots):
    if number in NOT_RESULTS:
        return
    # Values the training script prints rather than writes to a results file.
    np_shots = shots[~shots.is_penalty]
    printed = [shots[shots.is_penalty].xg.iloc[0], np_shots.xg.sum() % 1000]
    assert matches_at_precision(number, RESULT_VALUES + printed), number


def test_readme_shot_totals(shots):
    np_shots = shots[~shots.is_penalty]
    assert f"{len(shots):,} shots" in README
    assert f"{len(np_shots):,} non-penalty shots" in README
    assert f"{np_shots.xg.sum():,.1f} goals against {np_shots.is_goal.sum():,} actual" in README


# --- results files are current ---

def test_leicester_results_current(shots, matches):
    t = team_breakdown(shots, matches, PL)
    lei = t.loc[LEICESTER]
    text = (FIGURES / "leicester_results.md").read_text()
    assert f"Finishing: {lei.goals:.0f} goals from {lei.xg:.1f} xG ({lei.finishing:+.1f})" in text
    assert f"Defense: {lei.goals_against:.0f} conceded from {lei.xga:.1f} xG against" in text
    assert f"({lei.goals_prevented:+.1f} prevented)" in text
    assert f"Quality: {lei.xg_per_shot:.3f} xG per shot" in text

    pl_shots = shots[shots.competition == PL]
    xpts = expected_points(pl_shots, matches[matches.competition == PL]).loc[LEICESTER]
    assert f"Points: {xpts.points:.0f} actual vs {xpts.xpts:.1f} expected ({xpts.points_minus_xpts:+.1f})" in text

    scored = pl_shots[(pl_shots.team == LEICESTER) & ~pl_shots.is_penalty]
    conceded = pl_shots[(pl_shots.opponent == LEICESTER) & ~pl_shots.is_penalty]
    p_scored = p_at_least(scored.is_goal.sum(), simulate_goals(scored.xg))
    p_conceded = p_at_most(conceded.is_goal.sum(), simulate_goals(conceded.xg))
    assert f"| Leicester goals scored | {len(scored)} | {scored.xg.sum():.1f} | {scored.is_goal.sum()} |" in text
    assert f"| {p_scored:.3f} |" in text and f"| {p_conceded:.3f} |" in text


def test_champions_results_current(shots, matches):
    summary = champion_summary(shots, matches, n_sims=10_000)
    text = (FIGURES / "champions_results.md").read_text()
    for name, r in [("Leicester", summary.loc[LEICESTER]), ("Barcelona", summary.loc["Barcelona"])]:
        assert f"- {name}: {r.shots_vs_league:+.0%} shots, {r.quality_vs_league:+.0%} xG per shot, " \
               f"{r.finishing:+.1f} goals vs xG" in text


def test_formations_results_current(shots, formation_minutes):
    lei = formation_profile(shots, formation_minutes, team=LEICESTER).loc["4-4-2"]
    other = formation_profile(shots, formation_minutes, exclude_team=LEICESTER).loc["4-4-2"]
    text = (FIGURES / "formations_results.md").read_text()
    assert f"Leicester in 4-4-2: {lei.xg_per90:.2f} xG per 90 vs {other.xg_per90:.2f}" in text


def test_readme_key_claims_exact(shots, matches):
    """The headline sentences and simulation table, rebuilt from data, word for word."""
    t = team_breakdown(shots, matches, PL)
    lei = t.loc[LEICESTER]
    pl_shots = shots[shots.competition == PL]
    xpts = expected_points(pl_shots, matches[matches.competition == PL]).loc[LEICESTER]
    scored = pl_shots[(pl_shots.team == LEICESTER) & ~pl_shots.is_penalty]
    conceded = pl_shots[(pl_shots.opponent == LEICESTER) & ~pl_shots.is_penalty]
    rows = {
        "Leicester goals scored": (scored, p_at_least),
        "Leicester goals conceded": (conceded, p_at_most),
        "Jamie Vardy": (scored[scored.player.str.contains("Vardy")], p_at_least),
        "Riyad Mahrez": (scored[scored.player.str.contains("Mahrez")], p_at_least),
    }
    p = {}
    for label, (d, p_fn) in rows.items():
        sims = simulate_goals(d.xg)
        p[label] = p_fn(d.is_goal.sum(), sims)
        low, high = np.percentile(sims, [5, 95]).astype(int)
        assert f"| {label} | {len(d)} | {d.xg.sum():.1f} | {d.is_goal.sum()} | {low} to {high} | " \
               f"{p[label]:.3f} |" in README, label

    claims = [
        f"Leicester averaged {lei.xg_per_shot:.3f} xG per shot",
        f"They scored {lei.goals:.0f} non-penalty goals from {lei.xg:.1f} xG. In 10,000 simulated seasons, "
        f"{p['Leicester goals scored']:.0%} matched or beat that total.",
        f"Leicester conceded {lei.goals_against:.0f} non-penalty goals from {lei.xga:.1f} xG against, "
        f"{lei.goals_prevented:.1f} fewer than expected and the most in the league "
        f"(p = {p['Leicester goals conceded']:.3f}).",
        f"(p = {p['Riyad Mahrez']:.3f})",
        f"Leicester won {xpts.points:.0f} points from {xpts.xpts:.1f} expected points.",
    ]
    for claim in claims:
        assert claim in README, claim


def test_readme_defense_claims_exact(shots, matches):
    funnel = defense_funnel(shots, load_funnel(), PL).loc[LEICESTER]
    text = (FIGURES / "leicester_results.md").read_text()
    assert (f"Kasper Schmeichel's goalkeeping ({funnel.goalkeeping:+.1f}), opponents missing the target "
            f"({funnel.missed_target:+.1f}) and blocks ({funnel.blocks:+.1f})") in README
    assert f"| Goalkeeping (Schmeichel started all 38 matches) | {funnel.goalkeeping:+.1f} |" in README
    assert f"Leicester blocked {funnel.blocked:.0f} of {funnel.shots_against:.0f} shots, " \
           f"{funnel.blocked - funnel.expected_blocked:.0f} more than expected" in README
    keeper_p = re.search(r"p = (0\.\d+)\n", text.split("- Goalkeeping: ")[-1]).group(1)
    assert f"Schmeichel's share alone has p = {keeper_p}" in README
    any_team = re.search(r"by luck is (\d+%)", text).group(1)
    assert f"by luck is {any_team}" in README
