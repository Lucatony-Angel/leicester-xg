"""Part 1: how did Leicester win the 2015/16 Premier League?

Splits every PL team's goals into volume, quality and finishing, simulates
Leicester's finishing, and saves the charts and numbers used in the README.
Run after train.py.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from mplsoccer import VerticalPitch

from simulate import expected_points, p_at_least, p_at_most, simulate_goals
from train import PROCESSED_DIR, ROOT

FIGURES_DIR = ROOT / "figures"
PL = "England - Premier League"
LEICESTER = "Leicester City"
LEICESTER_BLUE = "#003090"
GRAY, LIGHT_GRAY, INK, MUTED = "#9a9a9a", "#d4d4d4", "#2b2b2b", "#8a8a8a"


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


def rank(t: pd.DataFrame, col: str, ascending: bool = False) -> int:
    return int(t[col].rank(ascending=ascending, method="min")[LEICESTER])


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def share_vs_league(shots: pd.DataFrame, column: str) -> pd.DataFrame:
    """Share of Leicester's non-penalty xG from each category, vs the rest of the PL."""
    s = shots[(shots.competition == PL) & ~shots.is_penalty]
    is_lei = s.team == LEICESTER
    out = pd.DataFrame({
        "leicester_xg": s[is_lei].groupby(column).xg.sum(),
        "leicester_share": s[is_lei].groupby(column).xg.sum() / s[is_lei].xg.sum(),
        "rest_of_pl_share": s[~is_lei].groupby(column).xg.sum() / s[~is_lei].xg.sum(),
    }).fillna(0)
    out["ratio"] = out.leicester_share / out.rest_of_pl_share
    return out.sort_values("leicester_xg", ascending=False)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK)


def team_colors(teams) -> list[str]:
    return [LEICESTER_BLUE if t == LEICESTER else LIGHT_GRAY for t in teams]


def plot_team_scatter(t: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    others = t.drop(LEICESTER)
    ax.scatter(others.shots_per_match, others.xg_per_shot, s=70, color=GRAY, edgecolor="white", linewidth=1.5)
    lei = t.loc[LEICESTER]
    ax.scatter(lei.shots_per_match, lei.xg_per_shot, s=130, color=LEICESTER_BLUE, edgecolor="white",
               linewidth=2, zorder=3)
    ax.annotate("Leicester", (lei.shots_per_match, lei.xg_per_shot), xytext=(8, 6),
                textcoords="offset points", color=INK, fontweight="bold")
    for team in ["Arsenal", "Tottenham Hotspur", "Manchester City", "Aston Villa"]:
        ax.annotate(team.replace(" Hotspur", ""), (t.loc[team, "shots_per_match"], t.loc[team, "xg_per_shot"]),
                    xytext=(6, -12), textcoords="offset points", color=MUTED, fontsize=9)
    ax.axvline(t.shots_per_match.mean(), color=LIGHT_GRAY, linestyle="--", linewidth=1, zorder=0)
    ax.axhline(t.xg_per_shot.mean(), color=LIGHT_GRAY, linestyle="--", linewidth=1, zorder=0)
    style(ax)
    ax.set_xlabel("Shots per match (volume)", color=INK)
    ax.set_ylabel("xG per shot (quality)", color=INK)
    ax.set_title(f"Leicester took mid-table shot volume ({ordinal(rank(t, 'shots_per_match'))}) but created "
                 f"the {ordinal(rank(t, 'xg_per_shot'))}-best chances in the league",
                 loc="left", fontsize=11, color=INK)
    fig.text(0.125, 0.005, "Non-penalty shots, 2015/16 Premier League. Dashed lines = league average.",
             fontsize=8, color=MUTED)
    fig.savefig(FIGURES_DIR / "team_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_goals_vs_xg(t: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=False)
    panels = [("finishing", "Attack: goals minus xG"), ("goals_prevented", "Defense: xG against minus goals conceded")]
    for ax, (col, label) in zip(axes, panels):
        d = t[col].sort_values()
        ax.barh([n.replace(" Hotspur", "").replace(" Albion", "") for n in d.index], d.values,
                color=team_colors(d.index), height=0.75)
        ax.axvline(0, color=MUTED, linewidth=1)
        style(ax)
        ax.tick_params(axis="y", length=0, labelsize=8.5)
        ax.set_title(label, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Goals", color=INK)
    lei = t.loc[LEICESTER]
    fig.suptitle(f"Leicester's attack finished about as expected ({lei.finishing:+.1f}); their defense conceded "
                 f"{lei.goals_prevented:.1f} fewer than expected, the most in the league",
                 x=0.06, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "goals_vs_xg.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_simulations(sims: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (label, r) in zip(axes.flat, sims.items()):
        values, counts = np.unique(r["sims"], return_counts=True)
        ax.bar(values, counts / counts.sum(), width=0.85, color=LIGHT_GRAY)
        ax.axvline(r["actual"], color=LEICESTER_BLUE, linewidth=2.5)
        ax.text(r["actual"], ax.get_ylim()[1] * 0.92, f"  actual: {r['actual']}", color=LEICESTER_BLUE,
                fontweight="bold", fontsize=9)
        style(ax)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{label}: {r['shots']} shots, {r['xg']:.1f} xG, p = {r['p']:.3f}", loc="left",
                     fontsize=10, color=INK)
    fig.suptitle("10,000 simulated seasons: Vardy finished on par, Mahrez and the defense beat the odds",
                 x=0.06, ha="left", fontsize=12, color=INK)
    fig.text(0.06, -0.01, "Grey = goals in each simulated season, flipping a coin weighted by each shot's xG. "
             "p = share of simulations at least as extreme as the real total.", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "simulation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_expected_points(xpts: pd.DataFrame) -> None:
    d = xpts.sort_values("points")
    names = [n.replace(" Hotspur", "").replace(" Albion", "") for n in d.index]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.hlines(names, d.xpts, d.points, color=LIGHT_GRAY, linewidth=2)
    ax.scatter(d.xpts, names, s=45, facecolor="white", edgecolor=GRAY, linewidth=1.5, label="Expected points", zorder=3)
    ax.scatter(d.points, names, s=55, color=team_colors(d.index), edgecolor=[
        LEICESTER_BLUE if t == LEICESTER else GRAY for t in d.index], label="Actual points", zorder=3)
    style(ax)
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.set_xlabel("Points", color=INK)
    ax.legend(frameon=False, loc="lower right")
    lei = xpts.loc[LEICESTER]
    xrank = int(xpts.xpts.rank(ascending=False)[LEICESTER])
    biggest = xpts.points_minus_xpts.idxmax() == LEICESTER
    ax.set_title(f"Leicester won {lei.points:.0f} points from {lei.xpts:.1f} expected ({lei.points_minus_xpts:+.1f}"
                 f"{', the biggest gap in the league' if biggest else ''})\n"
                 f"Their chances alone rank {ordinal(xrank)} in the league",
                 loc="left", fontsize=11, color=INK)
    fig.savefig(FIGURES_DIR / "expected_points.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shot_map(shots: pd.DataFrame) -> None:
    lei = shots[(shots.team == LEICESTER) & ~shots.is_penalty]
    goals, misses = lei[lei.is_goal == 1], lei[lei.is_goal == 0]
    pitch = VerticalPitch(half=True, pitch_type="statsbomb", line_color="#c8c8c8")
    fig, ax = pitch.draw(figsize=(8, 7))
    pitch.scatter(misses.x, misses.y, s=misses.xg * 600 + 15, ax=ax, facecolors="none", edgecolors=GRAY,
                  linewidth=1, label="Miss")
    pitch.scatter(goals.x, goals.y, s=goals.xg * 600 + 15, ax=ax, color=LEICESTER_BLUE, edgecolors="white",
                  linewidth=1, label="Goal")
    ax.legend(loc="lower left", frameon=False)
    ax.set_title(f"Leicester scored {lei.is_goal.sum()} non-penalty goals from {lei.xg.sum():.1f} xG "
                 f"({len(lei)} shots, dot size = xG)", fontsize=12, color=INK)
    fig.savefig(FIGURES_DIR / "leicester_shot_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    shots, matches = load()
    t = team_breakdown(shots, matches, PL)
    pl_shots = shots[shots.competition == PL]
    xpts = expected_points(pl_shots, matches[matches.competition == PL])

    lei_np = pl_shots[(pl_shots.team == LEICESTER) & ~pl_shots.is_penalty]
    against = pl_shots[(pl_shots.opponent == LEICESTER) & ~pl_shots.is_penalty]
    groups = {
        "Leicester goals scored": (lei_np, p_at_least),
        "Leicester goals conceded": (against, p_at_most),
        "Jamie Vardy": (lei_np[lei_np.player.str.contains("Vardy")], p_at_least),
        "Riyad Mahrez": (lei_np[lei_np.player.str.contains("Mahrez")], p_at_least),
    }
    sims = {}
    for label, (d, p_fn) in groups.items():
        s = simulate_goals(d.xg)
        actual = int(d.is_goal.sum())
        sims[label] = {"sims": s, "actual": actual, "shots": len(d), "xg": d.xg.sum(), "p": p_fn(actual, s),
                       "range_90": tuple(np.percentile(s, [5, 95]).astype(int))}

    patterns = share_vs_league(shots, "play_pattern")
    assists = share_vs_league(shots, "assist_type")

    plot_team_scatter(t)
    plot_goals_vs_xg(t)
    plot_simulations(sims)
    plot_expected_points(xpts)
    plot_shot_map(shots)

    lei, lx = t.loc[LEICESTER], xpts.loc[LEICESTER]
    cols = ["shots_per_match", "xg_per_shot", "xg", "goals", "finishing", "shots_against_per_match",
            "xga_per_shot", "xga", "goals_against", "goals_prevented", "xg_diff", "penalties", "penalty_goals"]
    lines = [
        "# Part 1 results: Leicester 2015/16",
        "",
        "Generated by `src/analysis.py`. Non-penalty shots unless stated.",
        "",
        "## Key numbers",
        "",
        f"- Points: {lx.points:.0f} actual vs {lx.xpts:.1f} expected ({lx.points_minus_xpts:+.1f}), "
        f"rank {int(xpts.xpts.rank(ascending=False)[LEICESTER])} on expected points",
        f"- Volume: {lei.shots_per_match:.1f} shots per match, rank {rank(t, 'shots_per_match')} of 20",
        f"- Quality: {lei.xg_per_shot:.3f} xG per shot, rank {rank(t, 'xg_per_shot')} of 20",
        f"- Finishing: {lei.goals:.0f} goals from {lei.xg:.1f} xG ({lei.finishing:+.1f}), "
        f"rank {rank(t, 'finishing')} of 20",
        f"- Defense: {lei.goals_against:.0f} conceded from {lei.xga:.1f} xG against ({lei.goals_prevented:+.1f} prevented), "
        f"rank {rank(t, 'goals_prevented')} of 20; {lei.xga_per_shot:.3f} xG per shot against, "
        f"rank {rank(t, 'xga_per_shot', ascending=True)} of 20 (lowest = 1)",
        f"- Penalties: {lei.penalties:.0f} taken ({lei.penalty_goals:.0f} scored), rank {rank(t, 'penalties')} of 20",
        f"- xG difference: {lei.xg_diff:+.1f}, rank {rank(t, 'xg_diff')} of 20",
        f"- Counterattacks: {patterns.loc['From Counter', 'leicester_share']:.1%} of Leicester's xG vs "
        f"{patterns.loc['From Counter', 'rest_of_pl_share']:.1%} for the rest of the PL "
        f"({patterns.loc['From Counter', 'ratio']:.1f}x)",
        "",
        "## Simulations (10,000 seasons)",
        "",
        "| Group | Shots | xG | Actual | 90% range | p |",
        "|---|---|---|---|---|---|",
        *[f"| {k} | {v['shots']} | {v['xg']:.1f} | {v['actual']} | {v['range_90'][0]} to {v['range_90'][1]} | "
          f"{v['p']:.3f} |" for k, v in sims.items()],
        "",
        "## All 20 teams",
        "",
        t[cols].sort_values("xg_diff", ascending=False).round(3).to_markdown(),
        "",
        "## Expected points",
        "",
        xpts.round(1).to_markdown(),
        "",
        "## Leicester xG by play pattern",
        "",
        patterns.round(3).to_markdown(),
        "",
        "## Leicester xG by assist type",
        "",
        assists.round(3).to_markdown(),
        "",
    ]
    (FIGURES_DIR / "leicester_results.md").write_text("\n".join(lines))
    print("\n".join(lines[:22]))


if __name__ == "__main__":
    main()
