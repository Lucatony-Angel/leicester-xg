"""Part 3: formations across the 5 leagues, compared at the same game state.

Formation choice is tied to team quality (strong teams pick some shapes more),
so these are patterns, not causes. Run after build_db.py and train.py.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from analysis import FIGURES_DIR, GRAY, INK, LEICESTER_BLUE, LIGHT_GRAY, MUTED, style
from team_stats import (LEICESTER, PL, formation_profile, load, load_formation_minutes, ordinal,
                        team_formation_profiles)

MIN_MINUTES = 5000  # about 55 full matches; smaller formations are too noisy to compare
PANELS = [
    ("xg_per90", "xG created per 90", "{:.2f}"),
    ("xga_per90", "xG conceded per 90", "{:.2f}"),
    ("xg_per_shot", "xG per shot", "{:.3f}"),
    ("counter_share", "Share of xG from counterattacks", "{:.1%}"),
]


def plot_formations(profile: pd.DataFrame, leicester: pd.Series) -> None:
    order = profile.sort_values("xg_per90").index
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.8), sharey=True)
    for ax, (col, label, fmt) in zip(axes, PANELS):
        ax.barh(order, profile.loc[order, col], color=LIGHT_GRAY, height=0.7)
        ax.scatter([leicester[col]], ["4-4-2"], color=LEICESTER_BLUE, s=70, zorder=3, edgecolor="white", linewidth=1.5)
        for y, f in enumerate(order):
            ax.text(profile.loc[f, col], y, fmt.format(profile.loc[f, col]) + " ", va="center", ha="right",
                    fontsize=8, color=INK)
        style(ax)
        ax.tick_params(axis="y", length=0)
        ax.set_title(label, loc="left", fontsize=10, color=INK)
        if fmt.endswith("%}"):
            ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlim(0, max(profile[col].max(), leicester[col]) * 1.2)
    axes[0].scatter([], [], color=LEICESTER_BLUE, s=60, label="Leicester in 4-4-2")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.suptitle("At the same scorelines, common formations differ by about 0.3 xG per 90. "
                 "Leicester's 4-4-2 out-created every formation average",
                 x=0.03, ha="left", fontsize=11.5, color=INK)
    fig.text(0.03, -0.03, f"Non-penalty shots, 5 leagues, 2015/16. Formations with at least {MIN_MINUTES:,} minutes. "
             "Rates are weighted to the same game state mix (time spent level, ahead and behind). "
             "Strong teams favor some formations, so these are patterns, not causes.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES_DIR / "formations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_442_teams(teams: pd.DataFrame) -> None:
    t = teams.reset_index()
    is_lei = t.team == LEICESTER
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(t[~is_lei].shots_per90, t[~is_lei].xg_per_shot, s=t[~is_lei].minutes / 25, color=GRAY,
               edgecolor="white", linewidth=1.2, alpha=0.9)
    lei = t[is_lei].iloc[0]
    ax.scatter(lei.shots_per90, lei.xg_per_shot, s=lei.minutes / 25, color=LEICESTER_BLUE, edgecolor="white",
               linewidth=1.5, zorder=3)
    ax.annotate("Leicester", (lei.shots_per90, lei.xg_per_shot), xytext=(10, 4), textcoords="offset points",
                color=INK, fontweight="bold")
    for _, r in t[~is_lei].nlargest(3, "xg_per90").iterrows():
        ax.annotate(r.team, (r.shots_per90, r.xg_per_shot), xytext=(7, -10), textcoords="offset points",
                    color=MUTED, fontsize=8.5)
    style(ax)
    ax.set_xlabel("Shots per 90 in 4-4-2", color=INK)
    ax.set_ylabel("xG per shot in 4-4-2", color=INK)
    rank = int(teams.xg_per90.rank(ascending=False)[(PL, LEICESTER)])
    most = "the most" if rank == 1 else f"the {ordinal(rank)}-most"
    ax.set_title(f"Of {len(t)} teams that played 4-4-2, Leicester created {most} xG per 90",
                 loc="left", fontsize=11, color=INK)
    fig.text(0.125, -0.02, "Teams with at least 900 minutes in 4-4-2, game-state adjusted. Dot size = minutes in 4-4-2.",
             fontsize=8, color=MUTED)
    fig.savefig(FIGURES_DIR / "formation_442_teams.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    shots, _ = load()
    minutes = load_formation_minutes()
    raw = formation_profile(shots, minutes, adjust=False)
    adjusted = formation_profile(shots, minutes)
    common = adjusted[adjusted.minutes >= MIN_MINUTES]
    leicester = formation_profile(shots, minutes, team=LEICESTER).loc["4-4-2"]
    other_442 = formation_profile(shots, minutes, exclude_team=LEICESTER).loc["4-4-2"]
    teams = team_formation_profiles(shots, minutes, "4-4-2")
    lei_key = (PL, LEICESTER)

    plot_formations(common, leicester)
    plot_442_teams(teams)

    def lei_rank(col):
        return f"{ordinal(int(teams[col].rank(ascending=False)[lei_key]))} of {len(teams)}"

    cols = ["minutes", "shots_per90", "xg_per90", "xg_per_shot", "shots_against_per90", "xga_per90",
            "counter_share", "set_piece_share", "cross_share", "through_ball_share"]
    spread = common.xg_per90.max() - common.xg_per90.min()
    shift = (common.xg_per90 - raw.loc[common.index, "xg_per90"]).abs().max()
    lines = [
        "# Part 3 results: formations",
        "",
        "Generated by `src/formations.py`. Non-penalty shots, 5 leagues, 2015/16. Formation and score are tracked",
        "minute by minute (Starting XI, Tactical Shifts, goals), so every rate is per 90 minutes actually played in",
        "that formation. Formation choice is tied to team quality: these are patterns, not causes.",
        "",
        "## Key numbers",
        "",
        f"- Across formations with at least {MIN_MINUTES:,} minutes, game-state-adjusted xG per 90 ranges from "
        f"{common.xg_per90.min():.2f} ({common.xg_per90.idxmin()}) to {common.xg_per90.max():.2f} "
        f"({common.xg_per90.idxmax()}), a spread of {spread:.2f}.",
        f"- Adjusting for game state moves any formation's xG per 90 by at most {shift:.2f}.",
        f"- Leicester in 4-4-2: {leicester.xg_per90:.2f} xG per 90 vs {other_442.xg_per90:.2f} for all other 4-4-2 "
        f"minutes ({lei_rank('xg_per90')} 4-4-2 teams).",
        f"- xG per shot: {leicester.xg_per_shot:.3f} vs {other_442.xg_per_shot:.3f} ({lei_rank('xg_per_shot')}).",
        f"- Shots per 90: {leicester.shots_per90:.1f} vs {other_442.shots_per90:.1f} ({lei_rank('shots_per90')}).",
        f"- Counterattack share of xG: {leicester.counter_share:.1%} vs {other_442.counter_share:.1%} "
        f"({lei_rank('counter_share')}).",
        f"- Cross share of xG: {leicester.cross_share:.1%} vs {other_442.cross_share:.1%}.",
        f"- xG conceded per 90: {leicester.xga_per90:.2f} vs {other_442.xga_per90:.2f}.",
        "",
        "`set_piece_share` is xG from possessions that started with a corner, free kick or throw-in, as StatsBomb",
        "tags the whole possession, not only the set piece itself.",
        "",
        "## Formations (game-state adjusted)",
        "",
        common[cols].round(3).to_markdown(),
        "",
        "## Formations (raw, not adjusted)",
        "",
        raw.loc[common.index, cols].round(3).to_markdown(),
        "",
        "## Leicester vs other 4-4-2 teams",
        "",
        pd.DataFrame({"Leicester in 4-4-2": leicester, "Other 4-4-2 minutes": other_442}).loc[cols].round(3)
        .to_markdown(),
        "",
        "## Every team with 900+ minutes in 4-4-2",
        "",
        teams[cols].sort_values("xg_per90", ascending=False).round(3).to_markdown(),
        "",
    ]
    (FIGURES_DIR / "formations_results.md").write_text("\n".join(lines))
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
