"""Shared data loading and chart helpers for the app pages.

The app only reads data/processed/ and figures/. No downloading or training here.
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulate import expected_points, p_at_least, p_at_most, simulate_goals  # noqa: E402
from team_stats import (  # noqa: E402
    CHAMPIONS,
    GAME_STATES,
    LEICESTER,
    PL,
    defense_funnel,
    formation_profile,
    load,
    load_formation_minutes,
    load_funnel,
    ordinal,
    rank,
    share_vs_league,
    team_breakdown,
    team_formation_profiles,
)

FIGURES_DIR = ROOT / "figures"
LEICESTER_BLUE = "#003090"
GRAY, LIGHT_GRAY, INK = "#9a9a9a", "#d4d4d4", "#2b2b2b"
LEAGUES = {
    "Premier League": "England - Premier League",
    "La Liga": "Spain - La Liga",
    "Serie A": "Italy - Serie A",
    "Ligue 1": "France - Ligue 1",
    "Bundesliga (Leverkusen only)": "Germany - 1. Bundesliga",
}


@st.cache_data
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load()


@st.cache_data
def get_breakdown(competition: str) -> pd.DataFrame:
    shots, matches = get_data()
    return team_breakdown(shots, matches, competition)


@st.cache_data
def get_expected_points(competition: str) -> pd.DataFrame:
    shots, matches = get_data()
    return expected_points(shots[shots.competition == competition], matches[matches.competition == competition])


@st.cache_data
def get_formation_minutes() -> pd.DataFrame:
    return load_formation_minutes()


@st.cache_data
def get_formation_profile(competition: str | None, game_state: str | None) -> pd.DataFrame:
    shots, _ = get_data()
    return formation_profile(shots, get_formation_minutes(), competition=competition, game_state=game_state)


@st.cache_data
def get_team_formation_profiles(formation: str) -> pd.DataFrame:
    shots, _ = get_data()
    return team_formation_profiles(shots, get_formation_minutes(), formation)


@st.cache_data
def get_champions() -> pd.DataFrame:
    # Precomputed by src/champions.py (40,000 simulated seasons): too slow to redo on page load.
    return pd.read_parquet(ROOT / "data" / "processed" / "champions.parquet")


@st.cache_data
def get_defense_funnel(competition: str) -> pd.DataFrame:
    shots, _ = get_data()
    return defense_funnel(shots, load_funnel(), competition)


@st.cache_data
def get_players() -> pd.DataFrame:
    """One row per player: non-penalty shots, goals, xG and the team(s) they shot for."""
    shots, _ = get_data()
    s = shots[~shots.is_penalty]
    players = s.groupby("player_id").agg(
        player=("player", "first"), teams=("team", lambda t: ", ".join(dict.fromkeys(t))),
        competition=("competition", "first"), shots=("shot_id", "size"), goals=("is_goal", "sum"), xg=("xg", "sum"))
    players["label"] = players.player + " (" + players.teams + ", " + players.shots.astype(str) + " shots)"
    return players.sort_values("xg", ascending=False)


@st.cache_data
def get_simulation(xg: tuple) -> pd.Series:
    return pd.Series(simulate_goals(list(xg)))


def short_name(team: str) -> str:
    return team.replace(" Hotspur", "").replace(" Albion", "").replace("Paris Saint-Germain", "PSG")


def highlight(team_field: str, team: str) -> alt.Color:
    """Leicester blue for the highlighted team, gray for everyone else."""
    return alt.condition(alt.datum[team_field] == team, alt.value(LEICESTER_BLUE), alt.value(LIGHT_GRAY))


def simulation_chart(sims: pd.Series, actual: int) -> alt.LayerChart:
    counts = sims.value_counts(normalize=True).rename_axis("goals").reset_index(name="share")
    bars = alt.Chart(counts).mark_bar(color=LIGHT_GRAY, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("goals:Q", title="Goals in a simulated season", scale=alt.Scale(zero=False, nice=False)),
        y=alt.Y("share:Q", title=None, axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("goals:Q", title="Goals"), alt.Tooltip("share:Q", title="Share of seasons", format=".1%")],
    )
    rule = alt.Chart(pd.DataFrame({"goals": [actual], "label": [f"Actual: {actual}"]})).mark_rule(
        color=LEICESTER_BLUE, strokeWidth=3).encode(x="goals:Q", tooltip=alt.Tooltip("label:N", title=None))
    return (bars + rule).properties(height=220)


def credit() -> None:
    st.caption("Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data). "
               "Data provided by StatsBomb.")
