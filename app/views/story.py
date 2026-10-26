import altair as alt
import streamlit as st

from common import (
    GRAY,
    LEICESTER,
    LEICESTER_BLUE,
    LIGHT_GRAY,
    PL,
    credit,
    get_breakdown,
    get_data,
    get_defense_funnel,
    get_expected_points,
    get_simulation,
    highlight,
    ordinal,
    p_at_least,
    p_at_most,
    rank,
    short_name,
    simulation_chart,
)

shots, _ = get_data()
t = get_breakdown(PL)
xpts = get_expected_points(PL)
lei, lx = t.loc[LEICESTER], xpts.loc[LEICESTER]
xpts_rank = int(xpts.xpts.rank(ascending=False, method="min")[LEICESTER])

st.title("Leicester won the 2015/16 Premier League with chance quality and a defense that beat its xG, "
         "not lucky finishing")
st.write(f"They created the {ordinal(rank(t, 'xg_per_shot'))}-best chances in the league on mid-table shot volume, "
         f"finished them at an average rate, and conceded {lei.goals_prevented:.1f} fewer goals than their "
         f"opponents' chances were worth.")

c1, c2, c3 = st.columns(3)
c1.metric("Points vs expected points", f"{lx.points:.0f} vs {lx.xpts:.1f}",
          f"{lx.points_minus_xpts:+.1f}, biggest gap in the league", border=True)
c2.metric("xG per shot (chance quality)", f"{lei.xg_per_shot:.3f}",
          f"{ordinal(rank(t, 'xg_per_shot'))} of 20 (volume: {ordinal(rank(t, 'shots_per_match'))})",
          delta_color="off", delta_arrow="off", border=True)
c3.metric("Goals conceded below xG", f"{lei.goals_prevented:.1f}",
          f"{ordinal(rank(t, 'goals_prevented'))} of 20 ({lei.goals_against:.0f} conceded from {lei.xga:.1f} xG)",
          delta_color="off", delta_arrow="off", border=True)

st.subheader(f"Mid-table volume ({ordinal(rank(t, 'shots_per_match'))}), "
             f"{ordinal(rank(t, 'xg_per_shot'))}-best chance quality")
teams = t.reset_index(names="team").assign(name=lambda d: d.team.map(short_name))
base = alt.Chart(teams).encode(
    x=alt.X("shots_per_match:Q", title="Shots per match (volume)", scale=alt.Scale(zero=False, padding=16)),
    y=alt.Y("xg_per_shot:Q", title="xG per shot (quality)", scale=alt.Scale(zero=False, padding=16)),
)
points = base.mark_circle(stroke="white", strokeWidth=1.5, opacity=1).encode(
    color=alt.condition(alt.datum.team == LEICESTER, alt.value(LEICESTER_BLUE), alt.value(GRAY)),
    size=alt.condition(alt.datum.team == LEICESTER, alt.value(220), alt.value(110)),
    tooltip=[alt.Tooltip("team:N", title="Team"),
             alt.Tooltip("shots_per_match:Q", title="Shots per match", format=".1f"),
             alt.Tooltip("xg_per_shot:Q", title="xG per shot", format=".3f"),
             alt.Tooltip("goals:Q", title="Goals (non-penalty)"),
             alt.Tooltip("xg:Q", title="xG", format=".1f")],
)
labels = base.transform_filter(alt.FieldOneOfPredicate("team", [LEICESTER, "Arsenal", "Tottenham Hotspur",
                                                                 "Aston Villa"])).mark_text(
    align="left", dx=10, dy=-8, color="#2b2b2b").encode(text="name:N")
st.altair_chart((points + labels).properties(height=420), width="stretch")
st.caption("Non-penalty shots. Hover a dot for details.")

st.subheader("Was it luck? 10,000 simulated seasons")
st.write("Each shot is replayed as a coin flip weighted by its xG. The blue line is what really happened.")
lei_np = shots[(shots.competition == PL) & (shots.team == LEICESTER) & ~shots.is_penalty]
against = shots[(shots.competition == PL) & (shots.opponent == LEICESTER) & ~shots.is_penalty]
groups = [
    ("Leicester goals scored", lei_np, p_at_least),
    ("Leicester goals conceded", against, p_at_most),
    ("Jamie Vardy", lei_np[lei_np.player.str.contains("Vardy")], p_at_least),
    ("Riyad Mahrez", lei_np[lei_np.player.str.contains("Mahrez")], p_at_least),
]
cols = st.columns(2)
p_values = {}
for i, (label, d, p_fn) in enumerate(groups):
    sims = get_simulation(tuple(d.xg))
    actual = int(d.is_goal.sum())
    p = p_values[label] = p_fn(actual, sims.values)
    with cols[i % 2]:
        st.markdown(f"**{label}**: {actual} actual from {d.xg.sum():.1f} xG (p = {p:.3f})")
        st.altair_chart(simulation_chart(sims, actual), width="stretch")
any_team = 1 - (1 - p_values["Leicester goals conceded"]) ** len(t)
st.caption(f"p = share of simulated seasons at least as extreme as the real one. With {len(t)} teams, the chance "
           f"that at least one defense gets this far ahead of its xG by luck alone is {any_team:.0%}.")

funnel = get_defense_funnel(PL)
lf = funnel.loc[LEICESTER]
stages = [("blocks", "Blocks"), ("missed_target", "Opponents missing the target"), ("goalkeeping", "Goalkeeping")]
st.subheader(f"Where the {lf.goals_prevented:.1f} goals were prevented: spread across all three stages")
st.write("Every shot against has to avoid a block, hit the target and beat the keeper. Each stage compares what "
         "happened with what shots of the same quality do on average across the 5 leagues.")
cols = st.columns(3)
shared = alt.Scale(domain=[min(funnel[f].min() for f, _ in stages), max(funnel[f].max() for f, _ in stages)])
for col, (field, label) in zip(cols, stages, strict=True):
    col.metric(label, f"{lf[field]:+.1f}", f"{ordinal(rank(funnel, field))} of 20", delta_color="off",
               delta_arrow="off", border=True)
    data = funnel.reset_index().assign(name=lambda d: d.team.map(short_name))
    chart = alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("name:N", sort=alt.EncodingSortField(field, order="descending"), title=None,
                axis=alt.Axis(labelOverlap=False)),
        x=alt.X(f"{field}:Q", title="Goals prevented", scale=shared),
        color=highlight("team", LEICESTER),
        tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip(f"{field}:Q", title=label, format="+.1f")],
    )
    col.altair_chart(chart.properties(height=420), width="stretch")
st.caption("The three stages add up to xG against minus goals conceded. PL shots were blocked a little more often "
           "than the 5-league average, so most PL teams are positive on blocks.")

st.subheader(f"{lx.points:.0f} points from {lx.xpts:.1f} expected: their chances alone rank {ordinal(xpts_rank)}")
table = xpts.reset_index(names="team").assign(name=lambda d: d.team.map(short_name))
order = table.sort_values(["points", "team"], ascending=[False, True]).name.tolist()
rule = alt.Chart(table).mark_rule(color=LIGHT_GRAY, strokeWidth=2).encode(
    y=alt.Y("name:N", sort=order, title=None),
    x=alt.X("xpts:Q", title="Points", scale=alt.Scale(zero=False)),
    x2="points:Q")
expected = alt.Chart(table).mark_point(filled=True, fill="white", stroke=GRAY, size=70, strokeWidth=1.5).encode(
    y=alt.Y("name:N", sort=order), x="xpts:Q",
    tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip("xpts:Q", title="Expected points", format=".1f")])
actual = alt.Chart(table).mark_circle(size=90, opacity=1).encode(
    y=alt.Y("name:N", sort=order), x="points:Q", color=highlight("team", LEICESTER),
    tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip("points:Q", title="Actual points"),
             alt.Tooltip("points_minus_xpts:Q", title="Points minus expected", format="+.1f")])
st.altair_chart((rule + expected + actual).properties(height=520), width="stretch")
st.caption("Hollow = expected points from simulating every match's shots. Filled = actual points.")

credit()
