"""Data integrity checks on the DuckDB build. Skipped when the database isn't built."""

import math

import pytest

EXPECTED_MATCHES = {
    "England - Premier League": 380,
    "Spain - La Liga": 380,
    "Italy - Serie A": 380,
    "France - Ligue 1": 377,          # 3 matches missing from the free data
    "Germany - 1. Bundesliga": 34,    # Leverkusen's matches only
}


def one(db, sql):
    return db.execute(sql).fetchone()[0]


def test_match_counts(db):
    counts = dict(db.execute("SELECT competition, count(*) FROM matches GROUP BY 1").fetchall())
    assert counts == EXPECTED_MATCHES


def test_every_match_has_events(db):
    assert one(db, "SELECT count(*) FROM matches WHERE match_id NOT IN (SELECT DISTINCT match_id FROM events)") == 0


def test_goals_rebuild_every_final_score(db):
    # The strongest check on game state: goals counted from events, including own goals,
    # must give the official final score in every match.
    wrong = one(db, """
        WITH g AS (SELECT match_id, scoring_team, count(*) n FROM goals GROUP BY ALL)
        SELECT count(*) FROM matches m
        LEFT JOIN g h ON h.match_id = m.match_id AND h.scoring_team = m.home_team
        LEFT JOIN g a ON a.match_id = m.match_id AND a.scoring_team = m.away_team
        WHERE coalesce(h.n, 0) != m.home_score OR coalesce(a.n, 0) != m.away_score
    """)
    assert wrong == 0


def test_one_name_per_team(db):
    assert one(db, """SELECT count(*) FROM (SELECT team_id FROM events WHERE team_id IS NOT NULL
                      GROUP BY 1 HAVING count(DISTINCT team) > 1)""") == 0


def test_every_shot_team_plays_in_its_match(db):
    assert one(db, """SELECT count(*) FROM shot_table s JOIN matches m USING (match_id)
                      WHERE s.team NOT IN (m.home_team, m.away_team)
                         OR s.opponent NOT IN (m.home_team, m.away_team) OR s.team = s.opponent""") == 0


def test_shot_table_one_row_per_shot(db):
    shots = one(db, "SELECT count(*) FROM events WHERE type = 'Shot' AND period <= 4")
    rows, unique = db.execute("SELECT count(*), count(DISTINCT shot_id) FROM shot_table").fetchone()
    assert rows == unique == shots


def test_no_shootout_shots(db):
    assert one(db, "SELECT count(*) FROM shot_table WHERE period > 4") == 0


def test_penalties_flagged(db):
    assert one(db, "SELECT count(*) FROM shot_table WHERE is_penalty != (shot_type = 'Penalty')") == 0
    assert one(db, "SELECT count(*) FROM np_shots WHERE shot_type = 'Penalty'") == 0


def test_locations_on_pitch(db):
    # 1 unit of tolerance: direct shots from corners are recorded at x = 120.2 (one of them a goal).
    assert one(db, "SELECT count(*) FROM shot_table WHERE x NOT BETWEEN 0 AND 121 OR y NOT BETWEEN -1 AND 81") == 0


def test_feature_ranges(db):
    bad = one(db, f"""
        SELECT count(*) FROM shot_table
        WHERE distance < 0 OR distance > 130 OR angle < 0 OR angle > {math.pi + 1e-9}
           OR defenders_in_cone < 0 OR nearest_defender_dist < 0
           OR gk_dist_from_line < -5 OR gk_off_center < 0
    """)
    assert bad == 0


@pytest.mark.parametrize("col", ["distance", "angle", "body_part", "technique", "shot_type", "play_pattern",
                                 "assist_type", "first_time", "under_pressure", "open_goal", "defenders_in_cone",
                                 "game_state", "formation", "is_goal"])
def test_no_missing_values(db, col):
    assert one(db, f"SELECT count(*) FROM shot_table WHERE NOT is_penalty AND {col} IS NULL") == 0


def test_keeper_missing_rarely(db):
    # Freeze frames sometimes don't show the keeper; it should be rare outside penalties.
    missing = one(db, "SELECT avg((gk_dist_from_line IS NULL)::INT) FROM shot_table WHERE NOT is_penalty")
    assert missing < 0.005


def test_goal_rate_plausible(db):
    rate = one(db, "SELECT avg(is_goal) FROM shot_table WHERE NOT is_penalty")
    assert 0.08 < rate < 0.12


def test_formation_minutes_cover_every_team_match(db):
    team_matches, low, high = db.execute("""
        SELECT count(*), min(m), max(m) FROM (SELECT match_id, team, sum(minutes) m FROM formation_minutes
                                              GROUP BY ALL)
    """).fetchone()
    assert team_matches == 2 * one(db, "SELECT count(*) FROM matches")
    assert 88 <= low and high <= 110


def test_every_shot_inside_a_formation_segment(db):
    assert one(db, """
        SELECT count(*) FROM shot_table s WHERE NOT EXISTS (
            SELECT 1 FROM formation_minutes f WHERE f.match_id = s.match_id AND f.team = s.team
               AND f.formation = s.formation AND f.game_state = s.game_state)
    """) == 0
