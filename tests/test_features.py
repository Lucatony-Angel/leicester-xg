import math
from pathlib import Path

import duckdb
import pytest

from freeze_frame import freeze_frame_features, in_cone

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute((SQL_DIR / "00_macros.sql").read_text())
    yield con
    con.close()


def sql_value(con, expr):
    return con.execute(f"SELECT {expr}").fetchone()[0]


# --- distance and angle (sql/00_macros.sql) ---

@pytest.mark.parametrize("x, y, expected", [
    (108, 40, 12.0),               # penalty spot, dead center
    (100, 40, 20.0),
    (108, 28, math.sqrt(12**2 + 12**2)),
    (120, 40, 0.0),
])
def test_distance(con, x, y, expected):
    assert sql_value(con, f"shot_distance({x}, {y})") == pytest.approx(expected)


@pytest.mark.parametrize("x, y, expected", [
    (108, 40, 2 * math.atan(4 / 12)),          # ~36.9 degrees
    (100, 40, 2 * math.atan(4 / 20)),
    (108, 28, math.atan(16 / 12) - math.atan(8 / 12)),
    (120, 40, math.pi),                        # on the goal line between the posts
    (120, 20, 0.0),                            # on the goal line, wide: no goal to aim at
])
def test_angle(con, x, y, expected):
    assert sql_value(con, f"shot_angle({x}, {y})") == pytest.approx(expected)


def test_angle_is_symmetric(con):
    assert sql_value(con, "shot_angle(108, 30)") == pytest.approx(sql_value(con, "shot_angle(108, 50)"))


def test_angle_shrinks_with_distance(con):
    assert sql_value(con, "shot_angle(110, 40)") > sql_value(con, "shot_angle(100, 40)")


# --- game state ---

@pytest.mark.parametrize("diff, expected", [
    (-4, "-2 or worse"), (-2, "-2 or worse"), (-1, "-1"), (0, "0"), (1, "+1"), (2, "+2 or better"), (5, "+2 or better"),
])
def test_game_state_bucket(con, diff, expected):
    assert sql_value(con, f"game_state_bucket({diff})") == expected


def test_game_state_counts_goals_before_each_shot(con):
    # Match: A scores, B scores an own goal for A, B scores. Shots taken in between.
    con.execute("""
        CREATE TABLE events AS SELECT * FROM (VALUES
            (1, 10, 'Shot',         'Goal',  1, 'A'),
            (1, 20, 'Own Goal For', NULL,    1, 'A'),
            (1, 30, 'Shot',         'Saved', 2, 'B'),
            (1, 40, 'Shot',         'Goal',  2, 'B'),
            (1, 50, 'Shot',         'Off T', 2, 'A')
        ) t(match_id, index, type, shot_outcome, period, team)
    """)
    con.execute("""
        CREATE TABLE shots AS
        SELECT index::VARCHAR AS shot_id, match_id, team, index AS event_index
        FROM events WHERE type = 'Shot'
    """)
    con.execute((SQL_DIR / "03_game_state.sql").read_text())
    result = dict(con.execute("SELECT shot_id, goal_diff FROM shot_game_state").fetchall())

    assert result["10"] == 0     # first shot: 0-0, and its own goal doesn't count yet
    assert result["30"] == -2    # B shooting at 0-2 (A's goal + own goal)
    assert result["40"] == -2    # still 0-2 when B scores
    assert result["50"] == 1     # A shooting at 2-1


def test_format_formation(con):
    assert sql_value(con, "format_formation(4231)") == "4-2-3-1"
    assert sql_value(con, "format_formation(442)") == "4-4-2"


# --- freeze frames (src/freeze_frame.py) ---

def player(x, y, teammate=False, position="Center Back"):
    return {"location": [x, y], "teammate": teammate, "position": {"name": position}}


def test_in_cone():
    shot = (108, 40)
    assert in_cone(shot, (114, 40))        # between shooter and goal center
    assert in_cone(shot, (119, 37))        # near the left post, inside
    assert not in_cone(shot, (114, 30))    # off to the side
    assert not in_cone(shot, (104, 40))    # behind the shooter


def test_freeze_frame_features():
    frame = [
        player(114, 40),                                   # in the cone, 6 away
        player(108, 43),                                   # nearest defender, 3 away, outside cone
        player(110, 40, teammate=True),                    # teammates are ignored
        player(118, 42, position="Goalkeeper"),
    ]
    f = freeze_frame_features(108, 40, frame)
    assert f["defenders_in_cone"] == 1
    assert f["nearest_defender_dist"] == pytest.approx(3.0)
    assert f["gk_dist_from_line"] == pytest.approx(2.0)
    assert f["gk_off_center"] == pytest.approx(2.0)


def test_keeper_not_counted_as_defender():
    f = freeze_frame_features(108, 40, [player(115, 40, position="Goalkeeper")])
    assert f["defenders_in_cone"] == 0
    assert f["nearest_defender_dist"] is None


def test_missing_keeper():
    f = freeze_frame_features(108, 40, [player(114, 40)])
    assert f["gk_dist_from_line"] is None
    assert f["gk_off_center"] is None
