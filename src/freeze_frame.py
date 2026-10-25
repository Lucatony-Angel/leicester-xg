"""Defender and goalkeeper features from shot freeze frames.

A freeze frame lists every visible player at the moment of the shot (the
shooter is not included). Coordinates are StatsBomb's 120 x 80 pitch.
"""

import json
import math

import duckdb
import pandas as pd

LEFT_POST = (120.0, 36.0)
RIGHT_POST = (120.0, 44.0)
GOAL_CENTER_Y = 40.0


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def in_cone(shot: tuple, point: tuple) -> bool:
    """True if point is inside the triangle from the shot to both posts (edges count)."""
    d1 = _cross(shot, LEFT_POST, point)
    d2 = _cross(LEFT_POST, RIGHT_POST, point)
    d3 = _cross(RIGHT_POST, shot, point)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)


def freeze_frame_features(x: float, y: float, frame: list[dict]) -> dict:
    """Features for one shot at (x, y) given its parsed freeze frame."""
    opponents = [p for p in frame if not p["teammate"]]
    keeper = next((p for p in opponents if p["position"]["name"] == "Goalkeeper"), None)
    defenders = [p["location"] for p in opponents if p is not keeper]

    return {
        "defenders_in_cone": sum(in_cone((x, y), d) for d in defenders),
        "nearest_defender_dist": min((math.dist((x, y), d) for d in defenders), default=None),
        "gk_dist_from_line": 120.0 - keeper["location"][0] if keeper else None,
        "gk_off_center": abs(keeper["location"][1] - GOAL_CENTER_Y) if keeper else None,
    }


def build_freeze_frame_table(con: duckdb.DuckDBPyConnection) -> None:
    """Compute features for every shot in `shots` and save as freeze_frame_features."""
    shots = con.execute("SELECT shot_id, x, y, freeze_frame FROM shots").df()
    rows = []
    for s in shots.itertuples():
        frame = json.loads(s.freeze_frame) if isinstance(s.freeze_frame, str) else []
        rows.append({"shot_id": s.shot_id, **freeze_frame_features(s.x, s.y, frame)})
    features = pd.DataFrame(rows)
    con.execute("CREATE OR REPLACE TABLE freeze_frame_features AS SELECT * FROM features")
