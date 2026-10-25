-- Shot geometry on StatsBomb's 120 x 80 pitch. Goal at x = 120, posts at y = 36 and y = 44.
-- Kept as macros so tests/test_features.py can check the exact SQL used in the pipeline.

CREATE OR REPLACE MACRO shot_distance(x, y) AS
    sqrt(power(120 - x, 2) + power(40 - y, 2));

-- Angle (radians) between the lines from the shot to each post. Wider = more goal to aim at.
CREATE OR REPLACE MACRO shot_angle(x, y) AS
    abs(atan2(44 - y, 120 - x) - atan2(36 - y, 120 - x));

-- Shooting team's goal difference bucketed into 5 game states.
CREATE OR REPLACE MACRO game_state_bucket(goal_diff) AS
    CASE
        WHEN goal_diff <= -2 THEN '-2 or worse'
        WHEN goal_diff = -1  THEN '-1'
        WHEN goal_diff = 0   THEN '0'
        WHEN goal_diff = 1   THEN '+1'
        ELSE '+2 or better'
    END;

-- 4231 -> '4-2-3-1'
CREATE OR REPLACE MACRO format_formation(f) AS
    array_to_string(string_split(f::VARCHAR, ''), '-');
