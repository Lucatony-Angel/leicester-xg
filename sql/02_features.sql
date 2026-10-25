-- Model features that can be built in SQL: geometry, shot context, and assist type.
-- Only things known at the moment of the shot. No end location, outcome or StatsBomb xG.

CREATE OR REPLACE TABLE shot_features AS
SELECT
    s.shot_id,
    shot_distance(s.x, s.y)                      AS distance,
    shot_angle(s.x, s.y)                         AS angle,
    s.body_part,
    s.technique,
    s.shot_type,
    s.play_pattern,
    CASE
        WHEN s.key_pass_id IS NULL                                   THEN 'None'
        WHEN p.pass_through_ball OR p.pass_technique = 'Through Ball' THEN 'Through Ball'
        WHEN p.pass_cut_back                                         THEN 'Cutback'
        WHEN p.pass_type = 'Corner'                                  THEN 'Corner'
        WHEN p.pass_cross                                            THEN 'Cross'
        ELSE 'Other Pass'
    END                                          AS assist_type,
    s.first_time,
    s.under_pressure,
    s.open_goal
FROM shots s
LEFT JOIN events p ON p.id = s.key_pass_id;
