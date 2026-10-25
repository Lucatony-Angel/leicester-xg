-- The one table everything downstream uses: one row per shot, with model
-- features, analysis-only columns and the target. Penalties stay in, flagged.

CREATE OR REPLACE TABLE shot_table AS
SELECT
    s.shot_id, s.match_id, s.competition, s.match_date, s.team, s.opponent, s.is_home,
    s.player, s.player_id, s.period, s.minute, s.second, s.x, s.y, s.is_penalty,
    -- model features
    f.distance, f.angle, f.body_part, f.technique, f.shot_type, f.play_pattern,
    f.assist_type, f.first_time, f.under_pressure, f.open_goal,
    ff.defenders_in_cone, ff.nearest_defender_dist, ff.gk_dist_from_line, ff.gk_off_center,
    -- analysis only
    gs.goal_diff, gs.game_state,
    fm.formation, fm.opponent_formation,
    -- benchmark and target
    s.statsbomb_xg,
    s.is_goal
FROM shots s
JOIN shot_features f USING (shot_id)
LEFT JOIN freeze_frame_features ff USING (shot_id)
JOIN shot_game_state gs USING (shot_id)
LEFT JOIN shot_formations fm USING (shot_id);
