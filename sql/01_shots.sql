-- One row per shot (penalty shootouts excluded). Penalties are kept but
-- flagged so they can be analyzed separately; use np_shots for the model.
-- Own goals are separate event types in StatsBomb, so they never appear here.

CREATE OR REPLACE TABLE shots AS
SELECT
    e.id                                         AS shot_id,
    e.match_id,
    m.competition,
    m.match_date,
    e.team,
    CASE WHEN e.team = m.home_team THEN m.away_team ELSE m.home_team END AS opponent,
    e.team = m.home_team                         AS is_home,
    e.player,
    e.player_id::BIGINT                          AS player_id,
    e.period,
    e.minute,
    e.second,
    e.index                                      AS event_index,
    e.possession,
    json_extract(e.location, '$[0]')::DOUBLE     AS x,
    json_extract(e.location, '$[1]')::DOUBLE     AS y,
    e.shot_body_part                             AS body_part,
    e.shot_technique                             AS technique,
    e.shot_type,
    e.play_pattern,
    coalesce(e.shot_first_time, false)           AS first_time,
    coalesce(e.under_pressure, false)            AS under_pressure,
    coalesce(e.shot_open_goal, false)            AS open_goal,
    e.shot_key_pass_id                           AS key_pass_id,
    e.shot_freeze_frame                          AS freeze_frame,
    e.shot_outcome                               AS outcome,
    (e.shot_outcome = 'Goal')::INTEGER           AS is_goal,
    e.shot_statsbomb_xg                          AS statsbomb_xg,  -- benchmark only, never a feature
    e.shot_type = 'Penalty'                      AS is_penalty
FROM events e
JOIN matches m USING (match_id)
WHERE e.type = 'Shot'
  AND e.period <= 4;

CREATE OR REPLACE VIEW np_shots AS
SELECT * FROM shots WHERE NOT is_penalty;
