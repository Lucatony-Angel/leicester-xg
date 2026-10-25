-- Formation each team was in at the moment of each shot: the Starting XI,
-- updated by any Tactical Shift before the shot. For analysis only.

CREATE OR REPLACE TABLE formation_changes AS
SELECT
    match_id,
    team,
    index AS event_index,
    format_formation(json_extract_string(tactics, '$.formation')) AS formation
FROM events
WHERE type IN ('Starting XI', 'Tactical Shift');

CREATE OR REPLACE TABLE shot_formations AS
SELECT
    s.shot_id,
    arg_max(f.formation, f.event_index) FILTER (WHERE f.team = s.team)  AS formation,
    arg_max(f.formation, f.event_index) FILTER (WHERE f.team != s.team) AS opponent_formation
FROM shots s
JOIN formation_changes f
    ON f.match_id = s.match_id
   AND f.event_index < s.event_index
GROUP BY s.shot_id;
