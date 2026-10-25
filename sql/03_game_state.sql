-- Score at the moment of each shot, from the shooting team's point of view.
-- For analysis only, not a model feature.

-- Every goal in order. 'Own Goal For' is credited to the team that benefits.
CREATE OR REPLACE TABLE goals AS
SELECT match_id, index AS event_index, team AS scoring_team
FROM events
WHERE (type = 'Shot' AND shot_outcome = 'Goal' AND period <= 4)
   OR type = 'Own Goal For';

CREATE OR REPLACE TABLE shot_game_state AS
SELECT
    s.shot_id,
    count(g.event_index) FILTER (WHERE g.scoring_team = s.team)  AS goals_for_before,
    count(g.event_index) FILTER (WHERE g.scoring_team != s.team) AS goals_against_before,
    goals_for_before - goals_against_before                      AS goal_diff,
    game_state_bucket(goal_diff)                                 AS game_state
FROM shots s
LEFT JOIN goals g
    ON g.match_id = s.match_id
   AND g.event_index < s.event_index
GROUP BY s.shot_id;
