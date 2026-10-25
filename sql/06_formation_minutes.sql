-- Minutes each team spent in each formation at each game state, so formations
-- can be compared per 90 minutes at the same score. For analysis only.

-- Every moment a team's formation or the score can change, plus period boundaries.
CREATE OR REPLACE TABLE team_breakpoints AS
WITH teams AS (SELECT DISTINCT match_id, team FROM formation_changes)
SELECT e.match_id, t.team, e.index AS event_index, e.period, e.minute * 60 + e.second AS seconds
FROM events e
JOIN teams t USING (match_id)
WHERE e.period <= 4
  AND (e.type IN ('Half Start', 'Half End', 'Starting XI', 'Tactical Shift', 'Own Goal For')
       OR (e.type = 'Shot' AND e.shot_outcome = 'Goal'));

-- State in force from each breakpoint until the next one in the same period.
CREATE OR REPLACE TABLE team_segments AS
WITH states AS (
    SELECT
        b.match_id, b.team, b.event_index, b.period, b.seconds,
        (SELECT arg_max(f.formation, f.event_index) FROM formation_changes f
          WHERE f.match_id = b.match_id AND f.team = b.team AND f.event_index <= b.event_index) AS formation,
        (SELECT count(*) FILTER (WHERE g.scoring_team = b.team) - count(*) FILTER (WHERE g.scoring_team != b.team)
           FROM goals g WHERE g.match_id = b.match_id AND g.event_index <= b.event_index) AS goal_diff
    FROM team_breakpoints b
)
SELECT
    *,
    CASE WHEN lead(period) OVER w = period THEN lead(seconds) OVER w - seconds ELSE 0 END AS duration_seconds
FROM states
WINDOW w AS (PARTITION BY match_id, team ORDER BY event_index);

CREATE OR REPLACE TABLE formation_minutes AS
SELECT
    m.competition, s.match_id, s.team, s.formation,
    game_state_bucket(s.goal_diff) AS game_state,
    sum(s.duration_seconds) / 60.0 AS minutes
FROM team_segments s
JOIN matches m USING (match_id)
WHERE s.formation IS NOT NULL
GROUP BY ALL
HAVING sum(s.duration_seconds) > 0;
