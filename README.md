# Leicester won the 2015/16 Premier League with chance quality and a defense that beat its xG, not lucky finishing

An expected goals (xG) model built from free StatsBomb event data, used to answer one question: **how did a 5000-to-1 underdog win the league: shot quality, shot volume, or finishing luck?**

**Live app:** coming soon

![Leicester took mid-table shot volume but created the 2nd-best chances in the league](figures/team_scatter.png)

## Key findings

- **Quality, not volume.** Leicester averaged 0.114 xG per shot, 2nd in the league behind Arsenal, while taking only the 8th-most shots per match (13.5).
- **Their attacking finishing was average.** They scored 57 non-penalty goals from 58.2 xG. In 10,000 simulated seasons, 60% matched or beat that total. Jamie Vardy scored 19 from 18.9 xG, exactly on par.
- **The overperformance was at the other end.** Leicester conceded 32 non-penalty goals from 44.6 xG against, 12.6 fewer than expected and the most in the league (p = 0.014).
- **Mahrez was the one standout finisher:** 13 goals from 7.2 xG (p = 0.019).
- **That gap was worth about 15 points.** Leicester won 81 points from 66.4 expected points. Their chances alone made them a top-4 side, not champions.

Counterattacks were part of the story: 12.5% of Leicester's xG came from counters, 1.8 times the rest of the league's share. Leicester also took the most penalties in the league (13, scoring 10).

## The question

Leicester were 5000-to-1 to win the 2015/16 Premier League and won it by 10 points. The usual explanations are a hot streak from Vardy and Mahrez, a fast counterattacking style, or plain luck. An xG model separates *how many* chances a team creates, *how good* those chances are, and *how well* they were finished, so it can test each explanation.

## Data

[StatsBomb Open Data](https://github.com/statsbomb/open-data), loaded with [statsbombpy](https://github.com/statsbomb/statsbombpy): 5.4M events from 1,551 matches of the 2015/16 season (Premier League, La Liga, Serie A, Ligue 1 and Bayer Leverkusen's Bundesliga matches), giving 38,728 shots.

**Data provided by StatsBomb.**

## Method

**Features** (14 per shot, built in SQL with DuckDB except for the freeze frame geometry):

| Group | Features |
|---|---|
| Geometry | Distance to goal, angle between the posts |
| Shot context | Body part, technique, free kick vs open play, play pattern, first time, under pressure, open goal |
| Assist | Through ball, cutback, cross, corner, other pass, or none (from the key pass) |
| Freeze frame | Defenders between the ball and the goal, nearest defender distance, keeper distance from line and from center |

Penalties are excluded from the model and given the overall penalty conversion rate across all 5 leagues (0.751) as their xG. Own goals aren't shots. Game state and formation are used for analysis only: the model only knows what the shot looked like.

**Models:** a distance + angle logistic regression baseline, a logistic regression with all features, and XGBoost tuned with 5-fold cross-validation. All splits are grouped by match, and 20% of matches (311) are held out until the end. Each shot's final xG comes from a model that never saw that shot's match, so a team's finishing can't leak into its own xG.

## Model results

Held-out test set: 7,554 shots from 311 matches.

| Model | Log loss | Brier | ROC AUC |
|---|---|---|---|
| Baseline (distance + angle) | 0.2788 | 0.0795 | 0.761 |
| Logistic regression | 0.2544 | 0.0732 | 0.819 |
| **XGBoost** | **0.2535** | **0.0729** | **0.821** |
| StatsBomb xG (benchmark) | 0.2513 | 0.0717 | 0.824 |

XGBoost's log loss is within 0.9% of StatsBomb's own model. Across all 38,318 non-penalty shots, the model predicts 3,647.5 goals against 3,651 actual.

![Calibration plot](figures/calibration.png)

The model is calibrated: shots it rates at 0.2 xG go in about 20% of the time. Distance, angle and the number of defenders in the way drive it most ([SHAP summary](figures/shap_summary.png)).

## Analysis: Leicester vs the Premier League

### Finishing vs defense

![Goals minus xG for all 20 teams](figures/goals_vs_xg.png)

### Was it luck?

Each shot is replayed 10,000 times as a coin flip weighted by its xG. The p-value is the share of simulated seasons at least as extreme as the real one.

![Simulation histograms](figures/simulation.png)

| | Shots | xG | Actual | 90% of simulations | p |
|---|---|---|---|---|---|
| Leicester goals scored | 512 | 58.2 | 57 | 48 to 69 | 0.600 |
| Leicester goals conceded | 513 | 44.6 | 32 | 35 to 54 | 0.014 |
| Jamie Vardy | 112 | 18.9 | 19 | 13 to 25 | 0.523 |
| Riyad Mahrez | 81 | 7.2 | 13 | 3 to 11 | 0.019 |

The defensive number mixes goalkeeping (Kasper Schmeichel), last-ditch blocks and luck, and this model can't separate them: xG only measures the chance, not what happened after the shot.

### Expected points

![Expected points vs actual points](figures/expected_points.png)

Full tables (all 20 teams, play patterns, assist types): [figures/leicester_results.md](figures/leicester_results.md).

## Limitations

- One season only, so a team's luck and skill can't be separated over time.
- Formation is tied to team quality, so formation comparisons show patterns, not causes.
- Player samples are small: Mahrez's 81 shots is a short run, and players with fewer than 30 shots shouldn't be judged on finishing.
- Freeze frames are a single snapshot and don't show player movement.
- xG ignores the shooter's skill by design, and pre-shot xG can't tell good goalkeeping from bad finishing.
- The free Bundesliga data covers only Leverkusen's matches, and 3 Ligue 1 matches are missing.

## How to run it

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/download.py     # StatsBomb events -> data/raw/ (about 15 minutes)
python src/build_db.py     # DuckDB tables and features
python src/train.py        # models + data/processed/shots_with_xg.parquet
python src/evaluate.py     # metrics, calibration, SHAP figures
python src/analysis.py     # Leicester analysis figures and tables

pytest                     # feature and simulation tests
```

## Repo structure

```text
sql/            shots, features, game state, formations (DuckDB)
src/            download, build, freeze frame features, train, evaluate, simulate, analysis
tests/          pytest tests for features and simulations
notebooks/      exploration only
data/processed/ shots_with_xg.parquet, matches.parquet (used by the app)
figures/        charts and results tables
models/         trained models
```

## Next steps

- More seasons, to test whether Leicester's defensive overperformance was skill that repeated.
- A post-shot xG model using shot end location, to separate goalkeeping from defending and luck.
- A finishing skill model for players, pooling information across seasons.
- Tracking data, to replace single-frame defender positions with movement.
