# Testing and QA

About 200 tests cover the pipeline from raw events to the README. GitHub Actions runs lint and every test that doesn't need the raw StatsBomb download on each push; the database tests skip there. Line coverage is 92% with the raw data downloaded and 84% in CI.

```bash
pip install -r requirements-dev.txt
make test                    # ruff + pytest (all but the slow test)
pytest --cov                 # with coverage
make test-slow               # full rebuild and retrain from the raw download, about 4 minutes
```

## What each layer checks

| Layer | File | What it checks |
|---|---|---|
| Features | `test_features.py` | Distance and angle against hand-checked points, run on the pipeline's own SQL macros; defenders in the cone; missing keepers; game state on a made-up match with an own goal |
| Raw data integrity | `test_data.py` | Match counts per league; goals rebuilt from events give the official final score in all 1,551 matches; one name per team; one row per shot; no shootout shots; penalties flagged; locations on the pitch; feature ranges; no missing model inputs; formation minutes cover every team-match; every shot falls in a formation segment |
| Processed data | `test_processed.py` | Schema; xG is a probability; train/test split grouped by match, about 20% of matches, every league in test; calibration within 3 points in every decile; xG beats a naive baseline; agreement with StatsBomb xG; game state consistent with goal difference; every shot-funnel stage calibrated overall and per decile |
| Model behavior | `test_model.py` | No leaky features (outcome, end location, StatsBomb xG, game state, formation) in the xG model or the funnel stages; closer, central, foot, unblocked, open goal and through ball shots all rate higher; unseen categories don't crash; model ranking in the results table; split is deterministic |
| Analysis math | `test_team_stats.py`, `test_simulate.py` | Team breakdowns, xG shares, ranks and ordinals on tiny hand-built leagues; the game state adjustment and shots-against attribution with known answers; the defensive funnel's three stages adding up exactly; simulation p-values and expected points with certain outcomes; results that don't depend on row order |
| Reproducibility | `test_pipeline.py` | Each analysis script, run fresh, reproduces its committed results file and data exactly (results tables, champions summary, funnel probabilities). The slow test rebuilds the database and retrains from the raw download and checks every shot's xG is identical |
| README accuracy | `test_results.py` | Every number in the README is traced, at the precision shown, to a generated results file; the headline sentences and simulation table are rebuilt from data word for word; every linked file and figure exists |
| App | `test_app.py` | All 6 pages run; team explorer for all 98 teams for and against; every formation league and scoreline filter; player lookup including small-sample warnings |

Numeric warnings (like dividing by zero) fail the tests rather than passing silently. Pages were also checked by hand in a browser at desktop and phone widths.

## Bugs the tests caught

- **Team names.** Some events call Marseille and Caen by short names, which split each into two teams. Rebuilding every final score from goal events exposed it; names are now standardized by team ID.
- **Training order.** Loading shots without a fixed order let a database rebuild change the cross-validation folds and the tuned model. Training now sorts its input, and the slow test proves a rebuild gives identical xG.
- **Simulation order.** Expected points and goal simulations depended on the order of rows in their input files. Each match is now seeded by its ID and shots are sorted before simulating.
- **Empty game states.** A formation with no shots at some scoreline divided zero by zero; it now reports a blank value.
- **Post-shot leakage.** A post-shot xG model built on shot end locations could tell goals from saves almost perfectly, because the free data records saved shots where the keeper stopped them. It was dropped for the shot funnel, which uses only pre-shot inputs.
