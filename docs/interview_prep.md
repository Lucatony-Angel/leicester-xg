# Interview prep

Answers to the questions this project invites, with the real numbers. Say them in your own words; the numbers are what matter.

## 60-second pitch

1. **Hook.** "I coached youth soccer for 3 years and always wondered what actually creates goals, so I built an expected goals model."
2. **Question.** "I used it to test how Leicester won the 2015/16 Premier League as a 5000-to-1 underdog: better chances, more chances, or lucky finishing."
3. **Method.** "Features in SQL with DuckDB from 5.4 million StatsBomb events, an XGBoost model calibrated and benchmarked against StatsBomb's own, and Monte Carlo simulation for the luck question."
4. **Finding.** "It wasn't finishing luck: their attack scored 57 from 58.3 xG. They created the 2nd-best chances in the league on mid-table volume, and their defense conceded 12.5 fewer goals than its xG, the most in the league. That was worth about 15 points: 81 actual from 66.5 expected."
5. **Lesson.** "My first attempt to split that defensive number used where shots ended up, and it looked amazing, until I realized the data records saves where the keeper touched the ball, so the model was reading the outcome. I replaced it with a model of each stage of a shot that only uses pre-shot information."

## Modeling

**Why log loss and not accuracy?**
Goals are about 9.5% of non-penalty shots, so a model that always says "no goal" is about 90% accurate and useless. xG is a probability that gets added up into team totals, so what matters is whether 0.2 means 20%. Log loss and Brier score reward honest probabilities; the calibration plot checks it directly (9 of 10 held-out bins sit within their 95% interval of the diagonal). AUC (0.821) only measures ranking.

**How did you avoid leakage?**
- Nothing recorded after the shot is a feature: no outcome, no end location, no StatsBomb xG (benchmark only). A test fails if any of these enter the feature list.
- Game state and formation are kept out of the model too. They describe the team and the scoreline, not the shot.
- Each shot's final xG comes from a model that never saw that shot's match (out-of-fold). Otherwise Leicester's goals would leak into Leicester's xG and shrink the very finishing luck being measured.

**Why group the splits by match?**
Shots in the same match share a lot: the same keeper, the same weather, rebounds from the same attack. If shots from one match land in both train and test, the test score is optimistic. Every split, the 5 cross-validation folds and the 20% held-out test set (311 matches), keeps each match on one side.

**How close are you to StatsBomb?**
Log loss 0.2535 against StatsBomb's 0.2513 on the same held-out shots, within 0.9%. StatsBomb's model uses more information than the free data has. A plain logistic regression got 0.2544, so the features do most of the work and XGBoost adds a little.

**What did SHAP show?**
Distance and angle matter most, then defenders between the ball and the goal and the nearest defender's distance, then the keeper's position and headers. A Vardy goal from 18.6 yards was a header with a defender 1.2 yards away, rated 0.009 xG: a genuinely unlikely goal, not a model bug.

**Why XGBoost if logistic regression is nearly as good?**
It's slightly better on every metric and handles interactions (a header from close range is different from a header from distance) without hand-built terms. If interpretability mattered more than the last 0.4% of log loss, I'd ship the logistic regression.

## Analysis

**How do you know the finishing wasn't luck?**
Each Leicester shot replayed 10,000 times as a coin flip weighted by its xG: 60% of simulated seasons scored at least their 57 goals. Vardy (19 from 18.9) was dead average. Mahrez (13 from 7.2, p = 0.021) was the exception.

**So was the defense lucky?**
Partly, and I'm careful about it. Conceding 32 from 44.5 xG has p = 0.016 for one team, but I picked Leicester because they won. If all 20 teams were average, there's a 27% chance at least one defense gets that far ahead of its xG. The split by stage shows it came from everywhere: Schmeichel +5.6, opponents missing the target +3.5, blocks +3.4. Schmeichel's share alone isn't significant (p = 0.137). So: a lot went right at once, some skill and some fortune.

**How did you split the defensive number?**
A shot has to avoid a block, hit the target and beat the keeper. For each stage I trained a model on the same pre-shot features to give the average chance of getting through, then compared what actually happened against Leicester. The three parts add up exactly to xG against minus goals conceded, and a test checks that.

**Why not post-shot xG?**
I tried it. In the free data, saved shots end where the keeper touched them and goals end on the goal line, so end location partly encodes the outcome, and a test model told goals from saves almost perfectly. That's leakage from the recording, not skill, so I dropped it.

**Is 4-4-2 why Leicester won?**
No, and I wouldn't claim it. Formations differ only modestly at the same scoreline (0.90 to 1.21 xG per 90), and strong teams choose some formations more, so it's correlation. What the data does say: among 20 teams that played 4-4-2, Leicester created the most xG per 90. They were an outlier within the shape, not a typical example of it.

**How would you tell a club to act on this?**
- Judge strikers on xG and shot quality over a season, not goals over a few weeks: Vardy's 19 goals were exactly what his chances were worth.
- Treat a defense that concedes far below its xG as likely to regress unless the stages show something repeatable, like a team that consistently blocks more than expected.
- Value chance quality: Leicester's 0.114 xG per shot on average volume beat teams that shot more.

## Engineering

**How do you know your numbers are right?**
About 200 tests. Goals rebuilt from raw events match the official final score in all 1,551 matches. Every number in the README is checked against generated results, and the headline sentences are rebuilt from data word for word. Every analysis script must reproduce its committed output exactly, and a slow test rebuilds the database and retrains from scratch and gets identical xG for every shot.

**Tell me about a bug.**
Rebuilding the database changed the tuned XGBoost settings, because I loaded shots without a fixed order and the cross-validation folds shifted. Several settings score almost the same, so a different one won. I fixed the order and added a test that a full rebuild reproduces every shot's xG. The tests later caught the same kind of problem in the simulations: expected points changed when the matches file was re-sorted.

**What would you do with more data?**
More seasons, to see whether Leicester's defensive edge repeated (skill) or faded (luck); consistent end locations for a real post-shot model; and tracking data instead of a single freeze frame.
