# NBA Star Player Points Prediction — Model Findings

## Overview

This document summarises the development and evaluation of an XGBoost model predicting per-game points for 31 NBA star players across multiple seasons. The model was built iteratively, starting from a single-player linear baseline on Luka Dončić and expanding to a multi-player gradient boosting approach.

---

## 1. Development Journey

### Phase 1: Single-Player Linear Model (Luka Dončić)

An OLS regression model was first built using ~400 Luka Dončić games. Key findings:

- The model converged to ~R² ≈ 0.30 on test data with features: `roll5_pts`, `roll10_pts`, `HOME`, `days_rest`, `opp_def_rating`, `age_at_game`.
- **Critical limitation**: the Lakers trade created a structural break — the model trained on Dallas Luka fundamentally could not generalise to Lakers Luka. `season_avg_so_far` partially mitigated this by adapting within a season, but the underlying distribution shift was irreducible.
- **Multicollinearity**: `season_num` and `age_at_game` were near-perfectly correlated and introduced a spurious time-trend. Dropping `season_num` reduced the condition number from ~7.8e+18 to a safe range.
- ~400 games is too small a dataset for XGBoost to avoid overfitting. Train RMSE was 4.75 pts versus test RMSE of 10.48 pts (gap: 5.73 pts) before regularisation.

### Phase 2: Multi-Player XGBoost Model (31 Players)

Expanding to 31 star players increased the dataset to ~15,000 games, providing sufficient volume for a regularised tree ensemble.

**Players included**: Luka Dončić, LeBron James, Stephen Curry, Kevin Durant, Giannis Antetokounmpo, Nikola Jokić, Joel Embiid, Jayson Tatum, Devin Booker, Damian Lillard, Kawhi Leonard, Paul George, Anthony Davis, Jimmy Butler, Ja Morant, Donovan Mitchell, Trae Young, Zion Williamson, Karl-Anthony Towns, Bam Adebayo, Rudy Gobert, Draymond Green, Chris Paul, Russell Westbrook, James Harden, Kyrie Irving, Anthony Edwards, Shai Gilgeous-Alexander, De'Aaron Fox, Tyrese Haliburton, Cade Cunningham.

---

## 2. Feature Engineering

All rolling and expanding features use `.shift(1)` — the prior game — to prevent data leakage. No current-game statistics are used as inputs.

| Feature | Description |
|---|---|
| `roll5_pts` | 5-game rolling average of points (lag-1) |
| `roll10_pts` | 10-game rolling average of points (lag-1) |
| `roll5_ast` | 5-game rolling average of assists (lag-1) |
| `roll5_tov` | 5-game rolling average of turnovers (lag-1) |
| `season_avg_so_far` | Expanding mean of points within season (lag-1) |
| `form_vs_season` | `roll5_pts − season_avg_so_far` (recent form vs baseline) |
| `HOME` | Binary: 1 = home game |
| `days_rest` | Days since last game, clipped at 14 |
| `b2b` | Binary: 1 = back-to-back game |
| `opp_def_rating` | Opponent team defensive rating from `LeagueDashTeamStats` (Advanced) |
| `opp_off_rating` | Opponent team offensive rating (context for pace) |
| `games_on_new_team` | Games played since last team change (trade signal) |

**Opponent ratings** were fetched via `nba_api.stats.endpoints.LeagueDashTeamStats` with `measure_type_detailed_defense='Advanced'` and cached to `team_ratings.json` to avoid rate limiting.

---

## 3. Model Architecture

### XGBoost Hyperparameters

```python
PARAMS = dict(
    n_estimators     = 500,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 10,
    reg_alpha        = 0.5,
    reg_lambda       = 2.0,
    objective        = 'reg:squarederror',
    random_state     = 42,
)
```

Regularisation was intentionally heavy (`min_child_weight=10`, `reg_lambda=2.0`) to control overfitting across a heterogeneous multi-player dataset.

### Train / Test Split

Data was split **chronologically** — never shuffled — to respect the time-series structure. All games from the final season in each player's history were held out as the test set. Shuffling would cause data leakage (future rolling averages would implicitly inform past predictions).

---

## 4. Performance Results

### Overall Metrics

| Split | RMSE | R² |
|---|---|---|
| Train | 7.32 pts | — |
| Test | 8.28 pts | 0.243 |
| Walk-forward (10,867 preds) | 8.40 pts | 0.210 |

**Train/test gap of 0.97 pts** indicates the model generalises well — no meaningful overfitting at this regularisation level.

### Walk-Forward Validation

Walk-forward (expanding window) validation simulates true deployment: the model is retrained on all data up to time _t_ and predicts game _t+1_ across 10 folds. The walk-forward RMSE of **8.40 pts** closely matches the held-out test RMSE of 8.28 pts, confirming that the test result is not an artefact of a single favourable split.

### Quantile Regression (80% Prediction Intervals)

Three models were trained — `q=0.10`, `q=0.50`, `q=0.90` — using `objective='reg:quantileerror'`.

| Metric | Value |
|---|---|
| Empirical coverage | 74.4% |
| Target coverage | 80.0% |
| Average interval width | 18.8 pts |

Coverage of 74.4% falls short of the 80% target. The intervals are slightly under-dispersed — the model is mildly overconfident. This is a known limitation of quantile XGBoost without calibration. A post-hoc conformal prediction step would close this gap. The 18.8 pt average width reflects genuine NBA scoring variance (a player projected for 28 pts might realistically score 19–38 pts on any given night).

### Per-Player RMSE (Selected)

| Player | RMSE |
|---|---|
| Rudy Gobert | 5.92 |
| Draymond Green | 6.31 |
| Bam Adebayo | 7.45 |
| Nikola Jokić | 8.14 |
| Luka Dončić | 8.67 |
| Stephen Curry | 10.02 |

High-variance scorers (Curry, Westbrook, Harden) exhibit the highest RMSE — their scoring swings are genuinely difficult to predict. Defensive specialists (Gobert, Green) score in narrow bands, making prediction easier. This is a structural property of the problem, not a model failure.

---

## 5. SHAP Feature Importance

SHAP (SHapley Additive exPlanations) was used to explain individual predictions.

**Dominant features (in order of mean |SHAP|):**

1. `season_avg_so_far` — the single most informative feature. A player's season baseline carries strong predictive signal.
2. `roll5_pts` — recent 5-game form matters, but less than the season baseline.
3. `roll10_pts` — medium-term trend.
4. `opp_def_rating` — meaningful but secondary. Facing an elite defence (e.g., Boston, Memphis) suppresses scoring.
5. `games_on_new_team` — newly traded players score fewer points in their first games on a new team. This feature confirmed the trade hypothesis.
6. `form_vs_season` — hot/cold streaks relative to season average have modest incremental value.
7. `HOME`, `days_rest`, `b2b` — situational factors with smaller but statistically consistent effects.
8. `opp_off_rating` — pace proxy; faster-paced opponents lead to marginally higher scoring opportunities.

**Key insight**: The model is fundamentally a **regression to the mean** device. It is very good at predicting that a 28 PPG scorer will score around 28 points, and modestly good at adjusting for opponent quality and recent form. It cannot predict the high-variance individual game outcomes that make NBA betting difficult.

---

## 6. Critical Limitations

### 6.1 Predicting the Mean, Not the Variance

An RMSE of 8.28 pts on a target that ranges 0–60 pts means the model's uncertainty is large in absolute terms. The model answers "what is the expected output for this player class of game?" more reliably than "what will happen in this specific game?"

### 6.2 Injury and Load Management

The model has no information about injuries, load management, or minute restrictions. A player returning from a knee injury will be systematically over-predicted. This is arguably the single largest recoverable blind spot.

### 6.3 Structural Breaks (Trades)

`games_on_new_team` partially captures adjustment periods, but the fundamental issue — a player's statistical profile shifts non-stationarily after a trade — is not fully resolved. Post-trade predictions should be interpreted with lower confidence.

### 6.4 Opponent Features Are Season-Level

`opp_def_rating` is a full-season team average, not a rolling in-season estimate. Early in the season, these ratings are especially noisy. A rolling 15-game opponent defensive rating would be more informative.

### 6.5 Cross-Player Generalisation

The model learns a single function across 31 very different players (centres, guards, scorers, defenders). A player-specific model or player embedding could capture idiosyncratic tendencies (e.g., Curry's deep three reliance vs. Embiid's post scoring). However, player-specific models would each have ~500 games — insufficient for XGBoost without significant regularisation.

### 6.6 No Lineup / Team Context

The model does not know who a player's teammates are. LeBron's points are influenced by whether Anthony Davis is healthy. Giannis's scoring changes with coaching schemes. These latent factors are partially absorbed into `season_avg_so_far` but not explicitly modelled.

---

## 7. What the Model is Good At

- **Ranking relative difficulty** of upcoming games (strong opponent defence → lower predicted output)
- **Identifying form** — players on hot streaks are predicted higher; cold stretches lower
- **Trade adjustment** — `games_on_new_team` correctly identifies the early-career-on-new-team suppression effect
- **Baseline point projection** as a prior for fantasy sports, sports analytics, or further modelling
- **Interval estimation** — the quantile intervals, while slightly underconfident, give a useful range of outcomes

---

## 8. Potential Next Steps

| Improvement | Expected Impact |
|---|---|
| Rolling opponent ratings (15-game window) | Moderate — removes early-season noise |
| Conformal prediction calibration | Fixes quantile coverage gap from 74.4% → ~80% |
| Minutes played as feature | High — accounts for load management |
| Player embeddings (learned latent vector) | High — captures player archetype differences |
| Injury indicator (binary) | Very high — removes systematic over-prediction post-injury |
| Opponent lineup quality (PER-weighted) | Moderate — better than season-average opponent rating |
| In-season retraining cadence | Moderate — model trained on stale data degrades late-season |

---

## 9. Conclusion

An XGBoost model trained on 15,000+ games from 31 NBA star players achieves a test RMSE of **8.28 points** and R² of **0.243**, with no meaningful overfitting (train/test gap of under 1 pt). Walk-forward validation corroborates this, returning an RMSE of 8.40 pts across 10,867 out-of-sample predictions.

The model is most accurately characterised as a **calibrated prior**: it reliably identifies a player's likely scoring range for a given game context, adjusts meaningfully for opponent defence, recent form, and situational factors, and provides prediction intervals that cover ~74% of outcomes. It does not — and cannot — predict individual game variance driven by hot/cold shooting nights, foul trouble, or opponent-specific matchup quirks.

For a quantitative sports analytics use case, this model represents a solid baseline. The clearest path to improvement is incorporating player availability and rolling opponent quality rather than season-aggregate ratings.
