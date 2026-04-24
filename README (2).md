# Grid Demand Forecast

Next-hour power demand prediction for the Bangladesh grid (PGCB), using hourly demand data enriched with weather and a handful of World Bank macro indicators. The final model is a tuned LightGBM; Ridge and XGBoost are kept alongside as reference points.

The target is `demand_mw` one hour ahead. The whole pipeline runs end-to-end from the notebook — loading the raw files, cleaning, feature engineering, training, tuning, and exporting a CSV of predictions.

---

## Contents

- `predictive_paradox_refined.ipynb` — the full pipeline
- `predictions_2023.csv` — test-year predictions (generated on run, saved to `/mnt/user-data/outputs`)
- `README.md` — this file

## Data

Three inputs, all expected in `DATA_DIR` (set to `/content` in the notebook — change it to wherever you keep them):

| File | Source | What it is |
|---|---|---|
| `PGCB_date_power_demand.xlsx` | Power Grid Company of Bangladesh | Hourly demand, generation, fuel mix, cross-border imports, load shedding |
| `weather_data.xlsx` | Hourly weather | Temperature, humidity, dew point, precipitation, cloud cover, wind, sunshine |
| `economic_full_1.csv` | World Bank indicators | Annual macro series (GDP, population, urbanization, electricity access, etc.) |

## How to run

```bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib openpyxl
```

Open the notebook, update `DATA_DIR` if needed, and run top-to-bottom. On a normal laptop the full run (including the 30-trial random search) takes a few minutes.

---

## Summary report

### Handling missing data and outliers

The demand file was messier than the headline stats suggested, so this bit got more care than anything else in the pipeline.

**Frequency.** The file had a mix of hourly and half-hour rows. Rather than try to resample, I kept the hourly rows only (minute == 0) — this matches the target frequency and avoids synthetic averaging where the data was already on the grid I wanted.

**Duplicates.** Some timestamps appeared more than once. I collapsed them with a group-by mean on the numeric columns. Mean was the right choice here over "first" or "last" since the duplicates looked like near-identical readings rather than corrections.

**Unit typos.** Two specific outlier patterns showed up that are clearly data-entry errors rather than real extremes:

- **Demand reported ~10x too high.** Flagged when `demand_mw > 5 × generation_mw` (demand physically can't exceed generation plus imports by that margin). Divided those values by 10.
- **Generation above 50,000 MW.** The Bangladesh grid capacity is nowhere near this, so anything over 50k is a misplaced decimal. Divided by 10,000.

I chose hard rules over statistical thresholds (like z-scores or IQR) because these are decimal-point errors with a clean signature, not distributional outliers. A z-score filter would either miss them or start shaving legitimate peak-load hours, which we specifically *don't* want the model to forget.

**Implausibly small values.** Any reading below 500 MW was set to NaN rather than kept or dropped — 500 MW is far below the country's overnight minimum, so these are almost certainly meter dropouts or comms failures.

**Gaps.** After reindexing to a full hourly DatetimeIndex, I filled gaps with time-based interpolation limited to 6 hours, then forward/back-fill for anything left over. The 6-hour cap matters: interpolating across a whole day of missing demand would smooth over real load patterns (morning ramp, evening peak) and give the model a cleaner signal than reality deserves. Short gaps get a principled fill; longer ones fall back to a held-constant value that the lag features can absorb.

For weather, values were already clean — I just deduped the index and merged with a left join so that any missing weather hour stays as NaN and the tree models handle it natively. For the macro indicators, which are annual, I forward-filled once (values don't move hour-to-hour anyway) and back-filled any leading gaps.

One column group (`solar`, `wind`, `india_adani`, `nepal`) was filled with 0 rather than interpolated — these are supply sources that were literally zero before they came online, so NaN means "not yet in the mix," not "missing."

### Temporal features

The target is next-hour demand, which has very strong autocorrelation, a hard daily cycle, and a softer weekly one. The features reflect that.

**Calendar fields.** Plain integer `hour`, `dow`, `day`, `month`, plus a binary `is_weekend`. Trees handle these fine as integers, so I didn't bother one-hot encoding.

**Cyclical encoding.** Sin/cos pairs for `hour` and `month`. Trees don't strictly need this (they can split on integer `hour`), but the Ridge baseline definitely does — without it, the model treats hour 23 and hour 0 as maximally distant, which they obviously aren't. Keeping the cyclical version also costs almost nothing for the tree models.

**Lags.** `demand_lag_{1, 2, 3, 6, 12, 24, 48, 72, 168}`. The logic:

- **1, 2, 3** — recent momentum. Lag-1 alone carries most of the predictive signal.
- **6, 12** — catches same-shift and half-day-ago behavior.
- **24** — same hour yesterday. Almost as important as lag-1 for daily-cycle workloads.
- **48, 72** — two and three days back, useful for sensing weekly drift.
- **168** — same hour, same weekday, one week ago. Captures the weekly seasonality directly.

**Rolling statistics.** Mean and standard deviation over windows of 3, 6, 24, and 168 hours, all computed on `demand.shift(1)` so no future information leaks in. The mean tracks the local level; the std tracks volatility (useful around weather fronts or grid events).

**Differences.** `demand_diff_1` (change over the last hour) and `demand_diff_24` (change vs. same hour yesterday). These give the model explicit access to recent momentum and same-hour-day-over-day drift without forcing it to subtract two lags itself.

**Weather lags.** `temp_lag_1` and a 24-hour rolling mean of temperature (both on shifted values). Cooling load responds to temperature with a lag, and a daily average smooths out the intraday temperature swing that would otherwise confuse the model around dawn and dusk.

All of the demand-derived features are built from `demand.shift(1)` or further back, and the target is `demand.shift(-1)`. Nothing in the feature set can see the value it's predicting.

### Feature importances

The final LightGBM's top features by gain are plotted in section 17 of the notebook. The ranking is dominated, unsurprisingly, by the recent demand lags — `demand_lag_1`, the short rolling means, and `demand_lag_24` — which is what you'd expect for short-horizon load forecasting. Calendar features (hour, month, weekend) come next, then temperature-related features. Macro indicators contribute the least per-feature, which also tracks — they barely vary at the hourly scale, so their value shows up more in setting the overall level than in hour-to-hour prediction.

The plot itself is generated at the bottom of the notebook (the top-20-by-gain bar chart). Check there for the exact ordering from the current run — it shifts slightly between random seeds.

---

## Pipeline overview

```
Raw demand (xlsx)  ──┐
                     ├── clean ──┐
Weather (xlsx)  ─────┤           ├── merge ── feature engineering ── chronological split
                     │           │
Economics (csv) ─────┘           │
                                 ▼
                    Ridge → XGBoost → LightGBM → random search → tuned LightGBM
                                                                        │
                                                                        ▼
                                                             predictions_2023.csv
```

### Train/test split

Chronological. Train is everything before 2023-01-01, test is all of 2023. For early stopping inside XGBoost and LightGBM, the last six months of training (July–December 2022) are held out as a validation set. No shuffling — this is a time series and a random split would leak future into past.

### Models

| Model | Role |
|---|---|
| Ridge (scaled) | Linear baseline. Sanity check that the feature set is well-behaved. |
| XGBoost | Gradient-boosted trees with early stopping on the 2022-H2 validation set. |
| LightGBM | Same family, usually faster; picked as the base for tuning. |
| LightGBM (tuned) | Final model. 30 trials of random search over num_leaves, learning rate, min_child_samples, subsample, colsample_bytree, and L1/L2 regularization. |

### Metrics

MAPE (%), MAE (MW), and RMSE (MW) on the 2023 hold-out. MAPE is the headline number since grid demand rarely dips near zero, so it's well-defined and interpretable.

### Output

`predictions_2023.csv` with one row per test hour:

- `datetime` — hour being predicted (i.e. the *next* hour at feature time)
- `demand_mw` — actual
- `predicted_demand_mw_next_hour` — predicted
- `abs_error_mw` — |actual − predicted|
- `pct_error` — abs error as a percentage of actual

A quick reload-and-verify step at the end confirms the CSV's MAPE matches the in-memory number, just so we catch any silent serialization issues.

---

## Notes and caveats

- The one-hour horizon makes lag-1 dominant. A longer horizon (say, 24h or 48h ahead) would lean much more heavily on calendar features and weather forecasts rather than recent demand — worth redoing the feature set if you're retargeting.
- Half-hour rows are dropped rather than averaged with their surrounding hours. If a newer PGCB dataset is half-hourly throughout, a resample-to-hourly step would be a cleaner fit.
- The macro indicators add marginal lift. They're kept in mostly because they're cheap and provide a slow-moving baseline for multi-year forecasts; on a pure next-hour task you could drop them with minimal cost.
- `DATA_DIR = Path("/content")` is the Colab default. Change it to your local path before running off Colab.
