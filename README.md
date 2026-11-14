# Freight Rate Prediction Challenge

Predict the posted rate ($) of freight loads. The model is trained on January–October 2025 loads
and predicts 12,000 unseen loads from November–December 2025, plus a fixed December 2025 chart load.

**Final model:** a 70/30 blend of two *hybrid* models (a Ridge regression for the market level and a
LightGBM model for route effects). Details are in [How the model works](#how-the-model-works).

## Results (honest backtest, past-to-future)

Each validation fold trains on earlier months and predicts the next 1–2 months, which mimics the real task.
Spotter's own score on the hidden set is calculated after submission.

| Model | WAPE, clean rows | WAPE, all rows (incl. corrupted labels) |
|---|---|---|
| Naive (average rate) | 47.8% | 48.9% |
| Lookup table (equipment x distance) | 3.6% | 5.8% |
| **Final model** | **2.8%** | **5.0%** |

- Predictions were unbiased across folds (sum predicted / sum actual between 0.978 and 1.026).
- 81.5% of loads were predicted within 5% of the true rate, 99.2% within 10% (clean rows).
- "All rows" is higher because about 1.4% of training labels are corrupted (3-5x too high or ~0.28x too low).
  The hidden test set probably has the same noise, so expect a score closer to the "all rows" column.

## Repository layout

```
.
├── freight_lib.py            # cleaning, features, models, metrics (the reusable code)
├── make_submission.py        # trains the final model and writes the two output files
├── score.py                  # official scorer provided with the assessment
├── requirements.txt          # packages needed to run the pipeline
├── requirements-dev.txt      # + Jupyter and pytest (for notebooks and tests)
├── pytest.ini
├── data/                     # put the four input CSVs here (not committed)
├── notebooks/
│   ├── 01_eda.ipynb                          # exploration and findings
│   ├── 02_cleaning_and_validation.ipynb      # cleaning, validation design, quote_signal test
│   ├── 03a_feature engineering and model selection.ipynb
│   ├── 03b_Tuning_and_checks.ipynb           # tuning, blending, December sanity check
│   ├── 04_final_pipeline.ipynb               # runs the pipeline and scorer
│   └── 05_evaluation.ipynb                   # baselines, bias, error by segment
├── tests/                    # unit tests and an end-to-end smoke test
├── validation_predictions.csv   # submission file (load_id,predicted_rate)
├── december_chart_inputs.csv    # December file with predicted_rate filled in
└── scorer_results/candidate_december.png
```

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:    .venv\Scripts\activate
# Mac/Linux:  source .venv/bin/activate
python -m pip install -r requirements.txt
```

Put these four files in `data/`. Hyphenated names such as `train-test.csv` also work:
`train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv`.

## Run

```bash
# 1. Train on all clean data and write the two output files
python make_submission.py --data-dir data --out-dir .

# 2. Validate the files and create the December chart
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv
```

Output: `validation_predictions.csv`, `december_chart_inputs.csv` and `scorer_results/candidate_december.png`.
Training takes a few seconds. Random seeds are fixed, so results are repeatable on the same package versions.

Run the tests (install `requirements-dev.txt` first):

```bash
python -m pytest
```

The notebooks run in the order of their numbers. In the first cell of each, `DATA_DIR` points to the folder
with the CSVs (it defaults to your Downloads folder; change it to `data`).
Notebook 04 moves to the project root itself.

## Approach

### Data-quality issues found and how they were handled

| Issue | Handling |
|---|---|
| `weight`: 292 negative values, 300 missing | `abs()`, then fill with the median for that equipment type (learned from train only) |
| `market_index` has gaps | It is one number per day, so a gap is filled with that day's median |
| ~1.4% of rates (677 rows) are corrupted (3-5x too high or ~0.28x too low) | Flagged by comparing $/mile with the typical value for the equipment and distance band; **dropped from training only** |
| 8 cities appear only in the validation set (12.1% of rows) | Model uses distance and equipment, not city names; checked that unseen-city loads get normal predictions |
| `quote_signal` flips its relationship with rate between months (+0.45 to -0.45) | **Excluded.** The validation period looks like August, when it had no effect; an ablation showed it did not help (WAPE 3.73% with it vs 3.65% without) |

### Validation and data split

The training file covers Jan-Oct 2025 and the validation set covers Nov-Dec 2025, so the real task is
predicting the future. A random split would let the model see neighbouring days with the same market
conditions (MAE 43 on a random split vs 91 on forward folds), so it would be misleadingly good.
Forward-chaining folds are used instead:

| Fold | Train | Test |
|---|---|---|
| V1 | Jan-Jun | Jul-Aug |
| V2 | Jan-Jul | Aug-Sep |
| V3 | Jan-Aug | Sep-Oct |
| V4 | Jan-Sep | Oct |

V2-V4 overlap, so they are effectively fewer than four independent tests.
All fill values and thresholds are learned from the training part of each fold only.

### How the model works

1. **Ridge line:** predicts log(rate) from log distance, `market_index`, months since 1 Jan, and equipment.
   This part follows the daily market level and can extrapolate to new dates.
2. **LightGBM on the leftovers:** learns what the line missed (curved distance effects, equipment, weight)
   from route features only.
3. **Prediction:** `exp(line + trees)`.
4. **Blend:** 70% of the version with the time term and 30% of the version without it. The two fail on different
   folds, so the blend has the best worst-fold result (3.15% vs 4.7% for the with-time model alone).

### December chart

For the fixed load (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb) only the date changes.
Each day's `market_index` is looked up from `validation.csv` (features only, no labels). The weekly
wave in the chart comes from the weekly cycle in `market_index`, which matches a weekday effect of
about 2% in the training data.

## Limitations

- Only 10 months of history. The time term assumes a drift that continues into Nov-Dec; blending limits the risk.
- Corrupted labels in the hidden set cannot be removed, so the final error will be higher than the clean-row numbers.
- No holiday effect can be learned (no December in the training data).
- The 70/30 blend weight was chosen on the same folds used for reporting, so the reported WAPE is slightly optimistic.
