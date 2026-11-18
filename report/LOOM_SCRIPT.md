# Loom Presentation Script: Freight Rate Prediction Challenge

**Presenter:** Nouman Aref  
**Target Duration:** 2 minutes 30 seconds (Hard cap: 3:00)  
**Setup:** Camera ON, Screen sharing repository code & December chart.

---

### [0:00 - 0:25] Introduction & Task Framing
* **Visual:** Face on camera / README.md overview.
* **Speaking Points:**
  > "Hi everyone, I'm Nouman. Today I'm walking you through my solution for the Freight Rate Prediction Challenge.
  >
  > The goal is to predict spot freight rates for 12,000 loads across November and December 2025, using development data from January through October.
  >
  > Because the evaluation set is strictly forward in time, our core objective is out-of-distribution temporal forecasting—not just static curve fitting."

---

### [0:25 - 0:55] Exploratory Data Analysis & Data Quality Fixes
* **Visual:** Scroll to `01_eda.ipynb` or `src/anomalies.py`.
* **Speaking Points:**
  > "During initial data exploration, I identified four critical data quality anomalies:
  >
  > 1. **Weight anomalies:** 292 records had negative weights—obvious sign errors—which were resolved with `abs()`. 300 missing values were imputed using equipment-specific medians learned strictly on train.
  > 2. **Market Index coverage:** `market_index` is a daily macro signal, but had isolated missing load entries. I built a continuous daily calendar with forward-fill fallback.
  > 3. **Corrupted target labels (~1.4%):** 677 loads had synthetic rate corruption—either 3 to 5 times too high or 70% too low. I flagged these using rate-per-mile deviation bounds by equipment and distance band, purging them **strictly from training** to protect gradient estimators, while retaining them in validation to reflect real-world noise.
  > 4. **Quote Signal trap:** `quote_signal` exhibited a severe regime inversion—its correlation swung from +0.45 to -0.45 across months. An ablation test proved that dropping it improved out-of-fold WAPE from 3.73% to 3.65%."

---

### [0:55 - 1:35] Validation Strategy & Data Split
* **Visual:** Point to Forward Chaining Fold table in README.
* **Speaking Points:**
  > "To ensure honest evaluation, standard random K-fold cross-validation was rejected. In my experiments, random splitting yielded a deceptively optimistic MAE of $43 due to temporal autocorrelation, whereas true held-out forward folds yielded an honest MAE of $91.
  >
  > I constructed a 4-fold forward-chaining backtest (V1 through V4) where models train strictly on earlier months and forecast the subsequent 1 to 2 month horizon—mirroring our exact November–December production task.
  >
  > All imputation statistics and scaling parameters were strictly fitted on the training split of each fold."

---

### [1:35 - 2:10] Model Architecture & Ensembling
* **Visual:** Open `src/models.py`.
* **Speaking Points:**
  > "Gradient-boosted trees alone cannot extrapolate macroeconomic time drift, while linear models struggle with non-linear route geometry.
  >
  > I designed a two-stage residual Hybrid architecture:
  > - **Stage 1 (Regularized Linear Core):** A Ridge model in log-space captures macro market shifts via `market_index`, calendar trends, and baseline equipment costs.
  > - **Stage 2 (Non-linear Residual Booster):** A LightGBM regressor fits the residuals to model non-linear distance economies of scale—where rate-per-mile falls with haul length—and equipment-weight interactions.
  >
  > To hedge against long-horizon drift risk, I ensembled a model with a linear time trend (70%) and a model without time trend (30%). On clean out-of-sample backtests, this achieved a **2.79% WAPE** and **0.998 financial bias**, beating the lookup table baseline of 3.56%."

---

### [2:10 - 2:40] Code Walkthrough & December Sanity Check
* **Visual:** Show `scorer_results/candidate_december.png`.
* **Speaking Points:**
  > "The codebase is organized as a modular production package under `src/` with 20 passing unit and integration tests.
  >
  > Running `make_submission.py` trains the production blend on all clean historical data, validates output schemas, and generates `validation_predictions.csv` alongside `december_chart_inputs.csv`.
  >
  > For our fixed Lexington to Fort Wayne dry van lane, December predictions average $847—tracking the historical October benchmark of $842. The weekly wave peaks on Thursdays and dips on weekends, exactly mirroring the weekly spot market cycle.
  >
  > Thank you for your time, and I look forward to your questions!"
