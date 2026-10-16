# Heat Wave Prediction in May for Nagpur (Vidarbha) using Error Backpropagation ANN, Fuzzy Logic & Genetic Optimization

Audience: the coding agent (Antigravity) and the student team.
Status: authoritative. If your instincts disagree with this file, follow the file and log the disagreement in `DECISIONS.md`.

---

## 0. Operating rules for the agent

1. Read this whole file before planning. Start in Planning Mode and produce a task list and implementation plan that maps onto the phases in Section 9. Do not skip the plan.
2. Make judgment calls inside these rules instead of stopping to ask. Log every non-trivial call in `DECISIONS.md` (what, why, alternatives considered).
3. **GRANULAR GIT COMMIT & PUSH RULE (CRITICAL):** Do NOT wait until the end of a phase to commit. You MUST make a small, atomic git commit (using Conventional Commits, e.g., `feat(data): ...`, `feat(sc): ...`, `test(leakage): ...`, `docs(audit): ...`) **immediately after creating or updating each individual module, feature, test, or figure**, and push to `origin main` (if remote is configured). The repository must show a realistic, step-by-step history of 25+ granular commits.
4. There is exactly one mandatory stop: the end of Phase 1 (data audit). Everything else runs through autonomously.
5. Never fabricate numbers, results, citations, or screenshots. Every figure and table in `results/` must be produced by code in this repo.
6. Never tune anything on the test years. The test set is evaluated only once after all models, hyperparameters, and thresholds are frozen using validation data.
7. No API keys or accounts are needed anywhere in this project.
8. Verify the Streamlit app in the browser: launch it, click through every tab, and save screenshots to `results/app_verification/`.
9. Write clean, well-commented, viva-friendly code explicitly referencing the KJSCE Soft Computing Lab concepts (McCulloch-Pitts, Perceptron, Delta Rule, Activation Functions, Error Backpropagation delta equations, Mamdani Fuzzy Inference, and Genetic Algorithm operators).

---

## 1. Project context & Syllabus Alignment

**Problem statement (verbatim from the course):** "Design and develop an artificial neural network to predict heat wave in May month considering climate conditions of a specific demographic area."

**Course:** Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP), K. J. Somaiya School of Engineering.

**Chosen demographic area:** **Nagpur Urban Agglomeration (Vidarbha region, Central Maharashtra)** — coordinates: **latitude 21.1458, longitude 79.0882** (elevation ~310m, inland plateau). Nagpur lies in India's Core Heatwave Zone (CHZ), regularly recording May maxima of 43°C to 47.5°C and providing frequent, genuine heatwave events.

**KJSCE Syllabus & Lab Alignment (Must be reflected in code comments and `RESULTS_SUMMARY.md`):**
- **Exp 1 (McCulloch-Pitts Neuron):** Excitatory/inhibitory meteorological threshold logic neuron (`src/sc_baselines.py`).
- **Exp 2 (Activation Functions):** Empirical comparison of Binary Sigmoid, Bipolar Sigmoid (`tanh`), and ReLU/Ramp in the hidden layer (`src/ann_numpy.py`).
- **Exp 4 & 5 (Perceptron & Delta Learning Rule / ADALINE):** Single-layer supervised learning rules demonstrating linear separability limits vs MLP (`src/sc_baselines.py`).
- **Exp 6 & 7 / CO3 (MLP & Error Backpropagation Network - EBPA):** Explicit forward pass, output error $\delta_k$, hidden layer backpropagated error $\delta_j$, and weight updates (`src/ann_numpy.py` & `src/train_ann.py`).
- **Module 4 / CO4 (Fuzzy Logic Controller & Neuro-Fuzzy System):** Mamdani fuzzy inference system with triangular/trapezoidal membership functions and centroid defuzzification (`src/fuzzy_controller.py`).
- **Module 5 / CO5 (Genetic Algorithm):** GA optimization of Neuro-Fuzzy ensemble weights and decision threshold (`src/ga_optimizer.py`).

### 1.1 Submission structure (fixed by the course)

MiniProject<TEAM_LEADER_ROLL_NUMBER>/        e.g. MiniProject12321218
Report/        report (max 15 pages) + PPT (5 to 6 slides: implementation, results, analysis), will complete this later after project is done
Code/          everything in this repo


---

## 2. Scientific design

### 2.1 Data source

Primary: **Open-Meteo Historical Weather API** (`https://archive-api.open-meteo.com/v1/archive`). Free, keyless, hourly ERA5-Land (0.1 degree, ~11 km) and ERA5 (0.25 degree) reanalysis.

- Request ERA5-Land (`models=era5_land` or best match) for `latitude=21.1458, longitude=79.0882`, `timezone=Asia/Kolkata`. Record the exact grid latitude/longitude and elevation returned in `config.yaml` and `results/data_audit.md`.
- Probe hourly variables (`temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `apparent_temperature`, `surface_pressure`, `wind_speed_10m`, `shortwave_radiation`, `soil_moisture_0_to_7cm`). If any variable is missing in ERA5-Land, fetch from ERA5 and log it in `DECISIONS.md`.
- Period: pull **1 April to 31 May for every year 1981 to 2026 (46 years)**. April days are used exclusively to compute lag and 7-day rolling features for early May (May 1–7).
- Cache raw responses to `data/raw/` so reruns never re-hit the API.

### 2.2 Daily variables

Aggregate local calendar days (`Asia/Kolkata`) from hourly into:
`tmax`, `tmin`, `tmean`, `rh_mean`, `rh_min`, `dewpoint_mean`, `apparent_tmax`, `wind_mean`, `pressure_mean`, `solar_sum`, `soil_moisture_mean` (if available).

### 2.3 Label definition (Nagpur Interior Plains Heat Wave Criteria)

Evaluate three candidate definitions across all May days (1981–2026), using WMO 1991–2020 (±7 calendar-day window) as the baseline climatology:

1. **`imd_reanalysis_plains` (Candidate A):** `tmax >= 43.5°C` (reanalysis-calibrated equivalent of 45°C station threshold) OR (`tmax >= 40.0°C` AND `tmax - normal_tmax_wmo >= 3.5°C`).
2. **`climatology_p85_42c` (Candidate B):** `tmax >= max(42.0°C, P85_calendar_day_1991_2020)`.
3. **`imd_strict_raw` (Reference):** Unadjusted station rule `tmax >= 45.0°C` OR (`tmax >= 40.0°C` AND `tmax - normal_tmax_wmo >= 4.5°C`).

**Selection Rule in Phase 1:** Automatically select whichever of Candidate A or Candidate B yields a **Training split (1981–2010) positive rate between 15% and 35%** as the primary `heatwave_day` target. If both are slightly outside, pick the one closest to 22% positive rate. Report counts for all three in `results/data_audit.md` and run sensitivity experiments on the alternatives in Phase 4.

### 2.4 Prediction task (100% leakage-safe)

One row per May day `T` (1 to 31 May, 1981–2026 = 1,426 May rows).
Question: *"Given atmospheric conditions observed up to the end of day `T-1`, will day `T` in Nagpur be a heat wave day?"*

- Features may use days up to `T-1` (including late April for May 1–7) plus deterministic calendar/climatology normals for day `T`.
- Features must NEVER use any observed weather value from day `T` or later.
- Write an automated test (`tests/test_leakage.py`) that perturbs all weather columns on day `T` by `+50.0` and verifies that the computed feature vector for day `T` is bitwise identical.

### 2.5 Features (~24 engineered features)

- **Lag 1 (day `T-1`):** `tmax_lag1`, `tmin_lag1`, `tmean_lag1`, `rh_mean_lag1`, `rh_min_lag1`, `dewpoint_mean_lag1`, `apparent_tmax_lag1`, `wind_mean_lag1`, `pressure_mean_lag1`, `solar_sum_lag1`, `soil_moisture_lag1` (if available)
- **Lag 2 and Lag 3:** `tmax_lag2`, `tmax_lag3`, `apparent_tmax_lag2`, `apparent_tmax_lag3`
- **Multi-day thermal build-up:** `tmax_roll3` (mean `T-3..T-1`), `tmax_roll7` (mean `T-7..T-1`), `solar_roll3` (mean `T-3..T-1`)
- **Trends & Anomalies:**
  - `tmax_trend3` = `tmax(T-1) - tmax(T-3)`
  - `tmax_anomaly_lag1` = `tmax(T-1) - normal_tmax(T-1)`
  - `dtr_lag1` = `tmax(T-1) - tmin(T-1)`
  - `pressure_tendency` = `pressure(T-1) - pressure(T-3)`
  - `hot_streak_len` = consecutive days up to `T-1` where `heatwave_day == 1`
- **Calendar & Climatology:** `normal_tmax_T`, `day_sin`, `day_cos`

Standardize continuous features with `StandardScaler` fit **strictly on the Training split (1981–2010)**.

### 2.6 Chronological Split (Never Random)

| Split | Years | May Rows | Purpose |
|---|---|---|---|
| **Train** | 1981 to 2010 (30 yrs) | 930 | Fit scaler, class weights, and neural weights |
| **Validation** | 2011 to 2018 (8 yrs) | 248 | Early stopping, hyperparameter selection, GA threshold tuning |
| **Test** | 2019 to 2026 (8 yrs) | 248 | Final locked evaluation only |

---

## 3. Models (Full KJSCE Soft Computing Suite)

### 3.1 Classical & Soft Computing Baselines (`src/sc_baselines.py` & `src/baselines.py`)
1. **Persistence Baseline:** Predict `heatwave_day(T) = heatwave_day(T-1)`.
2. **Exp 1 — McCulloch-Pitts Threshold Logic Neuron (`McCullochPittsNeuron`):** Unweighted/weighted binary excitatory inputs (`tmax_lag1 >= thresh`, `tmax_anomaly_lag1 >= 3.5`, `rh_min_lag1 <= 25`) and inhibitory input (`wind_mean_lag1 >= 15`), firing when net input $\ge \theta$.
3. **Exp 4 — Single-Layer Perceptron (`PerceptronRuleClassifier`):** Bipolar targets $\{-1, +1\}$, step activation, weight update $\Delta w = \eta \cdot t \cdot x$ when $y \neq t$.
4. **Exp 5 — Continuous Delta Learning Rule / ADALINE (`DeltaRuleADALINE`):** Supervised LMS gradient descent on linear/sigmoid single neuron ($\Delta w = \eta (t - y_{\text{in}}) f'(y_{\text{in}}) x$), demonstrating single-layer linear separability limits (Exp 6).
5. **Statistical Benchmark:** Class-weighted Logistic Regression.

### 3.2 Exp 2, 6 & 7 — From-Scratch Error Backpropagation Network (`src/ann_numpy.py`)
Implement a clean, well-commented 2-layer Error Backpropagation Network (EBPA) in **100% pure NumPy**:
- Architecture: `Input(n_features) -> Hidden(32) -> Output(1, Binary Sigmoid)`.
- **Exp 2 Activation Study:** Support switching the hidden layer activation between:
  1. **Binary Sigmoid:** $f(x) = \frac{1}{1 + e^{-x}}$, derivative $f'(x) = f(x)(1 - f(x))$
  2. **Bipolar Sigmoid:** $f(x) = \frac{2}{1 + e^{-x}} - 1 = \tanh(x/2)$, derivative $f'(x) = \frac{1}{2}(1 + f(x))(1 - f(x))$
  3. **ReLU (Ramp variant):** $f(x) = \max(0, x)$, derivative $f'(x) = \mathbb{I}(x > 0)$
- **Exp 7 Explicit Backprop Math:** Implement and comment the exact textbook equations:
  - Forward pass: $z_{\text{in}, j} = v_{0j} + \sum_i x_i v_{ij}$, $z_j = f(z_{\text{in}, j})$, $y_{\text{in}, k} = w_{0k} + \sum_j z_j w_{jk}$, $y_k = \sigma(y_{\text{in}, k})$.
  - Output error signal $\delta_k$ and hidden error signal $\delta_j = (\sum_k \delta_k w_{jk}) f'(z_{\text{in}, j})$.
  - Weight updates with learning rate $\eta$ and momentum $\alpha$: $\Delta w_{jk}(t) = \eta \delta_k z_j + \alpha \Delta w_{jk}(t-1)$.
- Train and log a comparison table (`results/tables/exp2_activation_comparison.csv`) comparing Binary Sigmoid, Bipolar Sigmoid, and ReLU on Validation PR-AUC and F1.

### 3.3 Primary Deep ANN Deliverable: 10-Seed Regularized MLP Ensemble (`src/train_ann.py`)
Implement using **PyTorch** (CPU, fast install) or **Keras** (with pure-NumPy/scikit-learn fallback if C++ wheel fails):
Input(n_features) -> Dense(units1, ReLU) -> BatchNorm -> Dropout(dropout) -> Dense(units2, ReLU) -> Dropout(dropout) -> Dense(1, Sigmoid)

- Class-weighted Binary Cross-Entropy loss (`pos_weight` from Train), AdamW optimizer, EarlyStopping on Validation PR-AUC / Loss (patience = 25, max 300 epochs).
- Grid search on Validation across 5 seeds: `units1` in `{32, 64}`, `units2` in `{16, 32}`, `dropout` in `{0.2, 0.3}`, `lr` in `{1e-3, 3e-3}`.
- Retrain winning architecture across **10 seeds (`42..51`)**. Export all 10 seed weights into `models/ensemble_weights.npz` and `models/scaler.json` so the Streamlit app runs 100% pure-NumPy inference without needing PyTorch/TensorFlow installed on the cloud server.

### 3.4 Module 4 (CO4) — Mamdani Fuzzy Logic Controller & Neuro-Fuzzy Hybrid (`src/fuzzy_controller.py`)
Implement a Mamdani Fuzzy Inference System (using `scikit-fuzzy` or clean NumPy trapezoidal/triangular membership functions + centroid defuzzification):
- **Antecedents (Fuzzy Sets):**
  1. `tmax_lag1`: *Normal* ($<40^\circ\text{C}$), *Hot* ($39\text{–}43.5^\circ\text{C}$), *Extreme* ($>42.5^\circ\text{C}$)
  2. `tmax_anomaly_lag1`: *Below/Normal* ($<1.5^\circ\text{C}$), *Elevated* ($1.0\text{–}4.0^\circ\text{C}$), *Severe Departure* ($>3.5^\circ\text{C}$)
  3. `rh_min_lag1`: *Arid/Dry* ($<18\%$), *Moderate* ($15\text{–}35\%$), *Humid* ($>30\%$)
- **Consequent:** `fuzzy_risk_score` $\in [0, 100]$ with linguistic sets *Low*, *Watch*, *Heatwave Warning* (defuzzified via Centroid method and scaled to $[0, 1]$).

### 3.5 Module 5 (CO5) — Genetic Algorithm Neuro-Fuzzy Optimizer (`src/ga_optimizer.py`)
Implement a clean Genetic Algorithm in NumPy to optimize the **Neuro-Fuzzy Hybrid** on the Validation set:
- **Chromosome Encoding (Real-valued vector of length 4):** $[w_{\text{ANN}}, w_{\text{EBPA}}, w_{\text{Fuzzy}}, \tau_{\text{threshold}}]$ where weights sum to 1 and $\tau \in [0.25, 0.75]$.
- **GA Operators:** Population size = 40, Generations = 50, Fitness function = Validation F1-score, Tournament Selection ($k=3$), BLX-$\alpha$ / Uniform Crossover ($p_c = 0.8$), Gaussian Mutation ($p_m = 0.15$).
- Save the evolved optimal weights and decision threshold $\tau^*$ to `models/threshold.json` and plot the GA fitness convergence curve across generations (`fig12_ga_convergence.png`).

---

## 4. Evaluation and Required Outputs

Metrics on Validation and Test: **PR-AUC, ROC-AUC, F1 (at frozen threshold $\tau^*$ and at 0.5), Precision, Recall, Brier Score, Confusion Matrix**.

Required figures saved at 200 DPI in `results/figures/`:
- `fig01_study_area.png`: Map/schematic of Nagpur (Vidarbha) grid cell & climate comparison vs coastal Mumbai.
- `fig02_tmax_climatology.png`: Nagpur May `tmax` seasonal curve (1991–2020 normal, P85/P90 band, and heatwave threshold).
- `fig03_positives_per_year.png`: Heat wave days per May (1981–2026) with Train/Val/Test split boundaries.
- `fig04_feature_correlation.png`: Feature correlation heatmap with `heatwave_day`.
- `fig05_training_curves.png`: Training vs Validation loss curves across the 10 ANN seeds (mean ± 1 SD) + Exp 2 Activation Function curves (Binary Sigmoid vs Bipolar Sigmoid vs ReLU).
- `fig06_roc_pr_curves.png`: ROC and PR curves comparing McCulloch-Pitts, Delta Rule (ADALINE), Mamdani Fuzzy, NumPy EBPA, 10-Seed Deep ANN, and GA-Optimized Neuro-Fuzzy Hybrid.
- `fig07_confusion_matrix.png`: Test set confusion matrices.
- `fig08_calibration.png`: Reliability / calibration diagram.
- `fig09_permutation_importance.png`: Top 12 features by validation permutation importance.
- `fig10_error_analysis.png`: Test-set timeline (2019–2026 May temperatures) showing True Positives, False Alarms, and Misses.
- `fig11_fuzzy_membership.png`: Plots of the Mamdani Fuzzy membership functions and defuzzification surface (CO4).
- `fig12_ga_convergence.png`: Genetic Algorithm best & mean population fitness across 50 generations (CO5).

---

## 5. Repository Layout (Git Root = `Code/`)

README.md
DECISIONS.md
PROJECT_SPEC.md
requirements.txt
requirements-app.txt         lightweight dependencies for Streamlit Cloud (no torch/tf)
config.yaml
data/raw/                    cached API responses
data/processed/              daily.csv, features.csv
src/
fetch_data.py
daily.py
climatology.py
labels.py
features.py
split.py
sc_baselines.py            Exp 1 (McCulloch-Pitts), Exp 4 (Perceptron), Exp 5 (Delta Rule/ADALINE)
baselines.py               Persistence & Logistic Regression
ann_numpy.py               Exp 2, 6, 7: From-scratch Error Backpropagation Network & activation study
train_ann.py               10-seed Deep MLP Ensemble
fuzzy_controller.py        Module 4 (CO4): Mamdani Fuzzy Inference System
ga_optimizer.py            Module 5 (CO5): Genetic Algorithm threshold & neuro-fuzzy weight optimizer
export_model.py            Exports weights to models/ensemble_weights.npz
evaluate.py                Generates all tables and figures fig01..fig12
pipeline.py                One command: python -m src.pipeline
notebooks/
01_eda.ipynb
02_modeling.ipynb
models/
results/
figures/  tables/  app_verification/  RESULTS_SUMMARY.md  data_audit.md
app/streamlit_app.py
tests/                       test_leakage.py, test_split.py, test_labels.py, test_scaler.py
scripts/package_submission.py


---

## 6. Demo App (`app/streamlit_app.py` — 4 Tabs)

1. **Tab 1 — Historical May Replay (1981–2026):** Choose any year and May date in Nagpur. Show Deep ANN probability, Mamdani Fuzzy risk score, GA-Neuro-Fuzzy decision, actual outcome, and a 10-day `tmax` chart.
2. **Tab 2 — Interactive What-If Simulator:** Move sliders for `tmax_lag1`, `tmin_lag1`, `rh_min_lag1`, `wind_mean_lag1`, and `hot_streak_len`. Recompute engineered features live and inspect how the McCulloch-Pitts neuron, Error Backpropagation ANN, and Fuzzy Membership rules respond.
3. **Tab 3 — KJSCE Soft Computing Lab Explorer (Exp 1–7 & CO1–CO5):** Interactive comparison table and plots covering Exp 1 (M-P Neuron), Exp 2 (Sigmoid vs Bipolar Sigmoid vs ReLU), Exp 4–5 (Perceptron & Delta Rule), Exp 7 (EBPA Backprop), CO4 (Fuzzy Membership plots), and CO5 (GA convergence).
4. **Tab 4 — Mathematical Walkthrough & Viva Guide:** Displays the exact forward/backward propagation equations, delta rule updates, fuzzy rule matrix, and dataset audit summary.

---

## 7. Phases & Execution Protocol

- **Phase 0 (Scaffold):** Layout, `config.yaml`, `requirements.txt`, `DECISIONS.md`, CLI `python -m src.pipeline --help`. Commit & push each piece.
- **Phase 1 (Data Fetch & Audit — MANDATORY STOP):** Fetch 1981–2026 April–May data for Nagpur (`21.1458, 79.0882`), build `daily.csv`, evaluate the 3 label candidates, select the primary label, and write `results/data_audit.md`. Commit & push after each module. **STOP and summarize `results/data_audit.md` for the user.**
- **Phase 2 (Features, Split & Unit Tests):** Build `features.csv`, chronological split, and pass all `pytest` tests (`test_leakage.py`, `test_split.py`, `test_labels.py`, `test_scaler.py`). Commit & push each file.
- **Phase 3 (Soft Computing Baselines — Exp 1, 4, 5):** Implement and evaluate Persistence, McCulloch-Pitts, Perceptron, Delta Rule (ADALINE), and Logistic Regression. Commit & push.
- **Phase 4 (Core Soft Computing — EBPA, Deep ANN, Fuzzy Logic, Genetic Algorithm):** Train `ann_numpy.py` (with Exp 2 activation study), 10-seed `train_ann.py`, `fuzzy_controller.py`, and `ga_optimizer.py`. Export `.npz` weights and run locked Test evaluation (`evaluate.py`). Commit & push after each model module.
- **Phase 5 (Figures & `RESULTS_SUMMARY.md`):** Generate `fig01`–`fig12`, all CSV/Markdown tables, and `results/RESULTS_SUMMARY.md`. Commit & push.
- **Phase 6 (Streamlit App & Browser Verification):** Build `app/streamlit_app.py`, launch in browser, click through all 4 tabs, and save screenshots to `results/app_verification/`. Commit & push.
- **Phase 7 (Packaging & Final Docs):** Finalize `README.md`, `notebooks/`, and `scripts/package_submission.py --roll <ROLL>`. Commit & push all final artifacts.
