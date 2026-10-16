# Results Summary & Report Handoff (`RESULTS_SUMMARY.md`)

**Problem Statement:** *"Design and develop an artificial neural network to predict heat wave in May month considering climate conditions of a specific demographic area."*

**Course:** Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP), K. J. Somaiya School of Engineering

---

## 1. Experimental Setup & Data Audit Summary

* **Study Area:** Nagpur Urban Agglomeration, Vidarbha, Maharashtra (`21.1458° N, 79.0882° E`).
* **Grid Cell Returned:** ECMWF ERA5 (`0.25°` resolution) grid center `21.25° N, 79.00° E`, elevation `311.0 m`, timezone `Asia/Kolkata` (IST).
* **Data Horizon:** 46 years (`1981–2026`), 1 April to 31 May each year = `2,806` total calendar days (`1,380` April days used strictly for lag/rolling window initialization + `1,426` May target days).
* **Daily Variables (11 total, 0.0% missing):** `tmax`, `tmin`, `tmean`, `apparent_tmax`, `rh_mean`, `rh_min`, `dewpoint_mean`, `wind_mean`, `pressure_mean`, `solar_sum`, `soil_moisture_mean`.

---

## 2. Heatwave Label Definitions & Split Counts

Climatological baseline is the WMO `1991–2020` 30-year reference period using a centered $\pm 7$-calendar-day window (`May 1` normal $T_{\max} = 41.54^\circ\text{C} \rightarrow \text{May 31}$ normal $T_{\max} = 42.44^\circ\text{C}$).

| Split | Years | May Days | Primary (`imd_reanalysis_plains`) | Candidate B (`climatology_p85_42c`) | Reference (`imd_strict_raw`) |
| --- | --- | --- | --- | --- | --- |
| **Train** | 1981–2010 | 930 | **226 (24.30%)** | 139 (14.95%) | 55 (5.91%) |
| **Validation** | 2011–2018 | 248 | **88 (35.48%)** | 51 (20.56%) | 23 (9.27%) |
| **Locked Test** | 2019–2026 | 248 | **37 (14.92%)** | 19 (7.66%) | 8 (3.23%) |
| **Total** | 1981–2026 | 1,426 | **351 (24.61%)** | 209 (14.66%) | 86 (6.03%) |

---

## 3. Leakage-Safe Feature Set (26 Features)

All features for predicting May day $T$ are computed strictly from days $\le T-1$ plus deterministic WMO climatology for day $T$:

* **Lag 1 ($T-1$):** `tmax_lag1`, `tmin_lag1`, `tmean_lag1`, `rh_mean_lag1`, `rh_min_lag1`, `dewpoint_mean_lag1`, `apparent_tmax_lag1`, `wind_mean_lag1`, `pressure_mean_lag1`, `solar_sum_lag1`, `soil_moisture_lag1`
* **Lag 2 & Lag 3 ($T-2, T-3$):** `tmax_lag2`, `tmax_lag3`, `apparent_tmax_lag2`, `apparent_tmax_lag3`
* **Rolling Accumulations:** `tmax_roll3` (mean $T-3..T-1$), `tmax_roll7` (mean $T-7..T-1$), `solar_roll3` (mean $T-3..T-1$)
* **Trends, Anomalies & Persistence:** `tmax_trend3` ($T_{\max, T-1} - T_{\max, T-3}$), `tmax_anomaly_lag1` ($T_{\max, T-1} - T_{\text{norm}, T-1}$), `dtr_lag1` ($T_{\max, T-1} - T_{\min, T-1}$), `pressure_tendency` ($P_{T-1} - P_{T-3}$), `hot_streak_len` (consecutive heatwave days ending at $T-1$)
* **Calendar & Climatology:** `normal_tmax_T`, `day_sin`, `day_cos`

---

## 4. Final ANN Architecture & Hyperparameters

1. **Primary 10-Seed Deep MLP ANN Ensemble (`src/train_ann.py`):**
* **Winning Grid Search Architecture:** `Input(26) -> Linear(32) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(32) -> ReLU -> Dropout(0.2) -> Linear(1) -> Sigmoid`
* **Loss & Optimizer:** Class-weighted Binary Cross-Entropy (`pos_weight = 3.115` from Train), AdamW (`lr = 0.003`, `weight_decay = 1e-4`), EarlyStopping on Validation PR-AUC (`patience = 25`), retrained across 10 seeds (`42..51`).

2. **From-Scratch NumPy Error Backpropagation Network (`src/ann_numpy.py`):**
* **Architecture:** `Input(26) -> Hidden(32, Bipolar Sigmoid) -> Output(1, Binary Sigmoid)` with momentum ($\eta = 0.01, \alpha = 0.9$).

---

## 5. Headline Validation & Locked Test Set Metrics

### A. Locked Test Set (2019–2026, 248 Days, 37 Positives, No-Skill PR-AUC = 0.1492)

| Model | Category / Syllabus | Threshold | Accuracy | Precision | Recall | F1 Score | PR-AUC | ROC-AUC | Brier | Confusion [TP, FP, TN, FN] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Always-Negative (Base Rate)** | Climatology Baseline | — | 0.8508 | 0.0000 | 0.0000 | 0.0000 | 0.1492 | 0.5000 | 0.1492 | [0, 0, 211, 37] |
| **Persistence (`T-1`)** | Synoptic Baseline | 0.5000 | 0.9274 | 0.7568 | 0.7568 | 0.7568 | 0.6090 | 0.8571 | 0.0726 | [28, 9, 202, 9] |
| **McCulloch-Pitts Neuron** | KJSCE Exp 1 | 0.5000 | 0.8226 | 0.4521 | **0.8919** | 0.6000 | 0.4193 | 0.8512 | 0.1774 | [33, 40, 171, 4] |
| **Perceptron Learning Rule** | KJSCE Exp 4 | 0.5000 | 0.8508 | 0.5000 | 0.7838 | 0.6105 | 0.4241 | 0.8232 | 0.1492 | [29, 29, 182, 8] |
| **ADALINE (Delta Rule)** | KJSCE Exp 5 | 0.5000 | 0.9315 | 0.7941 | 0.7297 | 0.7606 | 0.8293 | 0.9620 | **0.0510** | [27, 7, 204, 10] |
| **Logistic Regression** | Statistical Baseline | 0.5000 | 0.8911 | 0.6000 | 0.8108 | 0.6897 | 0.8476 | 0.9636 | 0.0704 | [30, 20, 191, 7] |
| **Pure-NumPy EBPA ANN** | **KJSCE Exp 2, 6, 7** | 0.5000 | **0.9395** | **0.8235** | 0.7568 | **0.7887** | 0.8382 | 0.9640 | 0.0511 | [28, 6, 205, 9] |
| **10-Seed Deep MLP ANN** | **Primary ANN (CO3)** | 0.5000 | 0.9032 | 0.6383 | 0.8108 | 0.7143 | **0.8749** | 0.9649 | 0.0631 | [30, 17, 194, 7] |
| **Mamdani Fuzzy System** | Module 4 (CO4) | 0.5000 | 0.8790 | 0.5593 | **0.8919** | 0.6875 | 0.7584 | 0.9600 | 0.0842 | [33, 26, 185, 4] |
| **Neuro-Fuzzy Hybrid** | ANN + Fuzzy Blend | 0.4979 | 0.9153 | 0.6905 | 0.7838 | 0.7342 | 0.8583 | **0.9668** | 0.0575 | [29, 13, 198, 8] |

---

## 6. Figure Index (`results/figures/`)

1. `fig01_study_area.png`: Meteorological comparison of Nagpur (Inland CHZ) vs. Coastal Mumbai justifying study area selection.
2. `fig02_tmax_climatology.png`: WMO 1991–2020 May daily normal $T_{\max}$, P85, P90, and 43.5°C threshold progression.
3. `fig03_positives_per_year.png`: Annual May heatwave counts (1981–2026) across Train, Validation, and Test splits.
4. `fig04_feature_correlation.png`: Pearson correlation of all 26 lagged predictors with `heatwave_day`.
5. `fig05_training_curves.png`: 10-seed ANN training/validation loss curves and Exp 2 activation study comparison.
6. `fig06_roc_pr_curves.png`: Test set ROC and Precision-Recall curves across ANN and baselines.
7. `fig07_confusion_matrix.png`: Test set confusion matrices for McCulloch-Pitts, ADALINE, Deep MLP ANN, and Neuro-Fuzzy.
8. `fig08_calibration.png`: Reliability diagram on the locked Test set.
9. `fig09_permutation_importance.png`: Validation PR-AUC drop permutation importance ranking top 12 features.
10. `fig10_error_analysis.png`: Locked Test set timeline (2019–2026) showing True Positives, False Alarms, and Misses.
11. `fig11_fuzzy_membership.png`: Mamdani Fuzzy membership functions and defuzzification sets.
12. `fig12_ga_convergence.png`: Evolutionary convergence of ensemble blending weights on Validation set.
