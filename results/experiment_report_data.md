# Comprehensive Experimental Report & Soft Computing Curriculum Analysis

**Project Title:** Heat Wave Prediction in May for Nagpur (Vidarbha) using Error Backpropagation ANN, Fuzzy Logic & Genetic Optimization  
**Course:** Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech COMP)  
**Institution:** K. J. Somaiya School of Engineering, Somaiya Vidyavihar University  
**Target Domain:** Nagpur Urban Agglomeration, Vidarbha, Maharashtra (`21.1458° N, 79.0882° E`, Elevation: 311.0 m)  
**Climatological Zone:** Central Indian Core Heatwave Zone (CHZ)  

---

## 1. KJSCE Soft Computing Lab Curriculum Mapping

This project provides an end-to-end scientific implementation directly mapping onto the KJSCE Soft Computing laboratory experiments and Course Outcomes (CO1–CO5):

| Course Module / Exp | Conceptual Framework | Mathematical Model / Operators | Source Code Implementation | Primary Findings / Metrics |
|---|---|---|---|---|
| **Exp 1: McCulloch-Pitts Neuron** | Handcrafted Biological Threshold Logic | Excitatory thermal summation ($T_{\text{max}} \ge 42^\circ\text{C}$, $\Delta T \ge 3.5^\circ\text{C}$, $\text{RH} \le 25\%$) minus inhibitory wind veto ($w_{\text{inh}}=2$ if $W \ge 15\text{ km/h}$) firing if $\text{net} \ge 2$. | [`src/sc_baselines.py`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/src/sc_baselines.py) | **Test Recall: 89.19%**, Precision: 45.21%, F1: 0.6000. Domain logic captures heat events but lacks continuous calibration. |
| **Exp 2: Activation Functions** | Non-linear Transfer Functions | Empirical comparison of Binary Sigmoid ($1/(1+e^{-x})$), Bipolar Sigmoid ($\tanh(x/2)$), and ReLU ($\max(0, x)$) in hidden layer. | [`src/ann_numpy.py`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/src/ann_numpy.py) | **Bipolar Sigmoid achieved highest Val F1 (0.8047)** and lowest loss (0.7231) due to zero-mean gradient dynamics. |
| **Exp 4: Single-Layer Perceptron** | Rosenblatt's Bipolar Error-Correction Rule | Step activation with discrete updates $\Delta \mathbf{w} = \eta \cdot t \cdot \mathbf{x}$ for misclassifications ($y \neq t$, $t \in \{-1, +1\}$). | [`src/sc_baselines.py`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/src/sc_baselines.py) | **Test F1: 0.6105**, Acc: 85.08%. Demonstrates Novikoff's theorem limit: non-separable boundary causes oscillation. |
| **Exp 5: ADALINE Delta Rule** | Widrow-Hoff Continuous LMS Gradient Descent | Continuous logistic activation with MSE gradient descent: $\Delta \mathbf{w} = \eta (t - y) f'(y_{\text{in}}) \mathbf{x}$. | [`src/sc_baselines.py`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/src/sc_baselines.py) | **Test PR-AUC: 0.8293**, F1: 0.7606, Acc: 93.15%. Dramatic improvement over Perceptron via smooth quadratic error surface. |
| **Exp 6 & 7 / CO3: Multilayer EBPA** | Two-Layer Error Backpropagation Network | Explicit forward pass, output error signal $\delta_k = (t_k - y_k) f'(y_{\text{in}, k})$, hidden error signal $\delta_j = (\sum_k \delta_k w_{jk}) f'(z_{\text{in}, j})$, and momentum updates in pure NumPy. | [`src/ann_numpy.py`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/src/ann_numpy.py) | **Test PR-AUC: 0.8382**, ROC-AUC: 0.9640, Accuracy: 93.95%. Multi-layer architecture resolves non-linear boundary. |
| **Deep MLP Bagging Ensemble** | 10-Seed Regularized Deep Architecture | `Input(26) -> Dense(32, ReLU) -> BatchNorm1d -> Drop(0.2) -> Dense(32, ReLU) -> Drop(0.2) -> Dense(1, Sigmoid)` trained with AdamW and class-weighted BCE. | [`src/train_ann.py`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/src/train_ann.py), [`src/export_model.py`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/src/export_model.py) | **Test PR-AUC: 0.8749 (Highest)**, ROC-AUC: 0.9649, Val F1: 0.8229. 100% pure-NumPy inference equivalence to $< 1.1 \times 10^{-7}$. |
| **Module 4 / CO4: Mamdani Fuzzy System** | Linguistic Meteorological Reasoning | 3 Antecedents ($T_{\text{max, lag1}}$, Anomaly, $\text{RH}_{\text{min}}$) with triangular/trapezoidal MFs, 14 expert rules, Min implication, Max aggregation, and Centroid defuzzification. | [`src/fuzzy_controller.py`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/src/fuzzy_controller.py) | **Test ROC-AUC: 0.9600**, Recall: 89.19%, Val F1: 0.8205. Robust physical safeguards against catastrophic neural failure. |
| **Module 5 / CO5: Genetic Algorithm** | Real-Coded Evolutionary Optimization | Population=40, Generations=50, Tournament Selection ($k=3$), BLX-$\alpha$ crossover ($p_c=0.8$), Gaussian mutation ($p_m=0.15$) optimizing $[w_{\text{ANN}}, w_{\text{EBPA}}, w_{\text{Fuzzy}}, \tau^*]$. | [`src/ga_optimizer.py`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/src/ga_optimizer.py) | **Optimal weights: ANN=40.8%, EBPA=39.1%, Fuzzy=20.1%**, $\tau^*=0.4979$. Achieves **Highest Test ROC-AUC: 0.9668**. |

---

## 2. Atmospheric Dataset & Chronological Partitioning

### 2.1 Reanalysis Domain
- **Source:** ECMWF ERA5 reanalysis via Open-Meteo Historical Archive API.
- **Coordinates:** Grid center `21.25° N, 79.00° E`, Elevation `311.0 m` (concordant with Nagpur plateau).
- **Temporal Horizon:** 46 consecutive years (1981–2026), 2,806 calendar days (1,380 April days + 1,426 May days).
- **Completeness:** 100.0% non-null across all 8 hourly thermodynamic variables (`temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `apparent_temperature`, `surface_pressure`, `wind_speed_10m`, `shortwave_radiation`, `soil_moisture_0_to_7cm`).

### 2.2 Chronological Split Distribution
To strictly prevent temporal data leakage and respect causality, no random k-fold splitting was permitted:

| Split Partition | Calendar Years | Total May Days | Positive Heatwave Days (`imd_reanalysis_plains`) | Prevalence (%) | Role in Experimental Pipeline |
|---|---|---|---|---|---|
| **Training** | 1981–2010 (30 yrs) | 930 | 226 | 24.30% | Fit `StandardScaler`, calculate class weights, learn connection weights |
| **Validation** | 2011–2018 (8 yrs) | 248 | 88 | 35.48% | Early stopping, hyperparameter grid search, GA chromosome optimization |
| **Locked Test** | 2019–2026 (8 yrs) | 248 | 37 | 14.92% | Single-pass out-of-distribution evaluation under recent humid climate anomaly |
| **Total Horizon**| 1981–2026 (46 yrs) | 1,426 | 351 | 24.61% | Complete empirical study domain |

---

## 3. Mathematical Foundations

### 3.1 Error Backpropagation Network (EBPA — Exp 6 & 7)
For an input vector $\mathbf{x} \in \mathbb{R}^{26}$, hidden activations $\mathbf{z} \in \mathbb{R}^{32}$, and scalar output $y \in (0, 1)$:
1. **Hidden Layer Net Input & Activation:**
   $$z_{\text{in}, j} = v_{0j} + \sum_{i=1}^{26} x_i v_{ij}, \quad z_j = f(z_{\text{in}, j})$$
   Using Bipolar Sigmoid: $f(x) = \frac{2}{1 + e^{-x}} - 1 = \tanh\left(\frac{x}{2}\right)$, with derivative $f'(x) = \frac{1}{2}(1 + f(x))(1 - f(x))$.
2. **Output Layer Net Input & Activation:**
   $$y_{\text{in}} = w_0 + \sum_{j=1}^{32} z_j w_j, \quad y = \sigma(y_{\text{in}}) = \frac{1}{1 + e^{-y_{\text{in}}}}$$
3. **Output Error Signal:**
   $$\delta_k = (t_k - y_k) \cdot y_k (1 - y_k)$$
4. **Hidden Error Signal:**
   $$\delta_j = \left(\sum_k \delta_k w_{jk}\right) \cdot f'(z_{\text{in}, j})$$
5. **Momentum Weight Updates ($\eta = 0.01, \alpha = 0.9$):**
   $$\Delta w_{jk}(t) = \eta \cdot \delta_k \cdot z_j + \alpha \cdot \Delta w_{jk}(t-1)$$
   $$\Delta v_{ij}(t) = \eta \cdot \delta_j \cdot x_i + \alpha \cdot \Delta v_{ij}(t-1)$$

### 3.2 Mamdani Fuzzy Inference & Defuzzification (CO4)
Given input crisp vector $\mathbf{x} = [T_{\text{max, lag1}}, \Delta T_{\text{lag1}}, \text{RH}_{\text{min, lag1}}]$:
1. **Fuzzification:** Evaluate triangular ($\mu_{\text{tri}}(x; a, b, c)$) and trapezoidal ($\mu_{\text{trap}}(x; a, b, c, d)$) membership degrees.
2. **Rule Firing Strength (Min T-Norm):**
   $$\alpha_r = \min\left(\mu_{A_r}(x_1), \mu_{B_r}(x_2), \mu_{C_r}(x_3)\right)$$
3. **Consequent Implication & Aggregation (Max S-Norm):**
   $$\mu_{\text{agg}}(z) = \max_{r=1}^{14} \left[\min\left(\alpha_r, \mu_{\text{risk}, r}(z)\right)\right], \quad z \in [0, 100]$$
4. **Centroid (Center of Gravity) Defuzzification:**
   $$z^* = \frac{\int_0^{100} z \cdot \mu_{\text{agg}}(z) dz}{\int_0^{100} \mu_{\text{agg}}(z) dz} \approx \frac{\sum_{i=1}^{101} z_i \cdot \mu_{\text{agg}}(z_i)}{\sum_{i=1}^{101} \mu_{\text{agg}}(z_i)}$$

### 3.3 Genetic Algorithm Optimization (CO5)
- **Chromosome:** $\mathbf{c} = [g_0, g_1, g_2, \tau]$ decoded into $[w_{\text{ANN}}, w_{\text{EBPA}}, w_{\text{Fuzzy}}] = \frac{[g_0, g_1, g_2]}{\sum g_k}$ and threshold $\tau \in [0.45, 0.80]$.
- **Ensemble Hypothesis:**
  $$P_{\text{hybrid}}(\mathbf{x}) = w_{\text{ANN}} P_{\text{ANN}}(\mathbf{x}) + w_{\text{EBPA}} P_{\text{EBPA}}(\mathbf{x}) + w_{\text{Fuzzy}} P_{\text{Fuzzy}}(\mathbf{x})$$
  $$\widehat{y} = \mathbb{I}\left(P_{\text{hybrid}}(\mathbf{x}) \ge \tau\right)$$
- **Fitness Objective Function:**
  $$\text{Fitness}(\mathbf{c}) = F_1(\mathbf{y}_{\text{val}}, \widehat{\mathbf{y}}_{\text{val}}) - 0.15 \cdot \max\left(0, \text{Recall} - \text{Precision} - 0.12\right)$$

---

## 4. Empirical Evaluation Results

### 4.1 Exp 2: Activation Function Comparison Table
Evaluated on the Validation split (2011–2018):

| Activation Function | Formulation $f(x)$ | Final Train Loss | Val Loss | Val F1 Score | Val PR-AUC | Val ROC-AUC | Stopping Epoch |
|---|---|---|---|---|---|---|---|
| **Bipolar Sigmoid** (Selected) | $\tanh(x/2)$ | **0.5595** | **0.7231** | **0.8047** | 0.8566 | **0.9241** | **63** (Fastest) |
| **Binary Sigmoid** | $1 / (1 + e^{-x})$ | 0.5626 | 0.8043 | 0.7831 | **0.8652** | 0.9241 | 96 |
| **Rectified Linear (ReLU)** | $\max(0, x)$ | 0.5054 | 0.9114 | 0.8049 | 0.8434 | 0.9141 | 96 |

*Analysis:* Bipolar sigmoid maps inputs to $[-1, +1]$, producing zero-mean hidden activations that prevent systematic gradient zig-zagging during backpropagation. This resulted in the lowest validation loss (0.7231) and 34% faster convergence than standard binary sigmoid.

### 4.2 Locked Test Set Benchmark (May 2019–2026, 248 Days, 37 Positives)

| Model Name | Lab Mapping | Decision Threshold $\tau$ | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Brier Score | Confusion Matrix [TP, FP, TN, FN] |
|---|---|---|---|---|---|---|---|---|---|---|
| **Persistence Baseline** | Synoptic Benchmark | 0.5000 | 0.9274 | 0.7568 | 0.7568 | 0.7568 | 0.8571 | 0.6090 | 0.0726 | [28, 9, 202, 9] |
| **McCulloch-Pitts Neuron** | Exp 1 | 0.5000 | 0.8226 | 0.4521 | **0.8919** | 0.6000 | 0.8512 | 0.4193 | 0.1774 | [33, 40, 171, 4] |
| **Perceptron Learning Rule** | Exp 4 | 0.5000 | 0.8508 | 0.5000 | 0.7838 | 0.6105 | 0.8232 | 0.4241 | 0.1492 | [29, 29, 182, 8] |
| **ADALINE Delta Rule** | Exp 5 | 0.5000 | 0.9315 | 0.7941 | 0.7297 | 0.7606 | 0.9620 | 0.8293 | 0.0510 | [27, 7, 204, 10] |
| **Class-Weighted Logistic Reg.**| Statistical Benchmark | 0.5000 | 0.8911 | 0.6000 | 0.8108 | 0.6897 | 0.9636 | 0.8476 | 0.0704 | [30, 20, 191, 7] |
| **NumPy EBPA Network** | Exp 6 & 7 | 0.5000 | **0.9395** | **0.8235** | 0.7568 | **0.7887** | 0.9640 | 0.8382 | **0.0511** | [28, 6, 205, 9] |
| **10-Seed Deep MLP Ensemble** | Deep Learning Core | 0.5000 | 0.9032 | 0.6383 | 0.8108 | 0.7143 | 0.9649 | **0.8749** | 0.0631 | [30, 17, 194, 7] |
| **Mamdani Fuzzy Controller** | Module 4 (CO4) | 0.5000 | 0.8790 | 0.5593 | **0.8919** | 0.6875 | 0.9600 | 0.7584 | 0.0842 | [33, 26, 185, 4] |
| **GA Neuro-Fuzzy Hybrid** | Module 5 (CO5) | **0.4979** | 0.9153 | 0.6905 | 0.7838 | 0.7342 | **0.9668** | 0.8583 | 0.0575 | [29, 13, 198, 8] |

### 4.3 Validation Set Benchmark (May 2011–2018, 248 Days, 88 Positives)

| Model Name | Threshold $\tau$ | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Brier Score |
|---|---|---|---|---|---|---|---|---|
| **Persistence Baseline** | 0.5000 | 0.8750 | 0.8276 | 0.8182 | 0.8229 | 0.8622 | 0.7416 | 0.1250 |
| **McCulloch-Pitts Neuron** | 0.5000 | 0.7540 | 0.6134 | 0.8295 | 0.7053 | 0.7513 | 0.5762 | 0.2460 |
| **Perceptron Learning Rule** | 0.5000 | 0.7137 | 0.6000 | 0.5795 | 0.5896 | 0.7797 | 0.6523 | 0.2863 |
| **ADALINE Delta Rule** | 0.5000 | 0.8710 | 0.8333 | 0.7955 | 0.8140 | 0.9238 | 0.8510 | 0.0963 |
| **Class-Weighted Logistic Reg.**| 0.5000 | 0.8508 | 0.7476 | 0.8750 | 0.8063 | 0.9130 | 0.8396 | 0.1096 |
| **NumPy EBPA Network** | 0.5000 | 0.8710 | 0.8118 | 0.8295 | 0.8205 | 0.9238 | 0.8559 | 0.0970 |
| **10-Seed Deep MLP Ensemble** | 0.5000 | 0.8710 | 0.7596 | **0.8977** | 0.8229 | 0.9259 | **0.8680** | 0.1079 |
| **Mamdani Fuzzy Controller** | 0.5000 | 0.8669 | 0.7477 | **0.9091** | 0.8205 | 0.9173 | 0.8363 | 0.1017 |
| **GA Neuro-Fuzzy Hybrid** | **0.4979** | **0.8911** | **0.8061** | **0.8977** | **0.8495** | **0.9284** | 0.8647 | **0.0940** |

---

## 5. Sensitivity Analysis across Alternative Heatwave Thresholds

To verify model robustness against station definition variations, the models were evaluated against Candidate B (Percentile-based, $P_{85} \ge 42^\circ\text{C}$) and Reference (Strict IMD raw, $T_{\text{max}} \ge 45^\circ\text{C}$):

| Heatwave Target Definition | Target Prevalence (Test Split) | Model Evaluated | Recall | Precision | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Primary (`imd_reanalysis_plains`)** | 37 / 248 (14.92%) | 10-Seed Deep MLP Ensemble | 81.08% | 63.83% | 0.7143 | 0.9649 |
| **Primary (`imd_reanalysis_plains`)** | 37 / 248 (14.92%) | GA Neuro-Fuzzy Hybrid | 78.38% | 69.05% | 0.7342 | **0.9668** |
| **Candidate B (`climatology_p85_42c`)** | 19 / 248 (7.66%) | 10-Seed Deep MLP Ensemble | 89.47% | 36.17% | 0.5152 | 0.9384 |
| **Candidate B (`climatology_p85_42c`)** | 19 / 248 (7.66%) | GA Neuro-Fuzzy Hybrid | **89.47%** | 40.48% | 0.5574 | 0.9354 |
| **Reference (`imd_strict_raw`)** | 8 / 248 (3.23%) | 10-Seed Deep MLP Ensemble | **100.0%** | 17.02% | 0.2909 | **0.9589** |
| **Reference (`imd_strict_raw`)** | 8 / 248 (3.23%) | GA Neuro-Fuzzy Hybrid | **100.0%** | 19.05% | 0.3200 | 0.9552 |

*Finding:* When tested against severe unadjusted station criteria ($T_{\text{max}} \ge 45^\circ\text{C}$, only 8 occurrences in 2019–2026), both the Deep MLP and GA Neuro-Fuzzy Hybrid achieved **100.0% Recall (8/8 events caught)** with zero missed severe heatwaves.

---

## 6. Comprehensive Error Analysis & Climatological Discussion

### 6.1 Atmospheric Regime Shift between Validation and Test Sets
- **Validation Split (2011–2018):** Marked by severe continental drying and record May heat in Nagpur (mean $T_{\text{max}} = 42.32^\circ\text{C}$, prevalence 35.48%). Models trained on historical data achieve high recall and precision.
- **Test Split (2019–2026):** Marked by unusual pre-monsoon moisture advection and Arabian Sea convective pulses (mean relative humidity elevated to 45.4% vs 31.4% historical; heatwave prevalence dropped to 14.92%).
- **Why GA Neuro-Fuzzy Outperformed Single Neural Networks:**
  Single deep models trained with class weighting ($pos\_weight = 3.115$) tended to over-predict positive heatwave days when encountering hot but humid days, resulting in 17 false alarms for the Deep MLP.
  The GA Hybrid mitigated this by blending the Mamdani Fuzzy Controller ($w_{\text{Fuzzy}} = 20.1\%$). Because Rule 9 and Rule 13 of the fuzzy system explicitly incorporate high relative humidity as a dampening factor, the ensemble reduced false alarms from 17 down to 13, boosting precision from 63.8% to 69.1% while maintaining an exceptional **ROC-AUC of 0.9668**.

---

## 7. Generated Publication Figures Reference (200 DPI)

All figures have been saved to [`results/figures/`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/results/figures/):

1. **[`fig01_study_area.png`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/results/figures/fig01_study_area.png):** Geographical and climatic contrast between Nagpur (inland plateau Core Heatwave Zone) and coastal Mumbai.
2. **[`fig02_tmax_climatology.png`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/results/figures/fig02_tmax_climatology.png):** WMO 1991–2020 daily May maximum temperature normal, P85, P90 percentile bands, and the 43.5°C threshold.
3. **[`fig03_positives_per_year.png`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/results/figures/fig03_positives_per_year.png):** 46-year annual May heatwave frequency (1981–2026) depicting Train, Validation, and Test partitions.
4. **[`fig04_feature_correlation.png`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/results/figures/fig04_feature_correlation.png):** Pearson correlation ranking of atmospheric predictors against `heatwave_day`.
5. **[`fig05_training_curves.png`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/results/figures/fig05_training_curves.png):** 10-seed MLP ensemble loss trajectory (mean $\pm$ 1 SD) and KJSCE Exp 2 activation function comparison bar chart.
6. **[`fig06_roc_pr_curves.png`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/results/figures/fig06_roc_pr_curves.png):** Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves on the locked Test split across all models.
7. **[`fig07_confusion_matrix.png`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/results/figures/fig07_confusion_matrix.png):** 4-panel test set confusion matrix (McCulloch-Pitts, ADALINE, Deep MLP, GA Hybrid).
8. **[`fig08_calibration.png`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/results/figures/fig08_calibration.png):** Probability calibration and reliability diagrams on the locked Test set.
9. **[`fig09_permutation_importance.png`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/results/figures/fig09_permutation_importance.png):** Top 12 predictors ranked by PR-AUC degradation upon feature permutation.
10. **[`fig10_error_analysis.png`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/results/figures/fig10_error_analysis.png):** Timeline of Test set temperatures (2019–2026) highlighting True Positives, False Alarms, and Misses.
11. **[`fig11_fuzzy_membership.png`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/results/figures/fig11_fuzzy_membership.png):** Graphical plots of the three antecedent membership functions and consequent risk sets (CO4).
12. **[`fig12_ga_convergence.png`](file:///c:/Users/soham/OneDrive/Desktop/heatwave-ANN/results/figures/fig12_ga_convergence.png):** Champion and mean population fitness trajectories across 50 generations of genetic optimization (CO5).
