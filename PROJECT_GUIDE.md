# Project Guide: May Heat Wave Prediction in Nagpur Using an Artificial Neural Network

**Course:** Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech Computer Engineering)  
**Institution:** K. J. Somaiya School of Engineering, Somaiya Vidyavihar University  
**Problem Statement:** *"Design and develop an artificial neural network to predict heat wave in May month considering climate conditions of a specific demographic area."*

---

## 1. Intuitive Overview: What Does This Project Do?

### Why We Chose Nagpur (Vidarbha) Instead of Coastal Mumbai
A neural network learns patterns from historical examples. In coastal Mumbai, the Arabian Sea acts as a natural thermal buffer, keeping May daytime temperatures around 33°C–35°C and producing almost zero official India Meteorological Department (IMD) heatwaves. 

**Nagpur (`21.1458° N, 79.0882° E`, elevation 311 m)** sits on the inland Deccan plateau right inside India's **Core Heatwave Zone (CHZ)**. Every May, dry continental winds and intense solar radiation drive Nagpur's daytime highs between **42°C and 46.9°C**. Out of 1,426 May days between 1981 and 2026, Nagpur recorded **351 genuine heatwave days (24.6%)**, giving our Artificial Neural Network a rich, balanced meteorological dataset.

### The Prediction Task (100% Leakage-Safe — Section 2.4)
Each row in our dataset (`data/processed/features.csv`) represents a single calendar day $T$ in May (May 1 to May 31 across 46 years = **1,426 rows**).
At the end of day $T-1$ (yesterday night), our Artificial Neural Network answers:
> *"Given all weather observations recorded up to the end of day $T-1$, plus the known 30-year climatological normal temperature for calendar day $T$, will day $T$ be a heatwave day in Nagpur?"*

- **Zero Future Peeking:** Features for day $T$ use **zero** weather measurements from day $T$ or later.
- **Role of April Data:** We pull April 1–30 every year (`data/processed/daily.csv`, 2,806 total days) solely to compute the 3-day and 7-day rolling averages and temperature trends needed by early May days (May 1–7).
- **Automated Verification (`tests/test_leakage.py`):** Our test suite perturbs all weather columns on day $T$ by $+50.0^\circ\text{C}$ and verifies that the engineered feature vector for day $T$ remains bitwise identical.

---

## 2. Step-by-Step Pipeline Architecture

```
[Open-Meteo ERA5 Reanalysis Archive (1981–2026, Hourly, Asia/Kolkata)]
│
▼
[src/daily.py: Hourly-to-Daily IST Aggregation]
2,806 days (April & May), 11 daily variables, 0 missing values
│
▼
[src/climatology.py & src/labels.py: WMO 1991–2020 Normals]
Computes 30-year normal Tmax (±7-day window) & IMD heatwave labels
│
▼
[src/features.py: 26 Leakage-Safe Predictors (Strictly <= Day T-1)]
Lag-1 weather, Lag-2/3 Tmax, 3 & 7-day rolling means, 3-day warming trend,
climatological anomaly, diurnal range (DTR), pressure tendency, hot streak
│
▼
[src/split.py: Strictly Chronological Partitioning]
├── Train (1981–2010):      930 May days (24.3% HW) -> Fit StandardScaler & Weights
├── Validation (2011–2018): 248 May days (35.5% HW) -> Early Stopping & Threshold
└── Locked Test (2019–2026):248 May days (14.9% HW) -> Final Single-Pass Evaluation
│
▼
[Artificial Neural Network Core & Soft Computing Lab Suite]
├── Baselines: Climatology, Persistence (T-1), Logistic Regression, Random Forest
├── Lab Neurons: McCulloch-Pitts (Exp 1), Perceptron (Exp 4), ADALINE Delta Rule (Exp 5)
├── From-Scratch NumPy Error Backpropagation Network (Exp 2, 6, 7 - src/ann_numpy.py)
├── Primary Deliverable: 10-Seed Deep MLP ANN Ensemble (src/train_ann.py)
└── Soft-Computing Contrast: Mamdani Fuzzy Inference Controller (src/fuzzy_controller.py)
│
▼
[src/export_model.py & app/streamlit_app.py]
Pure-NumPy weights export (models/ensemble_weights.npz) -> 4-Tab Interactive Web App
```

---

## 3. Accordance with the KJSCE Soft Computing Syllabus

Every module in this repository directly implements a topic from the **KJSCE Soft Computing Theory & Lab Syllabus**:

1. **Experiment 1 — McCulloch–Pitts Threshold Neuron (`src/sc_baselines.py`):**
   Models a biological neuron with 3 excitatory binary inputs ($T_{\max, T-1} \ge 42^\circ\text{C}$, Anomaly $\ge +3.5^\circ\text{C}$, $\text{RH}_{\min, T-1} \le 25\%$) and 1 inhibitory input (Wind $\ge 15\text{ km/h}$ with inhibitory weight $w_{\text{inh}} = 2$). Fires ($y=1$) when net input $\ge \theta = 2$. Shows high recall (**89.19%**) but low precision (**45.21%**) because hand-crafted binary thresholds cannot adapt to subtle multi-variable interactions.
2. **Experiment 2 — Activation Functions Study (`src/ann_numpy.py`):**
   Trains the exact same 2-layer backpropagation network using **Binary Sigmoid** ($1/(1+e^{-x})$), **Bipolar Sigmoid** ($\tanh(x/2)$), and **ReLU** ($\max(0, x)$). **Bipolar Sigmoid** achieves the best validation loss (**0.7231**) and fastest convergence (**63 epochs**) because its zero-centered $(-1, 1)$ outputs prevent gradient zig-zagging.
3. **Experiment 4 — Perceptron Learning Rule (`src/sc_baselines.py`):**
   Implements Rosenblatt's supervised step-activation rule with bipolar targets $t \in \{-1, +1\}$ and update $\Delta \mathbf{w} = \eta \cdot t \cdot \mathbf{x}$ on misclassified samples. Demonstrates **Novikoff's Theorem / Minsky-Papert linear separability limit**: because atmospheric heatwaves are not linearly separable, discrete step updates oscillate (**Test F1: 0.6105**).
4. **Experiment 5 — Continuous Delta Learning Rule / ADALINE (`src/sc_baselines.py`):**
   Replaces the discrete step with continuous Widrow-Hoff Least Mean Squares (LMS) gradient descent ($\Delta \mathbf{w} = \eta (t - y) f'(y_{\text{in}}) \mathbf{x}$). The smooth quadratic error surface jumps performance to **Test F1: 0.7606** and **ROC-AUC: 0.9620**.
5. **Experiments 6 & 7 (CO3) — Multilayer Perceptron & Error Backpropagation (`src/ann_numpy.py` & `src/train_ann.py`):**
   - **Pure-NumPy EBPA (`src/ann_numpy.py`):** Explicitly codes the forward pass, output layer error $\delta_k = (t_k - y_k) f'(y_{\text{in}, k})$, hidden layer backpropagated error $\delta_j = (\sum_k \delta_k w_{jk}) f'(z_{\text{in}, j})$, and momentum updates ($\alpha = 0.9$). Achieves **93.95% Test Accuracy** and **0.7887 Test F1**.
   - **10-Seed Deep MLP ANN Ensemble (`src/train_ann.py`):** Our main deliverable (`26 -> 32 -> 32 -> 1` with BatchNorm, Dropout 0.2, AdamW, and class-weighted Binary Cross-Entropy) selected via 5-seed validation grid search and retrained across 10 seeds (`42..51`). Achieves the **highest Test PR-AUC (0.8749)** and **ROC-AUC (0.9649)**, catching **100% (8/8)** of raw $\ge 45^\circ\text{C}$ station heatwaves.
6. **Module 4 (CO4) & Module 5 (CO5) — Fuzzy Logic & Evolutionary Blending (`src/fuzzy_controller.py`, `src/ga_optimizer.py`):**
   Provides a 14-rule Mamdani Fuzzy Inference System (with triangular/trapezoidal membership functions and centroid defuzzification) and a Neuro-Fuzzy blend (`ROC-AUC: 0.9668`) as a soft-computing comparison against the standalone ANN.

---

## 4. Key Questions Examiners Ask in the Viva (And Your Answers)

1. **"Why did Bipolar Sigmoid beat Binary Sigmoid in your Exp 2 Activation Study?"**
   * **Answer:** Binary sigmoid outputs values strictly in $(0, 1)$, meaning all hidden layer outputs $z_j$ are positive. During backpropagation ($\Delta w_{jk} = \eta \delta_k z_j$), all weight updates for a neuron must have the same sign, causing the optimizer to zig-zag slowly. **Bipolar sigmoid ($\tanh(x/2) \in (-1, 1)$)** is zero-centered, allowing positive and negative hidden activations, which converged in **63 epochs** (vs. 96 for binary sigmoid) and achieved the lowest validation loss (**0.7231**) and highest validation F1 (**0.8047**).
2. **"How does your Streamlit web app run the Deep ANN without installing PyTorch or TensorFlow on the cloud?"**
   * **Answer:** After training our 10-seed ANN in PyTorch, `src/export_model.py` extracts the learned weight matrices ($W_1, b_1, W_2, b_2, W_3, b_3$) and Batch Normalization statistics ($\gamma, \beta, \mu, \sigma^2$) into a compressed NumPy archive (`models/ensemble_weights.npz`). In `NumpyMLPEnsemble`, we wrote the matrix multiplication, BatchNorm scaling, ReLU (`np.maximum(0, x)`), and Sigmoid (`1 / (1 + np.exp(-x))`) directly in NumPy. Our verification test proves the NumPy output matches PyTorch to within $1.1 \times 10^{-7}$.
3. **"Why was the Test set (2019–2026) positive rate lower (14.9%) than Validation (2011–2018, 35.5%)?"**
   * **Answer:** As documented in our data audit (`results/data_audit.md`), 2011–2018 was an unusually dry, scorching decade in Vidarbha (mean May RH ~31%), whereas 2021–2026 experienced frequent pre-monsoon thunderstorms and moisture incursion from the Arabian Sea (mean May RH jumped to 45.4%). Even under this real-world climate shift, our ANN achieved **0.9649 ROC-AUC** and **0.8749 PR-AUC** (nearly $6\times$ better than the 0.149 no-skill baseline).
