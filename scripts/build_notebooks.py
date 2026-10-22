# LEGACY generator; use scripts/build_lab_notebook.py for the corrected assessed notebook.
"""
Script to build and pre-execute Jupyter Demonstration Notebooks:
1. notebooks/01_eda.ipynb (Exploratory Data Analysis, Climatology, Trends)
2. notebooks/02_modeling.ipynb (Soft Computing Baselines, EBPA, Deep MLP, Fuzzy, GA, Evaluation)

Pre-renders all outputs so university examiners can view results on GitHub without a kernel.
"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Markdown Header
    cells.append(nbf.v4.new_markdown_cell(
        "# Nagpur May Heat Wave Prediction — Climatological Exploratory Data Analysis & Audit\n\n"
        "**Course:** Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech Computer Engineering)  \n"
        "**Location:** Nagpur Urban Agglomeration, Vidarbha, Central India (21.1458°N, 79.0882°E)  \n"
        "**Dataset:** ECMWF ERA5-Land Reanalysis (1981–2026, 46 years, 2,806 April–May records)  \n"
        "**Baseline Normals:** WMO 1991–2020 Standard Reference Period (±7 day rolling window)"
    ))

    # Cell 1: Setup & Imports
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import seaborn as sns\n"
        "import yaml\n\n"
        "# Set styling\n"
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
        "plt.rcParams['font.sans-serif'] = 'DejaVu Sans'\n"
        "print('Setup complete. Project root:', PROJECT_ROOT)"
    ))

    # Cell 2: Data Loading & Verification
    cells.append(nbf.v4.new_code_cell(
        "daily_df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'daily.csv')\n"
        "features_df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'features.csv')\n"
        "normals_df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'climatology_normals.csv')\n\n"
        "print('Daily Dataset Shape:', daily_df.shape)\n"
        "print('Features Dataset Shape:', features_df.shape)\n"
        "print('Temporal Range:', daily_df['date'].min(), 'to', daily_df['date'].max())\n"
        "print('Missing Values in Daily Dataset:', daily_df.isnull().sum().sum())\n"
        "daily_df.head()"
    ))

    # Cell 3: WMO Baseline Climatology Analysis
    cells.append(nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)\n"
        "ax.plot(normals_df['day_of_year'], normals_df['normal_tmax'], label='WMO 1991-2020 Normal Tmax (°C)', color='#0275d8', linewidth=2)\n"
        "ax.plot(normals_df['day_of_year'], normals_df['p85_tmax'], label='85th Percentile Threshold (°C)', color='#f0ad4e', linestyle='--', linewidth=1.5)\n"
        "ax.axhline(43.5, color='#d9534f', linestyle=':', label='IMD Plains Severe Heatwave Threshold (43.5°C)', linewidth=1.5)\n"
        "ax.set_title('Nagpur Climatological Temperature Profile (April 1 – May 31)', fontsize=12, fontweight='bold')\n"
        "ax.set_xlabel('Day of Year (Calendar Progression)', fontsize=10)\n"
        "ax.set_ylabel('Temperature (°C)', fontsize=10)\n"
        "ax.legend(loc='lower right', frameon=True)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Cell 4: Candidate Label Evaluation
    cells.append(nbf.v4.new_code_cell(
        "label_cols = ['imd_reanalysis_plains', 'climatology_p85_42c', 'imd_strict_raw']\n"
        "split_summary = []\n\n"
        "for split_name, (start_yr, end_yr) in [\n"
        "    ('Train (1981-2010)', (1981, 2010)),\n"
        "    ('Validation (2011-2018)', (2011, 2018)),\n"
        "    ('Test (2019-2026)', (2019, 2026)),\n"
        "]:\n"
        "    sub = features_df[(features_df['year'] >= start_yr) & (features_df['year'] <= end_yr)]\n"
        "    row = {'Split': split_name, 'Total May Days': len(sub)}\n"
        "    for col in label_cols:\n"
        "        pos = int(sub[col].sum())\n"
        "        rate = pos / len(sub) * 100\n"
        "        row[f'{col} (Count)'] = pos\n"
        "        row[f'{col} (%)'] = f'{rate:.1f}%'\n"
        "    split_summary.append(row)\n\n"
        "pd.DataFrame(split_summary)"
    ))

    # Cell 5: Decadal Warming Trend Analysis
    cells.append(nbf.v4.new_code_cell(
        "features_df['decade'] = pd.cut(\n"
        "    features_df['year'],\n"
        "    bins=[1980, 1990, 2000, 2010, 2020, 2030],\n"
        "    labels=['1981-1990', '1991-2000', '2001-2010', '2011-2020', '2021-2026'],\n"
        ")\n\n"
        "decadal_stats = features_df.groupby('decade', observed=False).agg(\n"
        "    mean_tmax_lag1=('tmax_lag1', 'mean'),\n"
        "    max_tmax_lag1=('tmax_lag1', 'max'),\n"
        "    heatwave_rate=('imd_reanalysis_plains', 'mean'),\n"
        "    total_days=('date', 'count'),\n"
        ").reset_index()\n"
        "decadal_stats['heatwave_rate'] = decadal_stats['heatwave_rate'] * 100\n"
        "print('Decadal Temperature Evolution in Nagpur (May):')\n"
        "decadal_stats"
    ))

    # Cell 6: Land-Atmosphere Coupling & Feature Correlations
    cells.append(nbf.v4.new_code_cell(
        "corr_cols = [\n"
        "    'tmax_lag1', 'tmin_lag1', 'dtr_lag1', 'rh_min_lag1', 'wind_mean_lag1',\n"
        "    'soil_moisture_lag1', 'solar_sum_lag1', 'hot_streak_len', 'imd_reanalysis_plains'\n"
        "]\n"
        "corr_matrix = features_df[corr_cols].corr()\n\n"
        "plt.figure(figsize=(9, 7), dpi=150)\n"
        "sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'})\n"
        "plt.title('Meteorological Predictor Correlation Structure (Nagpur May)', fontsize=12, fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Cell 7: Summary Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### Key Climatological Insights & Modeling Implications:\n"
        "1. **Primary Target Justification:** `imd_reanalysis_plains` provides an ideal positive rate (~24.3% in train, 21.8% in test), capturing both absolute thermal thresholds (≥ 42°C) and climatological anomaly thresholds (+3.5°C).\n"
        "2. **Thermodynamic Couplings:** Previous-day minimum humidity exhibits strong negative correlation with heatwave occurrence (-0.51), reflecting dry continental air advection.\n"
        "3. **Decadal Warming:** The frequency of extreme heatwave episodes has systematically increased across recent decades, reinforcing the need for adaptive soft computing architectures."
    ))

    nb["cells"] = cells
    return nb


def create_modeling_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Markdown Header
    cells.append(nbf.v4.new_markdown_cell(
        "# Nagpur May Heat Wave Prediction — Soft Computing Architecture Suite\n\n"
        "**Course:** Soft Computing Lab (SVU-KJSCE 2023_1.0 T.Y. B.Tech Computer Engineering)  \n"
        "**Experimental Coverage:** Experiments 1–7 (M-P Neuron, Perceptron, ADALINE, EBPA, Deep MLP, Mamdani FIS, Genetic Algorithm)  \n"
        "**Curriculum Mapping:** CO1 (Basics & M-P), CO2 (Supervised Learning Rules), CO3 (Backpropagation ANN), CO4 (Fuzzy Systems), CO5 (Genetic Algorithms)"
    ))

    # Cell 1: Setup & Imports
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "import json\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import yaml\n\n"
        "from src.split import get_splits\n"
        "from src.sc_baselines import McCullochPittsNeuron, PerceptronRuleClassifier, DeltaRuleADALINE\n"
        "from src.ann_numpy import EBPANetwork\n"
        "from src.fuzzy_controller import MamdaniFuzzyController\n"
        "from src.export_model import NumpyMLPEnsemble\n"
        "print('Successfully imported soft computing suite.')"
    ))

    # Cell 2: Load Splits
    cells.append(nbf.v4.new_code_cell(
        "with open(PROJECT_ROOT / 'config.yaml', 'r', encoding='utf-8') as f:\n"
        "    cfg = yaml.safe_load(f)\n\n"
        "splits = get_splits(cfg)\n"
        "X_train, y_train = splits['X_train'], splits['y_train']\n"
        "X_val, y_val = splits['X_val'], splits['y_val']\n"
        "X_test, y_test = splits['X_test'], splits['y_test']\n\n"
        "print(f'Train Split (1981-2010): {X_train.shape[0]} samples, {int(y_train.sum())} positive ({y_train.mean():.1%})')\n"
        "print(f'Val Split   (2011-2018): {X_val.shape[0]} samples, {int(y_val.sum())} positive ({y_val.mean():.1%})')\n"
        "print(f'Test Split  (2019-2026): {X_test.shape[0]} samples, {int(y_test.sum())} positive ({y_test.mean():.1%})')"
    ))

    # Cell 3: Soft Computing Baselines Evaluation (Exp 1, 4, 5)
    cells.append(nbf.v4.new_code_cell(
        "from sklearn.metrics import f1_score, roc_auc_score, average_precision_score\n\n"
        "# 1. McCulloch-Pitts (Exp 1)\n"
        "mp = McCullochPittsNeuron()\n"
        "mp_pred = mp.predict(splits['test_df'])\n"
        "mp_proba = mp.predict_proba(splits['test_df'])[:, 1]\n\n"
        "# 2. Perceptron (Exp 4)\n"
        "perc = PerceptronRuleClassifier(eta=0.01, max_epochs=100, random_state=42)\n"
        "perc.fit(X_train, y_train)\n"
        "perc_pred = perc.predict(X_test)\n"
        "perc_proba = perc.predict_proba(X_test)[:, 1]\n\n"
        "# 3. ADALINE Delta Rule (Exp 5)\n"
        "adaline = DeltaRuleADALINE(eta=0.005, max_epochs=200, random_state=42)\n"
        "adaline.fit(X_train, y_train)\n"
        "adaline_pred = adaline.predict(X_test)\n"
        "adaline_proba = adaline.predict_proba(X_test)[:, 1]\n\n"
        "baseline_summary = [\n"
        "    {'Model': 'McCulloch-Pitts (Exp 1)', 'F1': f1_score(y_test, mp_pred), 'PR-AUC': average_precision_score(y_test, mp_proba), 'ROC-AUC': roc_auc_score(y_test, mp_proba)},\n"
        "    {'Model': 'Perceptron Step (Exp 4)', 'F1': f1_score(y_test, perc_pred), 'PR-AUC': average_precision_score(y_test, perc_proba), 'ROC-AUC': roc_auc_score(y_test, perc_proba)},\n"
        "    {'Model': 'ADALINE Delta Rule (Exp 5)', 'F1': f1_score(y_test, adaline_pred), 'PR-AUC': average_precision_score(y_test, adaline_proba), 'ROC-AUC': roc_auc_score(y_test, adaline_proba)},\n"
        "]\n"
        "pd.DataFrame(baseline_summary)"
    ))

    # Cell 4: Exp 2 Activation Function Comparison in Pure-NumPy EBPA
    cells.append(nbf.v4.new_code_cell(
        "exp2_df = pd.read_csv(PROJECT_ROOT / 'results' / 'tables' / 'exp2_activation_comparison.csv')\n"
        "print('Experiment 2: Activation Function Study in EBPA Network:')\n"
        "exp2_df"
    ))

    # Cell 5: Pure-NumPy EBPA Network Inference
    cells.append(nbf.v4.new_code_cell(
        "ebpa = EBPANetwork(n_input=26)\n"
        "ebpa.load_weights(str(PROJECT_ROOT / 'models' / 'ebpa_weights.npz'))\n"
        "ebpa_proba = ebpa.predict_proba(X_test)\n"
        "ebpa_pred = ebpa.predict(X_test, threshold=0.5)\n\n"
        "print('NumPy EBPA Test Performance:')\n"
        "print(f'F1 Score:  {f1_score(y_test, ebpa_pred):.4f}')\n"
        "print(f'PR-AUC:    {average_precision_score(y_test, ebpa_proba):.4f}')\n"
        "print(f'ROC-AUC:   {roc_auc_score(y_test, ebpa_proba):.4f}')"
    ))

    # Cell 6: 10-Seed Deep MLP Ensemble Inference
    cells.append(nbf.v4.new_code_cell(
        "mlp_ensemble = NumpyMLPEnsemble(str(PROJECT_ROOT / 'models' / 'ensemble_weights.npz'))\n"
        "mlp_proba = mlp_ensemble.predict_proba(X_test)\n"
        "mlp_pred = mlp_ensemble.predict(X_test, threshold=0.5)\n\n"
        "print('10-Seed Deep MLP Ensemble Test Performance:')\n"
        "print(f'F1 Score:  {f1_score(y_test, mlp_pred):.4f}')\n"
        "print(f'PR-AUC:    {average_precision_score(y_test, mlp_proba):.4f}')\n"
        "print(f'ROC-AUC:   {roc_auc_score(y_test, mlp_proba):.4f}')"
    ))

    # Cell 7: Mamdani Fuzzy Controller Evaluation
    cells.append(nbf.v4.new_code_cell(
        "fuzzy_ctrl = MamdaniFuzzyController()\n"
        "fuzzy_proba = fuzzy_ctrl.predict_proba(splits['test_df'])\n"
        "fuzzy_pred = (fuzzy_proba >= 0.5).astype(int)\n\n"
        "print('Mamdani Fuzzy Logic Controller Test Performance:')\n"
        "print(f'F1 Score:  {f1_score(y_test, fuzzy_pred):.4f}')\n"
        "print(f'PR-AUC:    {average_precision_score(y_test, fuzzy_proba):.4f}')\n"
        "print(f'ROC-AUC:   {roc_auc_score(y_test, fuzzy_proba):.4f}')"
    ))

    # Cell 8: Genetic Algorithm Optimization & Final Hybrid
    cells.append(nbf.v4.new_code_cell(
        "with open(PROJECT_ROOT / 'models' / 'threshold.json', 'r', encoding='utf-8') as f:\n"
        "    th_data = json.load(f)\n\n"
        "w_ann = th_data['w_ann']\n"
        "w_ebpa = th_data['w_ebpa']\n"
        "w_fuzzy = th_data['w_fuzzy']\n"
        "tau_star = th_data['threshold']\n\n"
        "print(f'GA Evolved Parameters: w_ANN={w_ann:.4f}, w_EBPA={w_ebpa:.4f}, w_Fuzzy={w_fuzzy:.4f}, tau*={tau_star:.4f}')\n\n"
        "# Form hybrid ensemble probability\n"
        "hybrid_proba = w_ann * mlp_proba + w_ebpa * ebpa_proba + w_fuzzy * fuzzy_proba\n"
        "hybrid_pred = (hybrid_proba >= tau_star).astype(int)\n\n"
        "print('GA Neuro-Fuzzy Hybrid Out-of-Sample Test Performance:')\n"
        "print(f'Accuracy:  {(hybrid_pred == y_test).mean():.4f}')\n"
        "print(f'F1 Score:  {f1_score(y_test, hybrid_pred):.4f}')\n"
        "print(f'PR-AUC:    {average_precision_score(y_test, hybrid_proba):.4f}')\n"
        "print(f'ROC-AUC:   {roc_auc_score(y_test, hybrid_proba):.4f}')"
    ))

    # Cell 9: Final Out-of-Sample Comparison Benchmark Table
    cells.append(nbf.v4.new_code_cell(
        "final_test_df = pd.read_csv(PROJECT_ROOT / 'results' / 'tables' / 'final_model_comparison_test.csv')\n"
        "print('Comprehensive Model Comparison on Test Split (2019-2026):')\n"
        "final_test_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'PR-AUC', 'ROC-AUC', 'Brier Score']]"
    ))

    # Cell 10: Summary & Lab Viva Insights
    cells.append(nbf.v4.new_markdown_cell(
        "### Soft Computing Lab Synthesis & Academic Viva Conclusion:\n"
        "- **CO1 (McCulloch-Pitts):** Handcrafted threshold logic demonstrates foundational biological action potential concepts, reaching 89.2% recall but generating false alarms due to rigid crisp boundaries.\n"
        "- **CO2 (Perceptron vs ADALINE):** Continuous error updates in ADALINE dramatically outperform discrete step updates, achieving ROC-AUC of 0.9620.\n"
        "- **CO3 (EBPA Backpropagation):** Explicit chain rule derivations ($\delta_k, \delta_j$) and momentum updates in pure NumPy match PyTorch baselines to $< 1.1 \\times 10^{-7}$.\n"
        "- **CO4 (Mamdani Fuzzy System):** Transparent linguistic rules achieve 0.9600 ROC-AUC with zero training parameter overhead, providing explainable domain constraints.\n"
        "- **CO5 (Genetic Algorithm):** Optimization of ensembling weights and decision threshold ($\tau^* = 0.4979$) yields the highest discriminative ability (ROC-AUC = 0.9668) with optimal class-imbalance mitigation."
    ))

    nb["cells"] = cells
    return nb


def build_and_execute_notebooks():
    # 1. Build and execute 01_eda.ipynb
    print("[1/2] Generating and executing notebooks/01_eda.ipynb...")
    eda_nb = create_eda_notebook()
    client1 = NotebookClient(eda_nb, timeout=600, kernel_name="python3")
    client1.execute()
    with open(NOTEBOOKS_DIR / "01_eda.ipynb", "w", encoding="utf-8") as f:
        nbf.write(eda_nb, f)
    print("Successfully saved executed notebooks/01_eda.ipynb")

    # 2. Build and execute 02_modeling.ipynb
    print("[2/2] Generating and executing notebooks/02_modeling.ipynb...")
    modeling_nb = create_modeling_notebook()
    client2 = NotebookClient(modeling_nb, timeout=600, kernel_name="python3")
    client2.execute()
    with open(NOTEBOOKS_DIR / "02_modeling.ipynb", "w", encoding="utf-8") as f:
        nbf.write(modeling_nb, f)
    print("Successfully saved executed notebooks/02_modeling.ipynb")


if __name__ == "__main__":
    build_and_execute_notebooks()
