# LEGACY/OPTIONAL: original experiment only. Use src.experiment and frontend/ for Heatline.
# Old weights, label definitions and reported metrics are not current assessed evidence.
"""
Genetic Algorithm Optimizer for Neuro-Fuzzy Ensemble Blending & Threshold Selection.
KJSCE Soft Computing Lab — Module 5 (CO5).

Soft Computing Concepts:
- Chromosome Representation: Real-valued vector of length 4:
  [w_ann, w_ebpa, w_fuzzy, tau]
  where w_ann + w_ebpa + w_fuzzy = 1.0 (normalized blending weights)
  and tau in [0.45, 0.80] (regularized decision threshold to prevent distribution-shift false alarms).
- Fitness Function: Validation F1-score with precision-balance regularization:
  Fitness = F1 - 0.1 * max(0.0, Recall - Precision - 0.10)
- GA Operators:
  1. Initialization: Uniform random population in hypercube
  2. Tournament Selection: k=3 tournament size
  3. Crossover: BLX-alpha (Blend Crossover, alpha=0.3) with p_c = 0.80
  4. Mutation: Gaussian perturbation with p_m = 0.15
  5. Elitism: Top 2 elite individuals cloned unconditionally into next generation
"""

import json
import logging
from pathlib import Path
import random
import sys
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import torch
import yaml

from src.split import get_splits
from src.ann_numpy import EBPANetwork
from src.train_ann import HeatwaveMLP
from src.fuzzy_controller import MamdaniFuzzyController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_all_model_probabilities(config: dict) -> Dict[str, Dict[str, np.ndarray]]:
    """Load pre-computed or generated probabilities for Deep MLP, EBPA, and Fuzzy."""
    splits = get_splits(config)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    # 1. Deep MLP Ensemble Probabilities
    chkpt_path = Path("models/mlp_ensemble_checkpoints.pt")
    if not chkpt_path.exists():
        raise FileNotFoundError("models/mlp_ensemble_checkpoints.pt not found. Run train_ann.py first.")
    
    chkpt = torch.load(chkpt_path, weights_only=False)
    params = chkpt["best_params"]
    state_dicts = chkpt["state_dicts"]
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    mlp_val_probs_list = []
    mlp_test_probs_list = []
    for sd in state_dicts:
        m = HeatwaveMLP(n_features=X_val.shape[1], units1=params["units1"], units2=params["units2"], dropout=params["dropout"])
        m.load_state_dict(sd)
        m.eval()
        with torch.no_grad():
            mlp_val_probs_list.append(torch.sigmoid(m(X_val_t)).cpu().numpy().ravel())
            mlp_test_probs_list.append(torch.sigmoid(m(X_test_t)).cpu().numpy().ravel())
            
    p_mlp_val = np.mean(mlp_val_probs_list, axis=0)
    p_mlp_test = np.mean(mlp_test_probs_list, axis=0)

    # 2. Pure NumPy EBPA Probabilities (train optimal bipolar model)
    ebpa_chkpt_path = Path("models/ebpa_weights.npz")
    ebpa = EBPANetwork(n_input=X_train.shape[1], n_hidden=32, activation="bipolar_sigmoid", learning_rate=0.01, momentum=0.9, seed=42)
    
    if ebpa_chkpt_path.exists():
        ebpa.load_weights(str(ebpa_chkpt_path))
    else:
        logger.info("Training EBPA network for neuro-fuzzy blending...")
        ebpa.train(X_train, y_train, X_val, y_val, max_epochs=200, patience=30)
        ebpa.save_weights(str(ebpa_chkpt_path))

    p_ebpa_val = ebpa.predict_proba(X_val)
    p_ebpa_test = ebpa.predict_proba(X_test)

    # 3. Mamdani Fuzzy Controller Probabilities (from raw features)
    features_csv = Path(config["data"]["features_file"])
    df = pd.read_csv(features_csv)
    val_df = df[(df["year"] >= config["splits"]["val"]["start_year"]) & (df["year"] <= config["splits"]["val"]["end_year"])]
    test_df = df[(df["year"] >= config["splits"]["test"]["start_year"]) & (df["year"] <= config["splits"]["test"]["end_year"])]

    fuzzy = MamdaniFuzzyController()
    p_fuzzy_val = fuzzy.predict_proba(val_df)
    p_fuzzy_test = fuzzy.predict_proba(test_df)

    return {
        "val": {
            "p_ann": p_mlp_val,
            "p_ebpa": p_ebpa_val,
            "p_fuzzy": p_fuzzy_val,
            "y": y_val,
        },
        "test": {
            "p_ann": p_mlp_test,
            "p_ebpa": p_ebpa_test,
            "p_fuzzy": p_fuzzy_test,
            "y": y_test,
        },
    }


class GeneticAlgorithmOptimizer:
    """
    Real-coded Genetic Algorithm for evolving Neuro-Fuzzy ensemble blending weights
    and decision threshold tau on the Validation set.
    """

    def __init__(
        self,
        pop_size: int = 40,
        n_generations: int = 50,
        crossover_rate: float = 0.80,
        mutation_rate: float = 0.15,
        tournament_k: int = 3,
        seed: int = 42,
    ):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def decode_chromosome(self, chrom: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert raw chromosome genes [g0, g1, g2, g3] into normalized weights and threshold."""
        raw_weights = np.maximum(0.01, chrom[:3])
        weights = raw_weights / np.sum(raw_weights)
        tau = float(np.clip(chrom[3], 0.45, 0.80))
        return float(weights[0]), float(weights[1]), float(weights[2]), tau

    def compute_fitness(self, chrom: np.ndarray, p_ann: np.ndarray, p_ebpa: np.ndarray, p_fuzzy: np.ndarray, y_true: np.ndarray) -> float:
        """
        Evaluate fitness = F1 score with precision-penalty constraint.
        Penalizes excessive false alarms to ensure generalization across climate regimes.
        """
        w_ann, w_ebpa, w_fuzzy, tau = self.decode_chromosome(chrom)
        p_hybrid = w_ann * p_ann + w_ebpa * p_ebpa + w_fuzzy * p_fuzzy
        preds = (p_hybrid >= tau).astype(int)

        if np.sum(preds) == 0:
            return 0.0

        f1 = f1_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)

        # Penalty for severe precision-recall imbalance
        penalty = 0.15 * max(0.0, rec - prec - 0.12)
        fitness = float(f1 - penalty)
        return max(0.0, fitness)

    def tournament_selection(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        """Tournament selection of size k."""
        selected_idx = np.random.choice(len(population), size=self.tournament_k, replace=False)
        best_in_tournament = selected_idx[np.argmax(fitnesses[selected_idx])]
        return population[best_in_tournament].copy()

    def crossover_blx_alpha(self, parent1: np.ndarray, parent2: np.ndarray, alpha: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """Blend Crossover (BLX-alpha) for real-valued chromosomes."""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        for i in range(len(parent1)):
            d = abs(parent1[i] - parent2[i])
            low = min(parent1[i], parent2[i]) - alpha * d
            high = max(parent1[i], parent2[i]) + alpha * d
            
            # Bound gene ranges
            if i < 3:
                low = max(0.01, low)
            else:
                low = max(0.45, low)
                high = min(0.80, high)

            child1[i] = random.uniform(low, high)
            child2[i] = random.uniform(low, high)

        return child1, child2

    def mutate_gaussian(self, chrom: np.ndarray, sigma: float = 0.06) -> np.ndarray:
        """Gaussian mutation operator."""
        mutated = chrom.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] += random.gauss(0.0, sigma)
                if i < 3:
                    mutated[i] = max(0.01, mutated[i])
                else:
                    mutated[i] = float(np.clip(mutated[i], 0.45, 0.80))
        return mutated

    def optimize(
        self,
        p_ann: np.ndarray,
        p_ebpa: np.ndarray,
        p_fuzzy: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Run real-coded Genetic Algorithm over Validation probabilities."""
        logger.info(f"Initializing GA population (size={self.pop_size}, generations={self.n_generations})...")

        # Initialize population: 3 weights in [0.1, 1.0], threshold in [0.45, 0.80]
        population = np.zeros((self.pop_size, 4))
        for i in range(self.pop_size):
            population[i, 0] = random.uniform(0.3, 0.8)   # ANN weight
            population[i, 1] = random.uniform(0.1, 0.5)   # EBPA weight
            population[i, 2] = random.uniform(0.1, 0.4)   # Fuzzy weight
            population[i, 3] = random.uniform(0.48, 0.72) # Threshold

        history = {"best_fitness": [], "mean_fitness": [], "generation": []}

        for gen in range(1, self.n_generations + 1):
            fitnesses = np.array([
                self.compute_fitness(ind, p_ann, p_ebpa, p_fuzzy, y_val)
                for ind in population
            ])

            best_idx = np.argmax(fitnesses)
            best_fit = float(fitnesses[best_idx])
            mean_fit = float(np.mean(fitnesses))

            history["generation"].append(gen)
            history["best_fitness"].append(best_fit)
            history["mean_fitness"].append(mean_fit)

            if gen % 10 == 0 or gen == 1 or gen == self.n_generations:
                w1, w2, w3, t = self.decode_chromosome(population[best_idx])
                logger.info(
                    f"Generation {gen:2d}/{self.n_generations}: Best Fitness={best_fit:.4f}, Mean={mean_fit:.4f} | "
                    f"Weights=[ANN:{w1:.3f}, EBPA:{w2:.3f}, Fuzzy:{w3:.3f}], Tau={t:.3f}"
                )

            # Elitism: retain top 2
            sorted_indices = np.argsort(fitnesses)[::-1]
            next_population = [
                population[sorted_indices[0]].copy(),
                population[sorted_indices[1]].copy(),
            ]

            # Generate rest of offspring
            while len(next_population) < self.pop_size:
                p1 = self.tournament_selection(population, fitnesses)
                p2 = self.tournament_selection(population, fitnesses)
                c1, c2 = self.crossover_blx_alpha(p1, p2)
                next_population.append(self.mutate_gaussian(c1))
                if len(next_population) < self.pop_size:
                    next_population.append(self.mutate_gaussian(c2))

            population = np.array(next_population)

        # Final evaluation of the champion chromosome
        final_fitnesses = np.array([
            self.compute_fitness(ind, p_ann, p_ebpa, p_fuzzy, y_val)
            for ind in population
        ])
        champion_idx = np.argmax(final_fitnesses)
        w_ann, w_ebpa, w_fuzzy, tau = self.decode_chromosome(population[champion_idx])

        # Compute full validation metrics for champion
        p_val_hybrid = w_ann * p_ann + w_ebpa * p_ebpa + w_fuzzy * p_fuzzy
        val_preds = (p_val_hybrid >= tau).astype(int)

        val_f1 = float(f1_score(y_val, val_preds, zero_division=0))
        val_prec = float(precision_score(y_val, val_preds, zero_division=0))
        val_rec = float(recall_score(y_val, val_preds, zero_division=0))
        val_pr_auc = float(average_precision_score(y_val, p_val_hybrid))
        val_roc_auc = float(roc_auc_score(y_val, p_val_hybrid))

        logger.info("=== Genetic Algorithm Neuro-Fuzzy Optimization Complete ===")
        logger.info(f"Optimal Blending Weights: ANN={w_ann:.4f}, EBPA={w_ebpa:.4f}, Fuzzy={w_fuzzy:.4f}")
        logger.info(f"Optimal Decision Threshold: tau* = {tau:.4f}")
        logger.info(f"Validation F1={val_f1:.4f}, Precision={val_prec:.4f}, Recall={val_rec:.4f}, PR-AUC={val_pr_auc:.4f}")

        return {
            "w_ann": round(w_ann, 4),
            "w_ebpa": round(w_ebpa, 4),
            "w_fuzzy": round(w_fuzzy, 4),
            "threshold": round(tau, 4),
            "val_f1": round(val_f1, 4),
            "val_precision": round(val_prec, 4),
            "val_recall": round(val_rec, 4),
            "val_pr_auc": round(val_pr_auc, 4),
            "val_roc_auc": round(val_roc_auc, 4),
            "history": history,
        }


def run_ga_optimization(config: dict) -> Dict[str, Any]:
    """Execute end-to-end GA optimization pipeline and save threshold.json."""
    probs_dict = load_all_model_probabilities(config)
    val_data = probs_dict["val"]

    optimizer = GeneticAlgorithmOptimizer(pop_size=40, n_generations=50, seed=42)
    results = optimizer.optimize(
        p_ann=val_data["p_ann"],
        p_ebpa=val_data["p_ebpa"],
        p_fuzzy=val_data["p_fuzzy"],
        y_val=val_data["y"],
    )

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save threshold.json
    threshold_data = {
        "w_ann": results["w_ann"],
        "w_ebpa": results["w_ebpa"],
        "w_fuzzy": results["w_fuzzy"],
        "threshold": results["threshold"],
        "val_f1": results["val_f1"],
        "val_precision": results["val_precision"],
        "val_recall": results["val_recall"],
        "val_pr_auc": results["val_pr_auc"],
        "val_roc_auc": results["val_roc_auc"],
    }
    with open(models_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=2)
    logger.info(f"Saved optimal weights and threshold to {models_dir / 'threshold.json'}")

    # Save GA convergence history for Fig 12
    with open(models_dir / "ga_convergence_history.json", "w", encoding="utf-8") as f:
        json.dump(results["history"], f, indent=2)
    logger.info(f"Saved GA convergence history to {models_dir / 'ga_convergence_history.json'}")

    return results


if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_ga_optimization(cfg)
