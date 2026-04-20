import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mab_framework.experiment.runner import ExperimentRunner
from mab_framework.environments.dataset_env import DatasetEnvironment
from mab_framework.models.fgts_lasso_model import FGTSLassoModel
from mab_framework.models.linear_model import OnlineRidgeRegression
from mab_framework.algorithms.thompson_sampling import ThompsonSampling
from mab_framework.algorithms.ucb import UCBAlgorithm
from mab_framework.algorithms.epsilon_greedy import EpsilonGreedy

N_RUNS = 3

def main():
    mab_root = Path(__file__).parents[1]
    data_path = mab_root.parent / "FGTS_LASSO" / "data" / "exp_10_2"

    if not data_path.exists():
        return

    env = DatasetEnvironment(dataset_path=str(data_path))
    n_arms = env.n_arms
    feature_dim = env.contexts.shape[1]
    steps = env.T

    results_dir = mab_root / "experiment" / "fgts_lasso"
    plots_dir = mab_root / "plots" / "fgts_lasso"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "name": "FGTS-LASSO",
            "algo_factory": lambda: ThompsonSampling(
                n_arms=n_arms,
                model=[FGTSLassoModel(feature_dim=feature_dim, lasso_start=100) for _ in range(n_arms)]
            ),
        },
        {
            "name": "LinUCB",
            "algo_factory": lambda: UCBAlgorithm(
                n_arms=n_arms,
                model=[OnlineRidgeRegression(feature_dim=feature_dim) for _ in range(n_arms)],
                alpha=1.0
            ),
        },
        {
            "name": "Epsilon-Greedy",
            "algo_factory": lambda: EpsilonGreedy(
                n_arms=n_arms,
                model=[OnlineRidgeRegression(feature_dim=feature_dim) for _ in range(n_arms)],
                epsilon=0.1
            ),
        },
    ]

    all_results = {}
    for config in configs:
        output_file = results_dir / f"results_{config['name'].lower().replace('-', '_')}.json"
        runner = ExperimentRunner(
            env=env,
            algorithm_factory=config["algo_factory"],
            steps=steps,
            n_runs=N_RUNS,
            output_file=str(output_file)
        )
        runner.run()
        with open(output_file, 'r') as f:
            all_results[config["name"]] = json.load(f)

    plt.figure(figsize=(12, 7))
    for name, data in all_results.items():
        plt.plot(data["cumulative_regret_mean"], label=name, linewidth=2.5)
    plt.xlabel("Round t", fontsize=14)
    plt.ylabel("Cumulative Regret", fontsize=14)
    plt.title("Paper 1: FGTS-LASSO (exp_10_2)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "cumulative_regret.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 7))
    for name, data in all_results.items():
        plt.plot(data["average_regret_mean"], label=name, linewidth=2.5)
    plt.xlabel("Round t", fontsize=14)
    plt.ylabel("Average Regret", fontsize=14)
    plt.title("Paper 1: FGTS-LASSO (exp_10_2)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "average_regret.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
