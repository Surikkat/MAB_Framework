import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiment.runner import ExperimentRunner
from environments.dataset_env import DatasetEnvironment
from models.nn_agp_model import NNAGPModel
from models.linear_model import OnlineRidgeRegression
from algorithms.nn_agp_ucb import NNAGPUCBAlgorithm
from algorithms.ucb import UCBAlgorithm

N_RUNS = 2
MAX_STEPS = 200

def main():
    mab_root = Path(__file__).parents[1]
    data_path = mab_root.parent / "NN_AGP" / "datasets" / "E1_lowdim.npz"

    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return

    env = DatasetEnvironment(dataset_path=str(data_path), max_steps=MAX_STEPS)
    n_arms = env.n_arms
    theta_dim = env.thetas.shape[1]
    x_dim = env.X_pool.shape[1]
    steps = env.T

    results_dir = mab_root / "experiment" / "nn_agp"
    plots_dir = mab_root / "plots" / "nn_agp"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "name": "NN-AGP-UCB",
            "algo_factory": lambda: NNAGPUCBAlgorithm(
                n_arms=n_arms,
                model=NNAGPModel(
                    theta_dim=theta_dim,
                    x_dim=x_dim,
                    m=5,
                    hidden_dim=32,
                    mll_steps=20,
                    lr=2e-3
                ),
                alpha=float(np.sqrt(2.0))
            ),
        },
        {
            "name": "LinUCB",
            "algo_factory": lambda: UCBAlgorithm(
                n_arms=n_arms,
                model=[OnlineRidgeRegression(feature_dim=theta_dim + x_dim) for _ in range(n_arms)],
                alpha=1.0
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
    plt.title("Paper 2: NN-AGP (E1_lowdim)", fontsize=16, fontweight='bold')
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
    plt.title("Paper 2: NN-AGP (E1_lowdim)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "average_regret.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
