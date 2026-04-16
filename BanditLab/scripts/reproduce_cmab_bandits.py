import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BanditLab.experiment.runner import ExperimentRunner
from BanditLab.environments.dataset_env import DatasetEnvironment
from BanditLab.models.gp_rff_model import GPRFFModel
from BanditLab.models.glm_laplace_model import GLMLaplaceModel
from BanditLab.models.linear_model import OnlineRidgeRegression
from BanditLab.algorithms.thompson_sampling import ThompsonSampling
from BanditLab.algorithms.ucb import UCBAlgorithm

N_RUNS = 2
MAX_STEPS = 500

def main():
    mab_root = Path(__file__).parents[1]
    data_path = mab_root.parent / "cMAB_bandits" / "data" / "data_5000.csv"

    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return

    env = DatasetEnvironment(dataset_path=str(data_path), max_steps=MAX_STEPS)
    n_arms = env.n_arms
    feature_dim = env.contexts_per_step.shape[2]
    steps = env.T

    results_dir = mab_root / "experiment" / "cmab_bandits"
    plots_dir = mab_root / "plots" / "cmab_bandits"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "name": "GP-TS-RFF",
            "algo_factory": lambda: ThompsonSampling(
                n_arms=n_arms,
                model=[GPRFFModel(feature_dim=feature_dim, n_features=200, nu0=1.0) for _ in range(n_arms)]
            ),
        },
        {
            "name": "GLM-TS-Laplace",
            "algo_factory": lambda: ThompsonSampling(
                n_arms=n_arms,
                model=[GLMLaplaceModel(feature_dim=feature_dim, alpha=1.0) for _ in range(n_arms)]
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
    plt.title("Paper 3: cMAB (data_5000)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "cumulative_regret.png", dpi=300)
    plt.close()

    t_range = np.arange(1, steps + 1)
    plt.figure(figsize=(12, 7))
    for name, data in all_results.items():
        regrets = np.array(data["cumulative_regret_mean"])
        avg_regret = regrets / t_range
        plt.plot(avg_regret, label=name, linewidth=2.5)
    plt.xlabel("Round t", fontsize=14)
    plt.ylabel("Average Regret", fontsize=14)
    plt.title("Paper 3: cMAB (data_5000)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "average_regret.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
