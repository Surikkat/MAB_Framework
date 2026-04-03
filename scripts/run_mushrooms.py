import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiment.runner import ExperimentRunner
from environments.dataset_env import DatasetEnvironment
from models.fgts_lasso_model import FGTSLassoModel
from models.linear_model import OnlineRidgeRegression
from models.gp_rff_model import GPRFFModel
from models.glm_laplace_model import GLMLaplaceModel
from models.nn_agp_model import NNAGPModel
from models.neural_network import NeuralLinearModel

from algorithms.thompson_sampling import ThompsonSampling
from algorithms.ucb import UCBAlgorithm
from algorithms.epsilon_greedy import EpsilonGreedy
from algorithms.neural_ucb import NeuralUCBAlgorithm

N_RUNS = 5
MAX_STEPS = 200

def main():
    mab_root = Path(__file__).parents[1]
    data_path = mab_root / "data" / "mushroom_bandit_5000.csv"

    if not data_path.exists():
        print(f"Dataset not found at: {data_path}")
        return

    env = DatasetEnvironment(dataset_path=str(data_path), max_steps=MAX_STEPS)
    n_arms = env.n_arms
    feature_dim = env.contexts_per_step.shape[2]
    steps = env.T

    results_dir = mab_root / "experiment" / "mushrooms"
    plots_dir = mab_root / "plots" / "mushrooms"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    configs = [
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
        {
            "name": "FGTS-LASSO",
            "algo_factory": lambda: ThompsonSampling(
                n_arms=n_arms,
                model=[FGTSLassoModel(feature_dim=feature_dim, lasso_start=100) for _ in range(n_arms)]
            ),
        },
        {
            "name": "GP-TS-RFF",
            "algo_factory": lambda: ThompsonSampling(
                n_arms=n_arms,
                model=[GPRFFModel(feature_dim=feature_dim, n_features=100, nu0=1.0) for _ in range(n_arms)]
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
            "name": "NeuralLinear",
            "algo_factory": lambda: NeuralUCBAlgorithm(
                n_arms=n_arms,
                model=[NeuralLinearModel(feature_dim=feature_dim, hidden_dim=32, lr=0.01) for _ in range(n_arms)],
                alpha=1.0
            ),
        },
        # {
        #     "name": "NN-AGP-UCB",
        #     "algo_factory": lambda: UCBAlgorithm(
        #         n_arms=n_arms,
        #         model=[NNAGPModel(
        #             theta_dim=feature_dim // 2, 
        #             x_dim=feature_dim - (feature_dim // 2)
        #         ) for _ in range(n_arms)],
        #         alpha=2.0
        #     ),
        # },
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
        try:
            runner.run()
        except KeyboardInterrupt:
            print(f"\n[!] Interrupted during {config['name']}")
            pass
        
        if output_file.exists():
            with open(output_file, 'r') as f:
                all_results[config["name"]] = json.load(f)
                
    if not all_results:
        print("No results to plot.")
        return

    plt.figure(figsize=(12, 7))
    for name, data in all_results.items():
        plt.plot(data["cumulative_regret_mean"], label=name, linewidth=2.5)
    plt.xlabel("Round t", fontsize=14)
    plt.ylabel("Cumulative Regret", fontsize=14)
    plt.title("All Models: Cumulative Regret (Mushrooms)", fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "cumulative_regret.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 7))
    for name, data in all_results.items():
        avg_regret = np.array(data["cumulative_regret_mean"]) / (np.arange(len(data["cumulative_regret_mean"])) + 1)
        plt.plot(avg_regret, label=name, linewidth=2.5)
    plt.xlabel("Round t", fontsize=14)
    plt.ylabel("Average Regret", fontsize=14)
    plt.title("All Models: Average Regret (Mushrooms)", fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "average_regret.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
