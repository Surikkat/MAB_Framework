import sys
import os

# Добавляем корень проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments.dataset_env import DatasetEnvironment
from algorithms.epsilon_greedy import EpsilonGreedy
from algorithms.linucb import LinUCBAlgorithm
from algorithms.fgts import FGTSAlgorithm
from experiment.runner import ExperimentRunner
from plots.plot_results import plot_experiments

def main():
    dataset_path = "data/fgts_exp_1"
    steps = 500
    n_arms = 5
    feature_dim = 10
    
    results_dir = "experiment/fgts_reproduce"
    os.makedirs(results_dir, exist_ok=True)
    env = DatasetEnvironment(dataset_path=dataset_path)
    
    algorithms = {
        "EpsilonGreedy": EpsilonGreedy(n_arms=n_arms, epsilon=0.1),
        "LinUCB": LinUCBAlgorithm(n_arms=n_arms, feature_dim=feature_dim, alpha=1.0),
        "FGTS_Lasso": FGTSAlgorithm(n_arms=n_arms, feature_dim=feature_dim, lasso_start=50, lasso_period=50)
    }
    
    for name, algo in algorithms.items():
        env.reset()
        output_file = f"{results_dir}/results_{name}.json"
        runner = ExperimentRunner(env=env, algorithm=algo, steps=steps, output_file=output_file)
        runner.run()

    plot_experiments(results_dir=results_dir, save_dir="plots/fgts_reproduce")

if __name__ == "__main__":
    main()