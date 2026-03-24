import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments.dataset_env import DatasetEnvironment
from algorithms.linucb import LinUCBAlgorithm
from algorithms.neural_ucb import NeuralUCBAlgorithm
from experiment.runner import ExperimentRunner
from plots.plot_results import plot_experiments

def main():
    dataset_path = "data/E1_dataset.npz"
    steps = 200
    n_arms = 50
    feature_dim = 5
    
    results_dir = "experiment/nn_agp_reproduce"
    os.makedirs(results_dir, exist_ok=True)
    
    env = DatasetEnvironment(dataset_path=dataset_path)
    
    algorithms = {
        "LinUCB": LinUCBAlgorithm(n_arms=n_arms, feature_dim=feature_dim, alpha=1.0),
        "NeuralUCB": NeuralUCBAlgorithm(n_arms=n_arms, feature_dim=feature_dim, hidden_dim=32, alpha=1.0, lr=0.01)
    }
    
    for name, algo in algorithms.items():
        env.reset()
        output_file = f"{results_dir}/results_{name}.json"
        runner = ExperimentRunner(env=env, algorithm=algo, steps=steps, output_file=output_file)
        runner.run()
        
    plot_experiments(results_dir=results_dir, save_dir="plots/nn_agp_reproduce")

if __name__ == "__main__":
    main()