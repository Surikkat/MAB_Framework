import yaml
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiment.runner import ExperimentRunner

import environments
import algorithms

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_1.yaml")
    args = parser.parse_args()

    config = load_config(args.config)['experiment']

    env_name = config['environment']['name']
    env_params = config['environment'].get('params', {})
    EnvClass = getattr(environments, env_name)
    env = EnvClass(**env_params)

    algo_name = config['algorithm']['name']
    algo_params = config['algorithm'].get('params', {})
    AlgoClass = getattr(algorithms, algo_name)
    algo = AlgoClass(**algo_params)

    steps = config['steps']
    output_file = f"experiment/results_{config['name']}.json"
    
    runner = ExperimentRunner(env=env, algorithm=algo, steps=steps, output_file=output_file)
    runner.run()

if __name__ == "__main__":
    main()