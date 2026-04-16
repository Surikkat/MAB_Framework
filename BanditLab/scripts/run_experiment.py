import yaml
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BanditLab.experiment.runner import ExperimentRunner
import BanditLab.environments as environments
import BanditLab.models as models
import BanditLab.algorithms as algorithms

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_1.yaml")
    args = parser.parse_args()

    config_all = load_config(args.config)
    config = config_all['experiment']

    env_config = config['environment']
    EnvClass = getattr(environments, env_config['name'])
    env_params = env_config.get('params', {})
    env = EnvClass(**env_params)

    n_arms = getattr(env, 'n_arms', config.get('n_arms'))
    if n_arms is None:
        raise ValueError("n_arms must be defined in environment or top-level config")

    model_config = config.get('model')
    algo_config = config['algorithm']

    def algorithm_factory():
        model = None
        if model_config:
            model_name = model_config['name']
            model_params = model_config.get('params', {})
            ModelClass = getattr(models, model_name)
            one_model_per_arm = config.get('one_model_per_arm', True)
            if one_model_per_arm:
                model = [ModelClass(**model_params) for _ in range(n_arms)]
            else:
                model = ModelClass(**model_params)

        algo_name = algo_config['name']
        algo_params = algo_config.get('params', {})
        AlgoClass = getattr(algorithms, algo_name)
        algo_params['n_arms'] = n_arms
        if model is not None:
            algo_params['model'] = model
        return AlgoClass(**algo_params)

    steps = config['steps']
    env_T = getattr(env, 'T', steps)
    if steps > env_T:
        print(f"[Warning] config steps ({steps}) > env.T ({env_T}). Clipping to {env_T}.")
        steps = env_T
    n_runs = config.get('n_runs', 1)
    output_file = f"experiment/results_{config['name']}.json"

    runner = ExperimentRunner(
        env=env,
        algorithm_factory=algorithm_factory,
        steps=steps,
        n_runs=n_runs,
        output_file=output_file
    )
    runner.run()

if __name__ == "__main__":
    main()