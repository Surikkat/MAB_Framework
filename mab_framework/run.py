import yaml
import shutil
import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from mab_framework.experiment.runner import ExperimentRunner
import mab_framework.environments as environments
import mab_framework.models as models
import mab_framework.algorithms as algorithms


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_algo_factory(algo_conf, n_arms, feature_dim=None):
    def factory():
        model = None
        model_config = algo_conf.get('model')
        if model_config:
            model_name = model_config['name']
            model_params = dict(model_config.get('params', {}))

            if feature_dim is not None:
                if 'feature_dim' not in model_params or model_params['feature_dim'] == 'auto':
                    model_params['feature_dim'] = feature_dim

            ModelClass = getattr(models, model_name)
            one_model = algo_conf.get('one_model_per_arm', model_config.get('one_model_per_arm', True))
            if one_model:
                model = [ModelClass(**model_params) for _ in range(n_arms)]
            else:
                model = ModelClass(**model_params)

        algo_name = algo_conf['name']
        algo_params = dict(algo_conf.get('params', {}))
        AlgoClass = getattr(algorithms, algo_name)
        algo_params['n_arms'] = n_arms
        if model is not None:
            algo_params['model'] = model
        return AlgoClass(**algo_params)
    return factory


def print_leaderboard(all_results):
    rows = []
    for name, data in all_results.items():
        cum_regret = data.get("cumulative_regret_mean", [])
        times = data.get("times_mean", [])
        rows.append({
            "Algorithm": name,
            "Regret": round(cum_regret[-1], 3) if cum_regret else None,
            "Runtime (s)": round(sum(times), 4) if times else None,
        })
    df = pd.DataFrame(rows).sort_values("Regret")
    print("\n" + df.to_string(index=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="MAB Framework Runner")
    parser.add_argument("config", type=str, help="Path to the experiment YAML config")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)
    exp_config = config.get('experiment', {})

    exp_name = exp_config.get('name', 'experiment')
    steps = exp_config.get('steps', 100)
    n_runs = exp_config.get('n_runs', 1)
    seed = exp_config.get('seed', None)

    env_config = config.get('environment', {})
    env_name = env_config.get('name', env_config.get('type'))
    if not env_name:
        raise ValueError("Environment 'name' or 'type' must be specified in the config")
    EnvClass = getattr(environments, env_name)
    env_params = env_config.get('params', {})

    if 'delay' in env_config:
        env_params['delay_config'] = env_config['delay']

    env = EnvClass(**env_params)

    n_arms = getattr(env, 'n_arms', env_config.get('n_arms'))
    if n_arms is None:
        raise ValueError("n_arms must be defined in environment or environment_config")

    feature_dim = None
    try:
        sample_context = env.get_context()
        feature_dim = sample_context.shape[-1]
        env.reset()
    except Exception as e:
        print(f"Warning: could not automatically determine feature_dim: {e}")

    env_max_steps = getattr(env, 'T', steps)
    if steps > env_max_steps:
        steps = env_max_steps

    output_config = config.get('output', {})
    save_path = Path(output_config.get('save_path', f"./results/{exp_name}"))
    save_path.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, save_path / "config.yaml")

    algo_configs = config.get('algorithms', [])
    if not algo_configs:
        raise ValueError("No algorithms specified in the config")

    env_meta = {
        "name": env_name,
        "dataset": env_config.get("params", {}).get("dataset_path", ""),
    }

    all_results = {}
    metrics_to_plot = config.get('metrics', ["cumulative_regret", "average_regret"])

    for algo_conf in algo_configs:
        a_name = algo_conf.get('display_name', algo_conf['name'])
        algo_dir = save_path / a_name.replace(' ', '_')

        metadata = {
            "algorithm": algo_conf['name'],
            "display_name": a_name,
            "environment": env_meta,
        }

        algo_factory = make_algo_factory(algo_conf, n_arms, feature_dim=feature_dim)

        runner = ExperimentRunner(
            env=env,
            algorithm_factory=algo_factory,
            steps=steps,
            n_runs=n_runs,
            seed=seed,
            save_dir=str(algo_dir),
            metadata=metadata,
        )
        all_results[a_name] = runner.run()

    t_range = np.arange(1, steps + 1)

    plot_labels = {
        "cumulative_regret": "Cumulative Regret",
        "average_regret": "Average Regret",
    }

    for target_metric in metrics_to_plot:
        metric_key = f"{target_metric}_mean"

        plt.figure(figsize=(10, 6))

        for a_name, data in all_results.items():
                if metric_key in data:
                    y_data = data[metric_key]
                    plt.plot(t_range[:len(y_data)], y_data, label=a_name, linewidth=2.0)

        plt.xlabel("Round t", fontsize=12)
        ylabel = plot_labels.get(target_metric, target_metric.replace('_', ' ').title())
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"{exp_name} - {ylabel}", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plot_file = save_path / f"{target_metric}.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()

    print_leaderboard(all_results)


if __name__ == "__main__":
    main()
