import yaml
import argparse
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mab_framework.experiment.runner import ExperimentRunner
import mab_framework.environments as environments
import mab_framework.models as models
import mab_framework.algorithms as algorithms


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def infer_feature_dim(env):
    """Пытается определить размерность контекста из среды."""
    try:
        ctx = env.get_context()
        if hasattr(ctx, 'shape') and len(ctx.shape) >= 1:
            return ctx.shape[-1]
    except Exception:
        pass

    # Fallback: ищем атрибут context_dim
    for attr in ('context_dim', 'feature_dim', 'd', 'x_dim'):
        if hasattr(env, attr):
            return getattr(env, attr)

    return None


def make_algorithm_factory(algo_spec, n_arms, feature_dim):
    """Создаёт фабрику для одного алгоритма из спецификации YAML."""
    def factory():
        # --- Модель ---
        model = None
        model_spec = algo_spec.get('model')
        if model_spec:
            ModelClass = getattr(models, model_spec['name'])
            model_params = dict(model_spec.get('params', {}))
            one_per_arm = model_spec.get('one_model_per_arm', True)

            # Автоподстановка feature_dim
            init_vars = ModelClass.__init__.__code__.co_varnames
            if feature_dim is not None:
                for key in ('feature_dim', 'd', 'input_dim', 'n_features'):
                    if key in init_vars:
                        model_params.setdefault(key, feature_dim)
                        break

            if one_per_arm:
                model = [ModelClass(**model_params) for _ in range(n_arms)]
            else:
                model = ModelClass(**model_params)

        # --- Алгоритм ---
        AlgoClass = getattr(algorithms, algo_spec['name'])
        algo_params = dict(algo_spec.get('params', {}))
        algo_params['n_arms'] = n_arms

        # Автоподстановка размерности контекста, если нужна
        init_vars = AlgoClass.__init__.__code__.co_varnames
        if feature_dim is not None:
            for key in ('d', 'x_dim', 'theta_dim', 'context_dim', 'input_dim', 'n_features'):
                if key in init_vars:
                    algo_params.setdefault(key, feature_dim)
                    break

        # Удаляем параметры, которые алгоритм не принимает (например, n_models/lr у BootstrapTSBandit)
        valid_keys = set(init_vars)
        algo_params = {k: v for k, v in algo_params.items() if k in valid_keys}

        if model is not None:
            algo_params['model'] = model

        return AlgoClass(**algo_params)

    return factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_all = load_config(args.config)
    config = config_all['experiment']
    env_config = config_all['environment']
    algo_list = config_all.get('algorithms', [])

    if not algo_list:
        raise ValueError("No algorithms defined in config.")

    # --- Среда ---
    EnvClass = getattr(environments, env_config['name'])
    env_params = env_config.get('params', {})
    env = EnvClass(**env_params)

    n_arms = getattr(env, 'n_arms', None) or config.get('n_arms')
    if n_arms is None:
        raise ValueError("n_arms must be defined in environment or top-level config")

    # --- Размерность контекста ---
    feature_dim = infer_feature_dim(env)
    print(f"[Info] Detected feature_dim = {feature_dim}, n_arms = {n_arms}")

    # --- Ограничение шагов ---
    steps = config['steps']
    env_T = getattr(env, 'T', None)
    if env_T is not None and steps > env_T:
        print(f"[Warning] config steps ({steps}) > env.T ({env_T}). Clipping to {env_T}.")
        steps = env_T
    n_runs = config.get('n_runs', 1)
    seed = config.get('seed', 42)

    # --- Выходная директория ---
    save_path = Path(config_all.get('output', {}).get('save_path', './results'))
    save_path.mkdir(parents=True, exist_ok=True)

    # --- Запуск каждого алгоритма ---
    for algo_spec in algo_list:
        display_name = algo_spec.get('display_name', algo_spec['name'])
        safe_name = (display_name
                     .replace(' ', '_')
                     .replace('(', '').replace(')', '')
                     .replace('=', '').replace('/', '_'))
        output_file = save_path / f"results_{safe_name}.json"

        print(f"\n=== Running: {display_name} ===")
        factory = make_algorithm_factory(algo_spec, n_arms, feature_dim)

        try:
            runner = ExperimentRunner(
                env=env,
                algorithm_factory=factory,
                steps=steps,
                n_runs=n_runs,
                output_file=str(output_file),
                seed=seed,
            )
            runner.run()
            print(f"✅ Saved to {output_file}")
        except Exception as e:
            print(f"❌ Error in {display_name}: {e}")
            continue


if __name__ == "__main__":
    main()