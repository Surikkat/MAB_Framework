import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Callable
import yaml

MAB_PATH = Path(__file__).resolve().parents[2]
if str(MAB_PATH) not in sys.path:
    sys.path.insert(0, str(MAB_PATH))

import mab_framework.environments as environments
import mab_framework.models as models
import mab_framework.algorithms as algorithms
from mab_framework.experiment.runner import ExperimentRunner


def get_available_environments():
    """Возвращает список доступных сред"""
    envs = [
        # === СИНТЕТИКА ===
        {
            'name': 'Linear Synthetic',
            'id': 'synthetic_linear',
            'type': 'synthetic',
            'description': 'Линейная среда: r = <θ_a, x> + шум',
            'env_class': 'SyntheticLinearEnv',
            'default_params': {'n_arms': 10, 'context_dim': 5, 'noise_std': 0.1},
            'n_arms': 10,
        },
        {
            'name': 'GLM Synthetic',
            'id': 'synthetic_glm',
            'type': 'synthetic',
            'description': 'Логистическая среда: P(r=1) = σ(<θ_a, x>)',
            'env_class': 'SyntheticGLMEnv',
            'default_params': {'n_arms': 10, 'context_dim': 5},
            'n_arms': 10,
        },
        {
            'name': 'Neural Synthetic',
            'id': 'synthetic_neural',
            'type': 'synthetic',
            'description': 'Нелинейная среда: r = MLP(x) с ReLU',
            'env_class': 'SyntheticNeuralEnv',
            'default_params': {'n_arms': 10, 'context_dim': 5, 'hidden_dim': 32, 'noise_std': 0.1},
            'n_arms': 10,
        },
        {
            'name': 'Non-contextual Synthetic',
            'id': 'synthetic_noncontextual',
            'type': 'synthetic',
            'description': 'Безконтекстная среда: у каждой руки фиксированное среднее',
            'env_class': 'SyntheticNonContextualEnv',
            'default_params': {'n_arms': 10, 'context_dim': 1, 'noise_std': 1.0},
            'n_arms': 10,
        },
        # === SOTA ===
        {
            'name': 'Mushrooms',
            'id': 'mushrooms',
            'type': 'real',
            'description': 'Классификация грибов (съедобный/ядовитый)',
            'env_class': 'DatasetEnvironment',
            'default_params': {
                'dataset_path': str(MAB_PATH / 'mab_framework/data/mushroom_bandit_5000.csv'),
            },
            'n_arms': None,
        },
        {
            'name': 'MovieLens 100K',
            'id': 'movielens',
            'type': 'real',
            'description': 'Реальные данные рекомендаций фильмов',
            'env_class': 'DatasetEnvironment',
            'default_params': {
                'dataset_path': str(MAB_PATH / 'mab_framework/data/movielens_bandit_5000.csv'),
            },
            'n_arms': None,
        },
    ]
    return pd.DataFrame(envs)


def get_available_algorithms_for_online():
    """Возвращает алгоритмы, готовые для онлайн-запуска"""
    algorithms = [
        {
            'name': 'Epsilon-Greedy',
            'algo_name': 'EpsilonGreedy',
            'params': {},
            'model_name': 'OnlineRidgeRegression',
            'model_params': {},
            'category': '🎲 Non-contextual',
        },
        {
            'name': 'UCB',
            'algo_name': 'UCBAlgorithm',
            'params': {},
            'model_name': 'OnlineRidgeRegression',
            'model_params': {},
            'category': '🎲 Non-contextual',
        },
        {
            'name': 'LinUCB',
            'algo_name': 'LinUCBAlgorithm',
            'params': {},
            'model_name': 'OnlineRidgeRegression',
            'model_params': {},
            'category': '📈 Linear',
        },
        {
            'name': 'Thompson Sampling',
            'algo_name': 'ThompsonSampling',
            'params': {},
            'model_name': 'OnlineRidgeRegression',
            'model_params': {},
            'category': '🎲 Non-contextual',
        },
        {
            'name': 'Bootstrap TS',
            'algo_name': 'BootstrapTSBandit',
            'params': {},
            'model_name': 'BootstrapEnsembleModel',
            'model_params': {},
            'category': '🎲 Non-contextual',
        },
        {
            'name': 'NonContextual TS',
            'algo_name': 'NonContextualTSBandit',
            'params': {},
            'model_name': None,
            'model_params': None,
            'category': '🎲 Non-contextual',
        },
        {
            'name': 'GP-UCB (mult kernel)',
            'algo_name': 'GPUCBKernelFlexibleAlgorithm',
            'params': {},
            'model_name': 'GPRFFModel',
            'model_params': {},
            'category': '🔮 Gaussian Process',
        },
        {
            'name': 'GP-UCB Kernel (adapt)',
            'algo_name': 'GPUCBKernelFlexibleAlgorithm',
            'params': {'kernel_type': 'adaptive'},
            'model_name': 'GPRFFModel',
            'model_params': {},
            'category': '🔮 Gaussian Process',
        },
        {
            'name': 'GP-TS',
            'algo_name': 'GPTSBandit',
            'params': {},
            'model_name': 'GPRFFModel',
            'model_params': {},
            'category': '🔮 Gaussian Process',
        },
        {
            'name': 'Exact GP',
            'algo_name': 'ThompsonSampling',
            'params': {},
            'model_name': 'ExactGPModel',
            'model_params': {},
            'category': '🔮 Gaussian Process',
        },
        {
            'name': 'Kernel UCB',
            'algo_name': 'UCBAlgorithm',
            'params': {},
            'model_name': 'KernelUCBModel',
            'model_params': {},
            'category': '🔮 Gaussian Process',
        },
        {
            'name': 'Neural UCB',
            'algo_name': 'NeuralUCBAlgorithm',
            'params': {},
            'model_name': 'NeuralUCBModel',
            'model_params': {},
            'category': '🧠 Neural',
        },
        {
            'name': 'NN-AGP UCB',
            'algo_name': 'NNAGPUCBAlgorithm',
            'params': {},
            'model_name': 'NNAGPModel',
            'model_params': {},
            'category': '🧠 Neural',
        },
        # {
        #     'name': 'PFN-TS (Adaptive TabICL)',
        #     'algo_name': 'PFNTSAlgorithm',
        #     'params': {},
        #     'model_name': 'TabICLRegressorPPD',
        #     'model_params': {},
        #     'category': '🧠 Neural',
        # },
        {
            'name': 'FGTS',
            'algo_name': 'FGTSAlgorithm',
            'params': {},
            'model_name': 'FGTSModel',
            'model_params': {},
            'category': '🎲 Non-contextual',
        },
        {
            'name': 'FGTS Lasso',
            'algo_name': 'FGTSAlgorithm',
            'params': {},
            'model_name': 'FGTSLassoModel',
            'model_params': {},
            'category': '🎲 Non-contextual',
        },
    ]
    
    algorithms.append({
        'name': 'RegCB',
        'algo_name': 'RegCBBandit',
        'params': {},
        'model_name': 'OnlineRidgeRegression',
        'model_params': {},
        'category': '📉 GLM',
    })
        
    algorithms.extend([
        {
            'name': 'SGD-TS',
            'algo_name': 'SGDTSBandit',
            'params': {},
            'model_name': 'SGDModel',
            'model_params': {},
            'category': '📈 Linear',
        },
        {
            'name': 'Custom TS (GLM Laplace)',
            'algo_name': 'CustomTSBandit',
            'params': {},
            'model_name': 'GLMLaplaceModel',
            'model_params': {},
            'category': '📉 GLM',
        },
        {
            'name': 'NN-AGP Adaptive',
            'algo_name': 'NNAGPUCBAdaptiveAlgorithm',
            'params': {},
            'model_name': 'NNAGPModel',
            'model_params': {},
            'category': '🧠 Neural',
        },
        {
            'name': 'NN-UCB',
            'algo_name': 'NNUCBAlgorithm',
            'params': {},
            'model_name': 'NNUCBModel',
            'model_params': {},
            'category': '🧠 Neural',
        },
        {
            'name': 'NN-TS-B',
            'algo_name': 'NNTSBAlgorithm',
            'params': {},
            'model_name': 'NeuralLinearModel',
            'model_params': {},
            'category': '🧠 Neural',
        },
        {
            'name': 'Neural Bandit (Limited Memory)',
            'algo_name': 'NeuralBanditWithLimitedMemory_5',
            'params': {},
            'model_name': 'NeuralLinearModel',
            'model_params': {},
            'category': '🧠 Neural',
        },
        {
            'name': 'Linear Normal (CMAB)',
            'algo_name': 'ThompsonSampling',
            'params': {},
            'model_name': 'LinearNormalModel',
            'model_params': {},
            'category': '📈 Linear',
        },
    ])
    return pd.DataFrame(algorithms)


def make_algo_factory(algo_row, n_arms, feature_dim):
    """Создаёт фабрику алгоритмов для ExperimentRunner"""
    def factory():
        AlgoClass = getattr(algorithms, algo_row['algo_name'])
        init_params = AlgoClass.__init__.__code__.co_varnames

        model = None
        if algo_row['model_name']:
            ModelClass = getattr(models, algo_row['model_name'])
            m_params = dict(algo_row.get('model_params') or {})
            m_init_params = ModelClass.__init__.__code__.co_varnames
            
            model_feature_dim = (n_arms * feature_dim) if algo_row['algo_name'] == 'SGDTSBandit' else feature_dim
            if 'feature_dim' in m_init_params:
                m_params['feature_dim'] = model_feature_dim
            elif 'd' in m_init_params:
                m_params['d'] = model_feature_dim
            elif 'input_dim' in m_init_params:
                m_params['input_dim'] = model_feature_dim
                
            single_model_algos = {
                'PFNTSAlgorithm',
                'NeuralUCBAlgorithm',
                'NNAGPUCBAlgorithm',
                'NNAGPUCBAdaptiveAlgorithm',
                'NNUCBAlgorithm',
                'CustomTSBandit',
                'SGDTSBandit',
            }
            if algo_row['algo_name'] in single_model_algos:
                model = ModelClass(**m_params)
            else:
                model = [ModelClass(**m_params) for _ in range(n_arms)]
        
        a_params = dict(algo_row['params'])
        a_params['n_arms'] = n_arms
        
        # Автоподстановка размерности
        for key in ('x_dim', 'theta_dim', 'd', 'context_dim', 'input_dim', 'n_features'):
            if key in init_params:
                a_params.setdefault(key, feature_dim)
                break
        
        # Фильтрация невалидных параметров (важно!)
        valid_keys = set(init_params)
        a_params = {k: v for k, v in a_params.items() if k in valid_keys}
        
        if model is not None:
            a_params['model'] = model
        
        return AlgoClass(**a_params)
    return factory


def run_online_experiment(env_row, selected_algos, env_params=None, steps=200, n_runs=3, progress_callback=None):
    """Запускает онлайн-эксперимент и возвращает результаты"""
    
    if env_params is None:
        env_params = env_row.get('default_params', {})
    
    # Создаём среду
    EnvClass = getattr(environments, env_row['env_class'])
    env = EnvClass(**env_params)
    
    # n_arms из среды (приоритет) или из env_row
    n_arms = getattr(env, 'n_arms', None) or env_row.get('n_arms') or 10
    
    # feature_dim из первого контекста
    feature_dim = None
    try:
        sample_context = env.get_context()
        if hasattr(sample_context, 'shape'):
            feature_dim = sample_context.shape[-1]
        env.reset()
    except Exception:
        pass
    
    if feature_dim is None:
        feature_dim = getattr(env, 'context_dim', 5)
    
    # Ограничение шагов
    env_T = getattr(env, 'T', None)
    if env_T is not None and steps > env_T:
        steps = env_T
    
    all_results = {}
    
    for _, algo_row in selected_algos.iterrows():
        algo_name = algo_row.get('name', algo_row.get('display_name', 'unknown'))
        
        if progress_callback:
            progress_callback(f"Запуск: {algo_name}...")
        
        algo_factory = make_algo_factory(algo_row, n_arms, feature_dim)
        
        runner = ExperimentRunner(
            env=env,
            algorithm_factory=algo_factory,
            steps=steps,
            n_runs=n_runs,
            seed=42,
        )
        
        try:
            result = runner.run()
            all_results[algo_name] = result
            env.reset()
        except (Exception, SystemExit) as e:
            all_results[algo_name] = {"error": str(e)}
    
    return all_results


def format_results_table(all_results, steps):
    """Форматирует результаты в таблицу"""
    rows = []
    for name, data in all_results.items():
        if 'error' in data:
            rows.append({
                'Алгоритм': name,
                'Cum. Regret': float('nan'),
                'Avg Regret': float('nan'),
                'Время (с)': float('nan'),
                'Статус': '❌ Ошибка'
            })
        else:
            cum = data.get('cumulative_regret_mean', [0])
            avg = data.get('average_regret_mean', [0])
            times = data.get('times_mean', [0])
            
            rows.append({
                'Алгоритм': name,
                'Cum. Regret': round(cum[-1], 2) if len(cum) > 0 else float('nan'),
                'Avg Regret': round(avg[-1], 4) if len(avg) > 0 else float('nan'),
                'Время (с)': round(sum(times), 2) if len(times) > 0 else float('nan'),
                'Статус': '✅ Успех'
            })
    
    df = pd.DataFrame(rows)
    if 'Cum. Regret' in df.columns:
        df = df.sort_values('Cum. Regret', na_position='last')
    return df