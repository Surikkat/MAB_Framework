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
        {
            'name': 'Синтетическая (линейная)',
            'id': 'synthetic_linear',
            'type': 'synthetic',
            'description': 'Линейная среда с гауссовским шумом',
            'env_class': 'SyntheticLinearEnv',
            'default_params': {
                'n_arms': 10,
                'context_dim': 5,
                'noise_std': 0.1,
            },
            'n_arms': 10,
        },
        {
            'name': 'Синтетическая (нейронная)',
            'id': 'synthetic_neural',
            'type': 'synthetic', 
            'description': 'Нелинейная среда с большей размерностью',
            'env_class': 'SyntheticLinearEnv',
            'default_params': {
                'n_arms': 20,
                'context_dim': 10,
                'noise_std': 0.2,
            },
            'n_arms': 20,
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
            'n_arms': 20,
        },
        {
            'name': 'Mushrooms',
            'id': 'mushrooms',
            'type': 'real',
            'description': 'Классификация грибов (съедобный/ядовитый)',
            'env_class': 'DatasetEnvironment',
            'default_params': {
                'dataset_path': str(MAB_PATH / 'mab_framework/data/mushroom_bandit_5000.csv'),
            },
            'n_arms': 20,
        },
        {
            'name': 'E1 Dataset (NPZ)',
            'id': 'e1_dataset',
            'type': 'real',
            'description': 'Синтетический E1 датасет из фреймворка',
            'env_class': 'NPZDatasetEnv',
            'default_params': {
                'dataset_path': str(MAB_PATH / 'mab_framework/data/E1_dataset.npz'),
            },
            'n_arms': 10,
        },
    ]
    return pd.DataFrame(envs)


def get_available_algorithms_for_online():
    """Возвращает алгоритмы, готовые для онлайн-запуска"""
    algos = [
        {
            'name': 'Epsilon-Greedy (ε=0.1)',
            'algo_name': 'EpsilonGreedy',
            'params': {'epsilon': 0.1},
            'model_name': 'OnlineRidgeRegression',
            'model_params': {'l2_reg': 1.0},
            'category': '🚀 Stochastic',
        },
        {
            'name': 'Epsilon-Greedy (ε=0.3)',
            'algo_name': 'EpsilonGreedy',
            'params': {'epsilon': 0.3},
            'model_name': 'OnlineRidgeRegression',
            'model_params': {'l2_reg': 1.0},
            'category': '🚀 Stochastic',
        },
        {
            'name': 'UCB (α=1.0)',
            'algo_name': 'UCBAlgorithm',
            'params': {'alpha': 1.0},
            'model_name': 'OnlineRidgeRegression',
            'model_params': {'l2_reg': 1.0},
            'category': '🚀 Stochastic',
        },
        {
            'name': 'LinUCB (α=0.5)',
            'algo_name': 'LinUCBAlgorithm',
            'params': {'alpha': 0.5},
            'model_name': 'OnlineRidgeRegression',
            'model_params': {'l2_reg': 1.0},
            'category': '🚀 Stochastic',
        },
        {
            'name': 'Thompson Sampling',
            'algo_name': 'ThompsonSampling',
            'params': {},
            'model_name': 'OnlineRidgeRegression',
            'model_params': {'l2_reg': 1.0},
            'category': '🚀 Stochastic',
        },
        {
            'name': 'Bootstrap TS',
            'algo_name': 'BootstrapTSBandit',
            'params': {},
            'model_name': 'BootstrapEnsembleModel',
            'model_params': {'n_models': 10, 'lr': 0.01},
            'category': '🚀 Stochastic',
        },
        {
            'name': 'NonContextual TS',
            'algo_name': 'NonContextualTSBandit',
            'params': {'prior_mean': 0.0, 'prior_var': 1.0, 'reward_var': 1.0},
            'model_name': None,
            'model_params': None,
            'category': '🚀 Stochastic',
        },
        {
            'name': 'GP-UCB (mult kernel)',
            'algo_name': 'GPUCBKernelFlexibleAlgorithm',
            'params': {'beta': 2.0, 'sigma_noise': 0.01, 'kernel_type': 'multiplicative'},
            'model_name': 'GPRFFModel',
            'model_params': {'n_features': 100},
            'category': '🔮 Bayesian',
        },
        {
            'name': 'GP-TS',
            'algo_name': 'GPTSBandit',
            'params': {'n_features': 100, 'kernel_scale': 1.0, 'lengthscale': 1.0},
            'model_name': 'GPRFFModel',
            'model_params': {'n_features': 100},
            'category': '🔮 Bayesian',
        },
        {
            'name': 'Neural UCB',
            'algo_name': 'NeuralUCBAlgorithm',
            'params': {'beta': 2.0},
            'model_name': 'NeuralUCBModel',
            'model_params': {},
            'category': '🧠 Neural',
        },
        {
            'name': 'NN-AGP UCB',
            'algo_name': 'NNAGPUCBAlgorithm',
            'params': {'beta': 2.0},
            'model_name': 'NNAGPModel',
            'model_params': {},
            'category': '🧠 Neural',
        },
        {
            'name': 'PFN-TS (Adaptive TabICL)',
            'algo_name': 'PFNTSAlgorithm',
            'params': {'encoding': 'adaptive', 'alpha': 1.0},
            'model_name': 'TabICLRegressorPPD',
            'model_params': {},
            'category': '🧠 Neural',
        },
        {
            'name': 'FGTS',
            'algo_name': 'FGTSAlgorithm',
            'params': {},
            'model_name': 'FGTSModel',
            'model_params': {},
            'category': '⚡ Special',
        },
        {
            'name': 'FGTS Lasso',
            'algo_name': 'FGTSAlgorithm',
            'params': {},
            'model_name': 'FGTSLassoModel',
            'model_params': {},
            'category': '⚡ Special',
        }
    ]
    
    try:
        import vowpalwabbit
        algorithms.append({
            'name': 'RegCB',
            'algo_name': 'RegCBBandit',
            'params': {},
            'model_name': 'OnlineRidgeRegression',
            'model_params': {'l2_reg': 1.0},
            'category': '⚡ Special',
        })
    except ImportError:
        pass
        
    algorithms.extend([
        {
            'name': 'SGD-TS',
            'algo_name': 'SGDTSBandit',
            'params': {},
            'model_name': 'SGDModel',
            'model_params': {},
            'category': '⚡ Special',
        },
        {
            'name': 'Custom TS (GLM Laplace)',
            'algo_name': 'CustomTSBandit',
            'params': {},
            'model_name': 'GLMLaplaceModel',
            'model_params': {},
            'category': '🔮 Bayesian',
        },
        {
            'name': 'NN-AGP Adaptive',
            'algo_name': 'NNAGPUCBAdaptiveAlgorithm',
            'params': {'beta': 2.0},
            'model_name': 'NNAGPModel',
            'model_params': {},
            'category': '🧠 Neural',
        },
        {
            'name': 'NN-UCB',
            'algo_name': 'NNUCBAlgorithm',
            'params': {'beta': 2.0},
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
            m_params = dict(algo_row['model_params'])
            m_init_params = ModelClass.__init__.__code__.co_varnames
            
            model_feature_dim = (n_arms * feature_dim) if algo_row['algo_name'] == 'SGDTSBandit' else feature_dim
            if 'feature_dim' in m_init_params:
                m_params['feature_dim'] = model_feature_dim
            elif 'd' in m_init_params:
                m_params['d'] = model_feature_dim
            elif 'input_dim' in m_init_params:
                m_params['input_dim'] = model_feature_dim
            elif 'n_features' in m_init_params:
                m_params['n_features'] = model_feature_dim
                
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
        
        if 'x_dim' in init_params:
            a_params['x_dim'] = feature_dim
        if 'theta_dim' in init_params:
            a_params['theta_dim'] = feature_dim
        if 'd' in init_params:
            a_params['d'] = feature_dim
        if 'n_features' in init_params:
            a_params['n_features'] = feature_dim
        if 'context_dim' in init_params:
            a_params['context_dim'] = feature_dim
        if 'input_dim' in init_params:
            a_params['input_dim'] = feature_dim
            
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
    
    n_arms = getattr(env, 'n_arms', env_row.get('n_arms', 10))
    
    # Определяем feature_dim
    feature_dim = 5
    try:
        sample_context = env.get_context()
        feature_dim = sample_context.shape[-1]
        env.reset()
    except Exception:
        pass
    
    all_results = {}
    
    for _, algo_row in selected_algos.iterrows():
        algo_name = algo_row['name']
        
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