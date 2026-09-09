import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Callable
from collections import Counter


_root_dir = str(Path(__file__).resolve().parent.parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)


# Stochastic
from mab_framework.algorithms.stochastic.epsilon_greedy import EpsilonGreedy
from mab_framework.algorithms.stochastic.ucb import UCBAlgorithm
from mab_framework.algorithms.stochastic.thompson_sampling import ThompsonSampling
from mab_framework.algorithms.stochastic.linucb import LinUCBAlgorithm
from mab_framework.algorithms.stochastic.noncontextual_ts_bandit import NonContextualTSBandit
from mab_framework.algorithms.stochastic.bootstrap_ts_bandit import BootstrapTSBandit
from mab_framework.algorithms.stochastic.fgts import FGTSAlgorithm

# Contextual
from mab_framework.algorithms.contextual.gp_ucb_multikernel import GPUCBKernelFlexibleAlgorithm
from mab_framework.algorithms.contextual.gpts_bandit import GPTSBandit
from mab_framework.algorithms.contextual.custom_ts_bandit import CustomTSBandit
from mab_framework.algorithms.contextual.sgd_ts_bandit import SGDTSBandit
from mab_framework.algorithms.contextual.regcb_bandit import RegCBBandit

# Neural
from mab_framework.algorithms.neural.nn_agp_ucb import NNAGPUCBAlgorithm
from mab_framework.algorithms.neural.nn_agp_adaptive import NNAGPUCBAdaptiveAlgorithm
from mab_framework.algorithms.neural.neural_ucb import NeuralUCBAlgorithm
from mab_framework.algorithms.neural.nn_ucb import NNUCBAlgorithm
from mab_framework.algorithms.neural.nn_ts_b import NNTSBAlgorithm
from mab_framework.algorithms.neural.nn_bandit_limited_memory import NeuralBanditWithLimitedMemory_5
from mab_framework.algorithms.neural.pfn_ts import PFNTSAlgorithm

# ВСЕ МОДЕЛИ
#from mab_framework.models.tabicl_model import TabICLRegressorPPD
from mab_framework.models.linear_model import OnlineRidgeRegression
from mab_framework.models.gp_rff_model import GPRFFModel
from mab_framework.models.nn_agp_model import NNAGPModel
from mab_framework.models.neural_network import NeuralLinearModel
from mab_framework.models.neural_ucb_model import NeuralUCBModel
from mab_framework.models.nn_ucb_model import NNUCBModel
from mab_framework.models.glm_laplace_model import GLMLaplaceModel
from mab_framework.algorithms.stochastic.bootstrap_ts_bandit import BootstrapEnsembleModel
from mab_framework.models.cmab_models import LinearNormalModel, GLMNormalModel, NeuralNormalModel
from mab_framework.models.exact_gp_model import ExactGPModel
from mab_framework.models.fgts_model import FGTSModel
from mab_framework.models.fgts_lasso_model import FGTSLassoModel
from mab_framework.models.kernel_ucb_model import KernelUCBModel
from mab_framework.models.sgd_model import SGDModel


class BanditCandidateWrapper:
    def __init__(self, algorithm_class, algorithm_kwargs: dict, 
                 model_class=None, model_kwargs: dict = None):
        self.algorithm_class = algorithm_class
        self.algorithm_kwargs = algorithm_kwargs
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self._propensity_cache = None
        
    def create_offline_propensity_fn(self, df_log: pd.DataFrame) -> Callable:
        
        if self._propensity_cache is not None:
            return self._propensity_cache
        
        df_work = df_log.copy()
        
        n_arms = df_work['item_id'].nunique()
        item_ids = sorted(df_work['item_id'].unique())
        item_to_idx = {item: idx for idx, item in enumerate(item_ids)}
        idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        
        feature_cols = self._prepare_features(df_work)
        context_dim = len(feature_cols)
        
        sort_col = 'hour_of_day' if 'hour_of_day' in df_work.columns else df_work.columns[0]
        df_sorted = df_work.sort_values(sort_col).reset_index(drop=True)

        actual_context_dim = len(feature_cols) + (1 if 'item_price' in df_work.columns else 0) + (1 if 'item_rating' in df_work.columns else 0)
        
        single_model_algos = {
            'PFNTSAlgorithm',
            'NeuralUCBAlgorithm',
            'NNAGPUCBAlgorithm',
            'NNAGPUCBAdaptiveAlgorithm',
            'NNUCBAlgorithm',
            'CustomTSBandit',
            'SGDTSBandit',
        }
        
        model_kwargs = self.model_kwargs.copy()
        m_init_params = self.model_class.__init__.__code__.co_varnames
        actual_model_dim = (n_arms * actual_context_dim) if self.algorithm_class.__name__ == 'SGDTSBandit' else actual_context_dim
        if 'feature_dim' in m_init_params:
            model_kwargs['feature_dim'] = actual_model_dim
        if 'd' in m_init_params:
            model_kwargs['d'] = actual_model_dim
        if 'input_dim' in m_init_params:
            model_kwargs['input_dim'] = actual_model_dim

        if self.algorithm_class.__name__ in single_model_algos:
            models_arg = self.model_class(**model_kwargs)
        else:
            models_arg = [self.model_class(**model_kwargs) for _ in range(n_arms)]

        algo_kwargs = self.algorithm_kwargs.copy()
        algo_kwargs['n_arms'] = n_arms
        algo_kwargs['model'] = models_arg
        
        init_params = self.algorithm_class.__init__.__code__.co_varnames
        if 'x_dim' in init_params:
            algo_kwargs['x_dim'] = actual_context_dim
        if 'theta_dim' in init_params:
            algo_kwargs['theta_dim'] = actual_context_dim
        if 'd' in init_params:
            algo_kwargs['d'] = actual_context_dim
        if 'context_dim' in init_params:
            algo_kwargs['context_dim'] = actual_context_dim
        if 'input_dim' in init_params:
            algo_kwargs['input_dim'] = actual_context_dim
        
        algorithm = self.algorithm_class(**algo_kwargs)

        n_play = min(5000, len(df_sorted))
        step = max(1, len(df_sorted) // n_play)
        
        actions_taken = []
        
        for idx in range(0, len(df_sorted), step):
            if len(actions_taken) >= n_play:
                break
                
            row = df_sorted.iloc[idx]
            user_features = row[feature_cols].fillna(0).values.astype(np.float32)
            
            contexts = []
            for item_id in item_ids:
                item_feats = []
                if 'item_price' in df_work.columns:
                    item_feats.append(df_work[df_work['item_id'] == item_id]['item_price'].iloc[0])
                if 'item_rating' in df_work.columns:
                    item_feats.append(df_work[df_work['item_id'] == item_id]['item_rating'].iloc[0])
                
                context = np.concatenate([user_features, item_feats]) if item_feats else user_features
                contexts.append(context)
            
            context_array = np.array(contexts, dtype=np.float32)
            
            chosen_arm = algorithm.select_arm(context_array)
            actions_taken.append(idx_to_item[chosen_arm])
            
            reward = float(row['reward']) if 'reward' in row and pd.notna(row['reward']) else 1.0
            algorithm.update([{'context': context_array, 'action': int(chosen_arm), 'reward': reward}])

        action_counts = Counter(actions_taken)
        total = len(actions_taken)
        alpha = 0.1
        
        item_propensity = {}
        for item_id in item_ids:
            item_propensity[item_id] = (action_counts.get(item_id, 0) + alpha) / (total + alpha * n_arms)
        
        def propensity_fn(df):
            props = np.array([item_propensity.get(item, 0.01/n_arms) for item in df['item_id'].values])
            return np.clip(props, 0.001, 1.0)
        
        self._propensity_cache = propensity_fn
        return propensity_fn
    
    def _prepare_features(self, df: pd.DataFrame) -> list:
        user_cols = []
        for col in ['user_age', 'user_activity_score', 'user_avg_check', 
                     'user_views_7d', 'user_clicks_7d', 'hour_of_day']:
            if col in df.columns:
                user_cols.append(col)
        
        if 'user_gender' in df.columns:
            df['gender_code'] = (df['user_gender'] == 'F').astype(float)
            user_cols.append('gender_code')
        
        if 'device_type' in df.columns:
            df['device_code'] = df['device_type'].map({'mobile': 0.0, 'desktop': 1.0, 'tablet': 2.0}).fillna(0.0)
            user_cols.append('device_code')
        
        if not user_cols:
            num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ('item_id', 'propensity', 'reward')]
            if num_cols:
                user_cols.append(num_cols[0])
            else:
                df['dummy_feature'] = 1.0
                user_cols.append('dummy_feature')
        return user_cols


class BanditCandidatePool:
    def __init__(self, df_log: pd.DataFrame):
        self.df_log = df_log
        self.n_arms = df_log['item_id'].nunique()
        self.context_dim = self._get_context_dim(df_log)
        
    def _get_context_dim(self, df: pd.DataFrame) -> int:
        dim = 0
        for col in ['user_age', 'user_activity_score', 'user_avg_check', 
                     'user_views_7d', 'user_clicks_7d', 'hour_of_day']:
            if col in df.columns:
                dim += 1
        if 'user_gender' in df.columns:
            dim += 1
        if 'device_type' in df.columns:
            dim += 1
        dim += 2
        return max(dim, 5)
    
    def list_candidates(self) -> pd.DataFrame:
        dim = self.context_dim
        
        candidates = [
            {
                'name': 'Epsilon-Greedy (ε=0.01)',
                'description': '1% exploration — almost pure greedy',
                'category': '🚀 Stochastic (Fast)',
                'complexity': '⭐',
                'wrapper': BanditCandidateWrapper(EpsilonGreedy, {'epsilon': 0.01}, OnlineRidgeRegression, {'feature_dim': dim})
            },
            {
                'name': 'Epsilon-Greedy (ε=0.10)',
                'description': '10% exploration — standard',
                'category': '🚀 Stochastic (Fast)',
                'complexity': '⭐',
                'wrapper': BanditCandidateWrapper(EpsilonGreedy, {'epsilon': 0.10}, OnlineRidgeRegression, {'feature_dim': dim})
            },
            {
                'name': 'Epsilon-Greedy (ε=0.30)',
                'description': '30% exploration — aggressive',
                'category': '🚀 Stochastic (Fast)',
                'complexity': '⭐',
                'wrapper': BanditCandidateWrapper(EpsilonGreedy, {'epsilon': 0.30}, OnlineRidgeRegression, {'feature_dim': dim})
            },
            {
                'name': 'UCB (α=0.5)',
                'description': 'Upper Confidence Bound — conservative',
                'category': '🚀 Stochastic (Fast)',
                'complexity': '⭐⭐',
                'wrapper': BanditCandidateWrapper(UCBAlgorithm, {'alpha': 0.5}, OnlineRidgeRegression, {'feature_dim': dim})
            },
            {
                'name': 'UCB (α=2.0)',
                'description': 'Upper Confidence Bound — exploratory',
                'category': '🚀 Stochastic (Fast)',
                'complexity': '⭐⭐',
                'wrapper': BanditCandidateWrapper(UCBAlgorithm, {'alpha': 2.0}, OnlineRidgeRegression, {'feature_dim': dim})
            },
            {
                'name': 'Thompson Sampling',
                'description': 'Bayesian TS with conjugate priors',
                'category': '🚀 Stochastic (Fast)',
                'complexity': '⭐⭐',
                'wrapper': BanditCandidateWrapper(ThompsonSampling, {}, OnlineRidgeRegression, {'feature_dim': dim})
            },
            {
                'name': 'LinUCB (α=0.5)',
                'description': 'Linear UCB — contextual',
                'category': '🚀 Stochastic (Fast)',
                'complexity': '⭐⭐',
                'wrapper': BanditCandidateWrapper(LinUCBAlgorithm, {'alpha': 0.5}, OnlineRidgeRegression, {'feature_dim': dim})
            },
            {
                'name': 'LinUCB (α=1.0)',
                'description': 'Linear UCB — more exploration',
                'category': '🚀 Stochastic (Fast)',
                'complexity': '⭐⭐',
                'wrapper': BanditCandidateWrapper(LinUCBAlgorithm, {'alpha': 1.0}, OnlineRidgeRegression, {'feature_dim': dim})
            },
            {
                'name': 'NonContextual TS',
                'description': 'Non-contextual Thompson Sampling',
                'category': '🚀 Stochastic (Fast)',
                'complexity': '⭐',
                'wrapper': BanditCandidateWrapper(NonContextualTSBandit, {}, OnlineRidgeRegression, {'feature_dim': dim})
            },
            {
                'name': 'Bootstrap TS',
                'description': 'Bootstrap Thompson Sampling with ensemble',
                'category': '🚀 Stochastic (Fast)',
                'complexity': '⭐⭐',
                'wrapper': BanditCandidateWrapper(BootstrapTSBandit, {}, BootstrapEnsembleModel, {'feature_dim': dim})
            },
            
            {
                'name': 'GP-UCB Kernel (mult)',
                'description': 'GP-UCB with multiplicative kernel',
                'category': '🔮 Bayesian (GP)',
                'complexity': '⭐⭐⭐',
                'wrapper': BanditCandidateWrapper(
                    GPUCBKernelFlexibleAlgorithm,
                    {'beta': 2.0, 'x_dim': dim, 'theta_dim': dim, 'sigma_noise': 0.01, 'kernel_type': 'multiplicative'},
                    GPRFFModel, {'feature_dim': dim * 2, 'n_features': 100}
                )
            },
            {
                'name': 'GP-UCB Kernel (adapt)',
                'description': 'GP-UCB with adaptive kernel weights',
                'category': '🔮 Bayesian (GP)',
                'complexity': '⭐⭐⭐',
                'wrapper': BanditCandidateWrapper(
                    GPUCBKernelFlexibleAlgorithm,
                    {'beta': 2.0, 'x_dim': dim, 'theta_dim': dim, 'sigma_noise': 0.01, 'kernel_type': 'adaptive', 'adaptive_weights': {'theta': 0.7, 'x': 0.3}},
                    GPRFFModel, {'feature_dim': dim * 2, 'n_features': 100}
                )
            },
            {
                'name': 'GP-Thompson Sampling',
                'description': 'Thompson Sampling with GP (RFF approx)',
                'category': '🔮 Bayesian (GP)',
                'complexity': '⭐⭐⭐',
                'wrapper': BanditCandidateWrapper(
                    GPTSBandit, {'d': dim, 'n_features': 100, 'kernel_scale': 1.0, 'lengthscale': 1.0},
                    GPRFFModel, {'feature_dim': dim, 'n_features': 100}
                )
            },
            {
                'name': 'Exact GP',
                'description': 'Exact Gaussian Process (no RFF approximation)',
                'category': '🔮 Bayesian (GP)',
                'complexity': '⭐⭐⭐',
                'wrapper': BanditCandidateWrapper(
                    ThompsonSampling, {},
                    ExactGPModel, {'feature_dim': dim}
                )
            },
            {
                'name': 'Kernel UCB',
                'description': 'Kernel-based UCB',
                'category': '🔮 Bayesian (GP)',
                'complexity': '⭐⭐⭐',
                'wrapper': BanditCandidateWrapper(UCBAlgorithm, {'alpha': 1.0}, KernelUCBModel, {'feature_dim': dim})
            },
            {
                'name': 'Custom TS (GLM Laplace)',
                'description': 'Thompson Sampling with Laplace GLM approximation',
                'category': '🔮 Bayesian (GP)',
                'complexity': '⭐⭐',
                'wrapper': BanditCandidateWrapper(CustomTSBandit, {}, GLMLaplaceModel, {'feature_dim': dim})
            },
            
            {
                'name': 'Neural UCB',
                'description': 'Neural network-based UCB',
                'category': '🧠 Neural (Slow)',
                'complexity': '⭐⭐⭐',
                'wrapper': BanditCandidateWrapper(NeuralUCBAlgorithm, {'beta': 2.0}, NeuralUCBModel, {'feature_dim': dim})
            },
            {
                'name': 'NN-AGP UCB',
                'description': 'Neural Network Augmented GP UCB',
                'category': '🧠 Neural (Slow)',
                'complexity': '⭐⭐⭐⭐',
                'wrapper': BanditCandidateWrapper(NNAGPUCBAlgorithm, {'beta': 2.0}, NNAGPModel, {'feature_dim': dim})
            },
            {
                'name': 'NN-AGP Adaptive',
                'description': 'NN-AGP with adaptive uncertainty',
                'category': '🧠 Neural (Slow)',
                'complexity': '⭐⭐⭐⭐',
                'wrapper': BanditCandidateWrapper(NNAGPUCBAdaptiveAlgorithm, {'beta': 2.0}, NNAGPModel, {'feature_dim': dim})
            },
            {
                'name': 'NN-UCB',
                'description': 'Deep Neural UCB',
                'category': '🧠 Neural (Slow)',
                'complexity': '⭐⭐⭐⭐',
                'wrapper': BanditCandidateWrapper(NNUCBAlgorithm, {'beta': 2.0}, NNUCBModel, {'feature_dim': dim})
            },
            {
                'name': 'NN-TS-B',
                'description': 'Neural Network Thompson Sampling (B)',
                'category': '🧠 Neural (Slow)',
                'complexity': '⭐⭐⭐⭐',
                'wrapper': BanditCandidateWrapper(NNTSBAlgorithm, {}, NeuralLinearModel, {'feature_dim': dim})
            },
            {
                'name': 'Neural Bandit (Limited Memory)',
                'description': 'Neural Bandit with limited memory buffer',
                'category': '🧠 Neural (Slow)',
                'complexity': '⭐⭐⭐⭐⭐',
                'wrapper': BanditCandidateWrapper(NeuralBanditWithLimitedMemory_5, {}, NeuralLinearModel, {'feature_dim': dim})
            },
            # {
            #     'name': 'PFN-TS (Adaptive TabICL)',
            #     'description': 'Thompson Sampling через Universal Subsampling CLT (PFN TabICL)',
            #     'category': '🧠 Neural (Slow)',
            #     'complexity': '⭐⭐⭐⭐⭐',
            #     'wrapper': BanditCandidateWrapper(
            #         PFNTSAlgorithm,
            #         {'encoding': 'adaptive', 'alpha': 1.0},
            #         TabICLRegressorPPD,
            #         {}
            #     )
            # },
            
            # SPECIAL — 5 алгоритмов
            {
                'name': 'FGTS',
                'description': 'Fast Greedy Thompson Sampling',
                'category': '⚡ Special',
                'complexity': '⭐⭐',
                'wrapper': BanditCandidateWrapper(FGTSAlgorithm, {}, FGTSModel, {'feature_dim': dim})
            },
            {
                'name': 'FGTS Lasso',
                'description': 'FGTS with Lasso regularization',
                'category': '⚡ Special',
                'complexity': '⭐⭐',
                'wrapper': BanditCandidateWrapper(FGTSAlgorithm, {}, FGTSLassoModel, {'feature_dim': dim})
            }
        ]
        
        try:
            import vowpalwabbit
            candidates.append({
                'name': 'RegCB',
                'description': 'Regression-based Confidence Bound',
                'category': '⚡ Special',
                'complexity': '⭐⭐',
                'wrapper': BanditCandidateWrapper(RegCBBandit, {}, OnlineRidgeRegression, {'feature_dim': dim})
            })
        except ImportError:
            pass
            
        candidates.extend([
            {
                'name': 'SGD-TS',
                'description': 'Stochastic Gradient Descent Thompson Sampling',
                'category': '⚡ Special',
                'complexity': '⭐⭐⭐',
                'wrapper': BanditCandidateWrapper(SGDTSBandit, {}, SGDModel, {'feature_dim': dim})
            },
            {
                'name': 'Linear Normal (CMAB)',
                'description': 'Linear Normal model for Contextual MAB',
                'category': '⚡ Special',
                'complexity': '⭐⭐',
                'wrapper': BanditCandidateWrapper(ThompsonSampling, {}, LinearNormalModel, {'feature_dim': dim})
            },
        ])
        
        return pd.DataFrame(candidates)