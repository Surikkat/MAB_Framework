"""
Реестр настраиваемых гиперпараметров для каждого алгоритма.
Используется для генерации Streamlit-виджетов в Online и Offline режимах.
"""
import streamlit as st


ALGO_HYPERPARAMS = {
    # ═══════════════════════════════════════════
    # 🚀 Stochastic
    # ═══════════════════════════════════════════
    'EpsilonGreedy': {
        'algo_params': [
            {'key': 'epsilon', 'label': 'ε (exploration rate)', 'type': 'number',
             'min': 0.01, 'max': 1.0, 'default': 0.1, 'step': 0.01},
        ],
        'model_params': [
            {'key': 'l2_reg', 'label': 'L2 regularization (λ)', 'type': 'number',
             'default': 1.0, 'min': 0.01, 'step': 0.1},
        ],
    },
    'UCBAlgorithm': {
        'algo_params': [
            {'key': 'alpha', 'label': 'α (exploration coef)', 'type': 'number',
             'min': 0.1, 'max': 5.0, 'default': 1.0, 'step': 0.1},
        ],
        'model_params': [
            {'key': 'l2_reg', 'label': 'L2 regularization (λ)', 'type': 'number',
             'default': 1.0, 'min': 0.01, 'step': 0.1},
        ],
    },
    'LinUCBAlgorithm': {
        'algo_params': [
            {'key': 'alpha', 'label': 'α (exploration coef)', 'type': 'number',
             'min': 0.01, 'max': 5.0, 'default': 0.5, 'step': 0.01},
        ],
        'model_params': [
            {'key': 'l2_reg', 'label': 'L2 regularization (λ)', 'type': 'number',
             'default': 1.0, 'min': 0.01, 'step': 0.1},
        ],
    },
    'ThompsonSampling': {
        'algo_params': [],
        'model_params': [
            {'key': 'nu', 'label': 'ν (variance scale)', 'type': 'number',
             'min': 0.01, 'max': 5.0, 'default': 1.0, 'step': 0.01},
            {'key': 'l2_reg', 'label': 'L2 regularization (λ)', 'type': 'number',
             'default': 1.0, 'min': 0.01, 'step': 0.1},
        ],
    },
    'BootstrapTSBandit': {
        'algo_params': [],
        'model_params': [
            {'key': 'n_models', 'label': 'Ensemble size', 'type': 'number',
             'min': 3, 'max': 30, 'default': 10, 'step': 1},
            {'key': 'lr', 'label': 'Learning rate', 'type': 'number',
             'default': 0.01, 'min': 0.001, 'step': 0.001, 'format': '%.4f'},
        ],
    },
    'NonContextualTSBandit': {
        'algo_params': [
            {'key': 'prior_var', 'label': 'Prior variance', 'type': 'number',
             'default': 1.0, 'min': 0.01, 'step': 0.1},
            {'key': 'reward_var', 'label': 'Reward variance', 'type': 'number',
             'default': 1.0, 'min': 0.01, 'step': 0.1},
        ],
        'model_params': [],
    },

    # ═══════════════════════════════════════════
    # 🔮 Bayesian
    # ═══════════════════════════════════════════
    'GPUCBKernelFlexibleAlgorithm': {
        'algo_params': [
            {'key': 'beta', 'label': 'β (exploration coef)', 'type': 'number',
             'min': 0.1, 'max': 10.0, 'default': 2.0, 'step': 0.1},
            {'key': 'sigma_noise', 'label': 'σ_noise', 'type': 'number',
             'default': 0.01, 'min': 0.001, 'step': 0.001, 'format': '%.4f'},
            {'key': 'kernel_type', 'label': 'Kernel type', 'type': 'selectbox',
             'options': ['multiplicative', 'additive', 'rbf'], 'default': 'multiplicative'},
        ],
        'model_params': [],
    },
    'GPTSBandit': {
        'algo_params': [
            {'key': 'kernel_scale', 'label': 'Kernel scale', 'type': 'number',
             'default': 1.0, 'min': 0.01, 'step': 0.1},
            {'key': 'lengthscale', 'label': 'Lengthscale', 'type': 'number',
             'default': 1.0, 'min': 0.01, 'step': 0.1},
            {'key': 'n_features', 'label': 'RFF features', 'type': 'number',
             'min': 50, 'max': 500, 'default': 100, 'step': 50},
        ],
        'model_params': [],
    },
    'CustomTSBandit': {
        'algo_params': [],
        'model_params': [
            {'key': 'prior_var', 'label': 'Prior variance', 'type': 'number',
             'default': 1.0, 'min': 0.01, 'step': 0.1},
            {'key': 'lr', 'label': 'Learning rate', 'type': 'number',
             'default': 0.1, 'min': 0.001, 'step': 0.01, 'format': '%.4f'},
        ],
    },

    # ═══════════════════════════════════════════
    # 🧠 Neural
    # ═══════════════════════════════════════════
    'NeuralUCBAlgorithm': {
        'algo_params': [
            {'key': 'nu', 'label': 'ν (exploration scale)', 'type': 'number',
             'min': 0.01, 'max': 5.0, 'default': 1.0, 'step': 0.01},
        ],
        'model_params': [],
    },
    'NNAGPUCBAlgorithm': {
        'algo_params': [
            {'key': 'beta', 'label': 'β (exploration coef)', 'type': 'number',
             'min': 0.1, 'max': 10.0, 'default': 2.0, 'step': 0.1},
        ],
        'model_params': [],
    },
    'NNAGPUCBAdaptiveAlgorithm': {
        'algo_params': [
            {'key': 'beta', 'label': 'β (exploration coef)', 'type': 'number',
             'min': 0.1, 'max': 10.0, 'default': 2.0, 'step': 0.1},
        ],
        'model_params': [],
    },
    'NNUCBAlgorithm': {
        'algo_params': [
            {'key': 'beta', 'label': 'β (exploration coef)', 'type': 'number',
             'min': 0.1, 'max': 10.0, 'default': 2.0, 'step': 0.1},
            {'key': 'lambda_', 'label': 'λ (regularization)', 'type': 'number',
             'default': 1.0, 'min': 0.01, 'step': 0.1},
        ],
        'model_params': [],
    },
    'NNTSBAlgorithm': {
        'algo_params': [
            {'key': 'v', 'label': 'ν (exploration scale)', 'type': 'number',
             'min': 0.01, 'max': 2.0, 'default': 0.1, 'step': 0.01},
            {'key': 'lr', 'label': 'Learning rate', 'type': 'number',
             'default': 0.01, 'min': 0.001, 'step': 0.001, 'format': '%.4f'},
        ],
        'model_params': [],
    },
    'NeuralBanditWithLimitedMemory_5': {
        'algo_params': [
            {'key': 'lr', 'label': 'Learning rate', 'type': 'number',
             'default': 0.001, 'min': 0.0001, 'step': 0.0001, 'format': '%.4f'},
            {'key': 'lambda_prior', 'label': 'λ prior', 'type': 'number',
             'default': 0.1, 'min': 0.001, 'step': 0.01, 'format': '%.4f'},
        ],
        'model_params': [],
    },
    'PFNTSAlgorithm': {
        'algo_params': [
            {'key': 'encoding', 'label': 'Encoding', 'type': 'selectbox',
             'options': ['adaptive', 'disjoint', 'one_hot', 'block'], 'default': 'adaptive'},
            {'key': 'alpha', 'label': 'α (exploration scale)', 'type': 'number',
             'min': 0.1, 'max': 5.0, 'default': 1.0, 'step': 0.1},
        ],
        'model_params': [],
    },

    # ═══════════════════════════════════════════
    # ⚡ Special
    # ═══════════════════════════════════════════
    'FGTSAlgorithm': {
        'algo_params': [],
        'model_params': [
            {'key': 'sigma_noise', 'label': 'σ_noise', 'type': 'number',
             'default': 0.1, 'min': 0.001, 'step': 0.01, 'format': '%.4f'},
            {'key': 'sigma_prior', 'label': 'σ_prior', 'type': 'number',
             'default': 1.0, 'min': 0.01, 'step': 0.1},
        ],
    },
    'SGDTSBandit': {
        'algo_params': [],
        'model_params': [
            {'key': 'nu', 'label': 'ν (exploration scale)', 'type': 'number',
             'min': 0.01, 'max': 2.0, 'default': 0.1, 'step': 0.01},
            {'key': 'lr', 'label': 'Learning rate', 'type': 'number',
             'default': 0.01, 'min': 0.001, 'step': 0.001, 'format': '%.4f'},
        ],
    },
    'RegCBBandit': {
        'algo_params': [],
        'model_params': [
            {'key': 'l2_reg', 'label': 'L2 regularization (λ)', 'type': 'number',
             'default': 1.0, 'min': 0.01, 'step': 0.1},
        ],
    },
}


def render_hyperparams(algo_name: str, unique_key: str) -> dict:
    """
    Рендерит Streamlit-виджеты для гиперпараметров алгоритма.
    
    Args:
        algo_name: Имя класса алгоритма (например, 'EpsilonGreedy')
        unique_key: Уникальный ключ для Streamlit виджетов (чтобы не было коллизий)
    
    Returns:
        {'algo_params': {key: value, ...}, 'model_params': {key: value, ...}}
    """
    spec = ALGO_HYPERPARAMS.get(algo_name)
    if not spec:
        return {'algo_params': {}, 'model_params': {}}

    all_params = spec.get('algo_params', []) + spec.get('model_params', [])
    if not all_params:
        return {'algo_params': {}, 'model_params': {}}

    algo_result = {}
    model_result = {}

    # Определяем количество колонок (макс 3)
    n_cols = min(len(all_params), 3)
    cols = st.columns(n_cols)

    for i, param in enumerate(all_params):
        col = cols[i % n_cols]
        widget_key = f"hp_{unique_key}_{param['key']}"

        with col:
            if param['type'] == 'number':
                value = st.number_input(
                    param['label'],
                    value=param['default'],
                    min_value=param.get('min', None),
                    step=param.get('step', 0.1),
                    format=param.get('format', '%.2f'),
                    key=widget_key,
                )
            elif param['type'] == 'selectbox':
                options = param['options']
                default_idx = options.index(param['default']) if param['default'] in options else 0
                value = st.selectbox(
                    param['label'],
                    options=options,
                    index=default_idx,
                    key=widget_key,
                )
            else:
                value = param['default']

        # Распределяем по algo_params / model_params
        is_model_param = param in spec.get('model_params', [])
        if is_model_param:
            model_result[param['key']] = value
        else:
            algo_result[param['key']] = value

    return {'algo_params': algo_result, 'model_params': model_result}
