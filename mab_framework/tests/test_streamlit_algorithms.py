import pytest
import numpy as np
import pandas as pd
import sys
import os

# Добавляем корневую директорию и streamlit_bandit в sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
streamlit_dir = os.path.join(root_dir, 'streamlit_bandit')
if streamlit_dir not in sys.path:
    sys.path.insert(0, streamlit_dir)

from streamlit_bandit.core.bandit_candidates import BanditCandidatePool
from streamlit_bandit.core.online_experiment import get_available_algorithms_for_online, make_algo_factory
from streamlit_bandit.core.evaluator import OPEEvaluator


def _create_dummy_dataframe():
    np.random.seed(42)
    n_rows = 12
    return pd.DataFrame({
        'item_id': np.random.choice([1, 2, 3], size=n_rows),
        'propensity': np.random.uniform(0.1, 0.5, size=n_rows),
        'reward': np.random.choice([0.0, 1.0], size=n_rows),
        'user_age': np.random.uniform(18, 60, size=n_rows),
        'user_activity_score': np.random.uniform(0, 1, size=n_rows),
        'user_avg_check': np.random.uniform(10, 100, size=n_rows),
        'user_views_7d': np.random.uniform(0, 50, size=n_rows),
        'user_clicks_7d': np.random.uniform(0, 20, size=n_rows),
    })


def test_streamlit_offline_candidates():
    """Тестирует инициализацию и расчет offline propensity для всех кандидатов Streamlit."""
    df = _create_dummy_dataframe()
    pool = BanditCandidatePool(df)
    cands_df = pool.list_candidates()
    
    assert len(cands_df) > 0, "BanditCandidatePool должен возвращать непустой список кандидатов"
    
    success_count = 0
    for idx, row in cands_df.iterrows():
        cand_name = row['name']
        # Если алгоритм требует внешних C++ библиотек, которые могут быть не установлены
        if 'Vowpal' in cand_name or 'Open Bandit' in cand_name or 'RegCB' in cand_name:
            try:
                fn = row['wrapper'].create_offline_propensity_fn(df)
                res = fn(df)
                success_count += 1
            except BaseException as e:
                # Если упало на внешнем пакете (или если нет vowpalwabbit), пропускаем
                if 'vowpal' in str(e).lower() or 'obp' in str(e).lower() or 'coba' in str(e).lower() or 'requires the' in str(e).lower():
                    continue
                raise e
        else:
            fn = row['wrapper'].create_offline_propensity_fn(df)
            res = fn(df)
            assert isinstance(res, np.ndarray), f"Результат для {cand_name} должен быть numpy-массивом"
            assert len(res) == len(df), f"Длина результата для {cand_name} должна совпадать с числом строк df"
            assert not np.isnan(res).any(), f"Результат для {cand_name} содержит NaN"
            success_count += 1
            
    print(f"Успешно проверено {success_count} offline кандидатов из {len(cands_df)}.")


def test_streamlit_online_algorithms():
    """Тестирует инициализацию, выбор руки и обновление в online-режиме для всех доступных алгоритмов."""
    online_algos = get_available_algorithms_for_online()
    assert len(online_algos) > 0, "Список доступных online алгоритмов не должен быть пустым"
    
    n_arms = 3
    feature_dim = 4
    
    for idx, algo_cfg in online_algos.iterrows():
        algo_name = algo_cfg['algo_name']
        model_name = algo_cfg['model_name']
        
        # Если алгоритм требует внешних C++ библиотек (например, RegCB)
        if 'RegCB' in algo_name or 'Vowpal' in algo_name or 'OpenBandit' in algo_name:
            try:
                factory = make_algo_factory(algo_cfg, n_arms=n_arms, feature_dim=feature_dim)
                algo = factory()
            except BaseException as e:
                if 'vowpal' in str(e).lower() or 'coba' in str(e).lower() or 'requires the' in str(e).lower():
                    continue
                raise e
        else:
            factory = make_algo_factory(algo_cfg, n_arms=n_arms, feature_dim=feature_dim)
            algo = factory()
            assert algo is not None, f"Фабрика не смогла создать {algo_name}"
            
            # Делаем 2 шага взаимодействия (select_arm + update)
            for _ in range(2):
                context = np.random.randn(n_arms, feature_dim).astype(np.float32)
                chosen_arm = algo.select_arm(context)
                assert 0 <= chosen_arm < n_arms, f"Выбранная рука {chosen_arm} вне диапазона [0, {n_arms}) для {algo_name}"
                
                feedback = [{'context': context, 'action': int(chosen_arm), 'reward': 1.0}]
                algo.update(feedback)
