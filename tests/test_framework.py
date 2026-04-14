import pytest
import numpy as np
import yaml
import tempfile
import os
from pathlib import Path

# Импорты из твоего фреймворка
from environments.base import DelayedFeedbackBuffer
from environments.dataset_env import NPZDatasetEnv
from models.linear_model import OnlineRidgeRegression
from algorithms.ucb import UCBAlgorithm
from experiment.runner import ExperimentRunner
from experiment.config_models import ExperimentConfig

# ---------------------------------------------------------
# 1. ТЕСТЫ ЯДРА: Delayed Feedback Buffer
# ---------------------------------------------------------
def test_delayed_feedback_buffer_no_delay():
    buffer = DelayedFeedbackBuffer()
    # Награда без задержки
    buffer.add(action=1, reward=10.0, delay=0, context=np.array([1, 0]))
    
    matured = buffer.step()
    assert len(matured) == 1
    assert matured[0]["action"] == 1
    assert matured[0]["reward"] == 10.0
    
def test_delayed_feedback_buffer_with_delay():
    buffer = DelayedFeedbackBuffer()
    # Награда придет через 2 шага (delay = 2)
    buffer.add(action=2, reward=5.0, delay=2, context=np.array([0, 1]))
    
    # Шаг 0 (текущий, куда мы добавили)
    assert len(buffer.step()) == 0
    # Шаг 1
    assert len(buffer.step()) == 0
    # Шаг 2 (награда должна созреть)
    matured = buffer.step()
    assert len(matured) == 1
    assert matured[0]["reward"] == 5.0

# ---------------------------------------------------------
# 2. ТЕСТЫ МОДЕЛЕЙ И АЛГОРИТМОВ (Decoupling check)
# ---------------------------------------------------------
def test_online_ridge_regression_shapes():
    feature_dim = 3
    model = OnlineRidgeRegression(feature_dim=feature_dim)
    
    x = np.array([1.0, 0.5, -0.5])
    model.fit(x, y=1.0)
    
    mu, sigma = model.predict(x)
    assert isinstance(mu, np.ndarray) and len(mu) == 1
    assert isinstance(sigma, np.ndarray) and len(sigma) == 1
    # После одного сэмпла mu не должен быть NaN
    assert not np.isnan(mu[0])

def test_ucb_algorithm_selection():
    n_arms = 2
    feature_dim = 3
    # Создаем 2 независимые модели для 2 рук
    models =[OnlineRidgeRegression(feature_dim=feature_dim) for _ in range(n_arms)]
    algo = UCBAlgorithm(n_arms=n_arms, model=models, alpha=1.0)
    
    context = np.array([
        [1.0, 0.0, 0.0], # Контекст ручки 0
        [0.0, 1.0, 0.0]  # Контекст ручки 1
    ])
    
    action = algo.select_arm(context)
    assert action in [0, 1]
    
    # Симулируем фидбек
    feedbacks =[{"action": action, "reward": 1.0, "context": context}]
    algo.update(feedbacks)
    
    # Проверяем, что модель выбранной ручки обновила t (счетчик)
    assert models[action].t == 1
    assert models[1 - action].t == 0 # Вторая ручка не обновлялась

# ---------------------------------------------------------
# 3. ТЕСТЫ ИНТЕГРАЦИИ (Experiment Runner)
# ---------------------------------------------------------
class DummyEnv:
    """Простая мок-среда для тестов runner'a"""
    def __init__(self):
        self.T = 10
        self.n_arms = 2
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        
    def get_context(self):
        return np.ones((self.n_arms, 2))
        
    def step(self, action):
        self.current_step += 1
        reward = 1.0 if action == 0 else 0.0
        # Возвращаем в формате, который ожидает runner
        return {
            "available_rewards":[{"action": action, "reward": reward, "context": self.get_context()}],
            "instant_reward": reward,
            "optimal_reward": 1.0
        }

def test_experiment_runner():
    env = DummyEnv()
    
    def algo_factory():
        models =[OnlineRidgeRegression(feature_dim=2) for _ in range(env.n_arms)]
        return UCBAlgorithm(n_arms=env.n_arms, model=models)
        
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        runner = ExperimentRunner(
            env=env,
            algorithm_factory=algo_factory,
            steps=5,
            n_runs=2,
            output_file=tmp.name
        )
        runner.run()
        
        # Проверяем, что файл создался и не пустой
        assert os.path.exists(tmp.name)
        file_size = os.path.getsize(tmp.name)
        assert file_size > 0
        os.remove(tmp.name)

# ---------------------------------------------------------
# 4. ТЕСТ ВАЛИДАЦИИ КОНФИГОВ (Pydantic)
# ---------------------------------------------------------
def test_config_validation():
    yaml_content = """
    experiment:
      name: "test"
      steps: 10
    environment:
      name: "DatasetEnvironment"
      params:
        dataset_path: "fake.csv"
    algorithms:
      - name: "LinUCBAlgorithm"
        params: { alpha: 1.0 }
    """
    config_dict = yaml.safe_load(yaml_content)
    # Pydantic должен успешно распарсить
    config = ExperimentConfig(**config_dict)
    
    assert config.experiment.name == "test"
    assert config.environment.name == "DatasetEnvironment"
    assert len(config.algorithms) == 1

def test_config_validation_fails_missing_env():
    yaml_content = """
    experiment:
      name: "test"
    algorithms:
      - name: "LinUCBAlgorithm"
    """
    config_dict = yaml.safe_load(yaml_content)
    
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ExperimentConfig(**config_dict)
