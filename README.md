# Modular Multi-Armed Bandit (MAB) Framework

Унифицированный модульный фреймворк для разработки, тестирования и бенчмаркинга алгоритмов контекстных многоруких бандитов (Contextual Multi-Armed Bandits). 

Фреймворк объединяет SOTA-алгоритмы и модели из различных научных статей, предоставляя стандартизированный пайплайн для их честного сравнения на единых наборах данных (как синтетических, так и реальных).

## ✨ Ключевые возможности

* **Строгая модульность (Decoupled Architecture):** Стратегии выбора (Exploration/Exploitation) полностью отделены от прогностических моделей. Вы можете комбинировать любую модель (Linear, Neural, GP) с любым алгоритмом (UCB, Thompson Sampling) "из коробки".
* **Config-driven эксперименты:** Полная настройка среды, моделей и алгоритмов через YAML-конфигурации. Запуск экспериментов не требует написания кода.
* **Нативная поддержка Delayed Feedback:** Архитектура симулятора среды поддерживает задержки в получении наград (настраивается через очередь событий).
* **Воспроизводимость (Reproducibility):** Фиксация seed для всех библиотек, стандартизированное логирование метрик (Cumulative/Average Regret, Runtime) и встроенная генерация графиков.
* **Готовые бенчмарки:** Включены имплементации алгоритмов из научных статей: FGTS-LASSO, NN-AGP, NeuralUCB, GP-TS, GLM-TS и др.

---

## 🏗 Архитектура

Фреймворк строится на 4 независимых компонентах:

1. **Environments (`environments/`)** — симуляторы среды (например, `DatasetEnvironment`). Отвечают за выдачу контекстов (признаков), сокрытие истинных наград и симуляцию задержек (Delayed Feedback).
2. **Models (`models/`)** — аппроксимирующие модели. Оценивают ожидаемую награду и неопределенность (variance) для конкретного контекста.
   * *Примеры:* `OnlineRidgeRegression`, `NeuralLinearModel`, `ExactGPModel`, `GLMLaplaceModel`.
3. **Algorithms (`algorithms/`)** — стратегии многоруких бандитов, реализующие баланс исследования и использования.
   * *Примеры:* `UCBAlgorithm`, `ThompsonSampling`, `EpsilonGreedy`, `NeuralUCBAlgorithm`.
4. **Experiment Runner (`experiment/`)** — движок для связывания среды и алгоритмов, прогона итераций, сбора метрик и сохранения результатов.

---

## 🚀 Установка

Рекомендуется использовать Python 3.10+. 

Клонируйте репозиторий и установите необходимые зависимости:

```bash
git clone https://github.com/your-repo/MAB_Framework.git
cd MAB_Framework
pip install -r requirements.txt
```

*(Основные зависимости: `numpy`, `pandas`, `scikit-learn`, `torch`, `scipy`, `matplotlib`, `pyyaml`, `pydantic`)*.

---

## 📊 Быстрый старт

### Вариант 1: Запуск через YAML-конфиг (No-code)
Настройте эксперимент в файле `configs/test.yaml`:

```yaml
experiment:
  name: "demo_experiment"
  steps: 1000
  n_runs: 5
  seed: 42

environment:
  name: "DatasetEnvironment"
  params:
    dataset_path: "data/mushroom_bandit_5000.csv"

algorithms:
  - name: "UCBAlgorithm"
    display_name: "LinUCB"
    params: { alpha: 1.0 }
    model:
      name: "OnlineRidgeRegression"
      params: { l2_reg: 1.0 }
      
metrics:
  - cumulative_regret
  - average_regret
```

Запустите эксперимент через CLI:
```bash
python run.py configs/test.yaml
```
Результаты (JSON-логи и графики `.png`) будут сохранены в директорию `results/`.

### Вариант 2: Запуск через Python API

Для более сложных пайплайнов или отладки используйте Python API:

```python
from environments.dataset_env import DatasetEnvironment
from models.neural_network import NeuralLinearModel
from algorithms.ucb import UCBAlgorithm
from experiment.runner import ExperimentRunner

# 1. Инициализация среды
env = DatasetEnvironment(dataset_path="data/mushroom_bandit_5000.csv", max_steps=1000)

# 2. Определение фабрики алгоритма (создает свежие инстансы для каждого seed-прогона)
def algo_factory():
    n_arms = env.n_arms
    feature_dim = env.get_context().shape[1]
    models =[NeuralLinearModel(feature_dim=feature_dim, hidden_dim=64) for _ in range(n_arms)]
    return UCBAlgorithm(n_arms=n_arms, model=models, alpha=1.0)

# 3. Запуск эксперимента
runner = ExperimentRunner(
    env=env,
    algorithm_factory=algo_factory,
    steps=env.T,
    n_runs=5,
    output_file="results/neural_ucb_results.json"
)
runner.run()
```

---

## 🔬 Воспроизведение бенчмарков (Reproducibility)

В папке `scripts/` находятся готовые пайплайны для воспроизведения результатов из имплементированных научных статей:

* `reproduce_fgts_lasso.py` — бенчмаркинг Feature Gated Thompson Sampling (FGTS-LASSO) vs LinUCB.
* `reproduce_nn_agp.py` — бенчмаркинг алгоритма Neural-Network Approximate Gaussian Process UCB.
* `reproduce_cmab_bandits.py` — сравнение аппроксимационных методов (GP-TS-RFF, GLM-TS-Laplace).
* `run_mushrooms.py` — **глобальный бенчмарк**. Сравнение всех реализованных алгоритмов (линейных, нейросетевых и GP) на реальном датасете UCI Mushrooms.

Запуск любого скрипта:
```bash
python scripts/run_mushrooms.py
```

---

## 🛠 Подключение собственных данных

Фреймворк позволяет легко тестировать алгоритмы на ваших данных без изменения исходного кода.

1. Подготовьте `CSV` файл. Обязательные колонки:
   * `t` — номер шага/итерации (int).
   * `arm` — идентификатор/номер ручки (int).
   * `reward` — полученная награда (float, оптимальная награда на шаге `t` должна быть максимальной).
   * Все остальные колонки, начинающиеся с `context_` (или любые другие, кроме служебных), автоматически интерпретируются как признаки (features).
2. Передайте путь к файлу в `DatasetEnvironment`:

```python
env = DatasetEnvironment(dataset_path="path/to/custom_dataset.csv")
```

---

## 📄 Структура директорий

* `algorithms/` — алгоритмы (балансировка Exploration-Exploitation).
* `models/` — модели для оценки `predict()` (mu, sigma) или `sample()`.
* `environments/` — интерфейсы сред и парсеры данных.
* `experiment/` — инструменты логирования, метрики и базовый цикл (Runner).
* `scripts/` — готовые сценарии тестирования.
* `configs/` — конфигурации YAML.
* `data/` — датасеты для тестирования.
* `tests/` — `pytest` тесты для проверки целостности ядра (Delayed Feedback, валидация и т.д.).