import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

obp = pytest.importorskip("obp")

from mab_framework.environments.open_bandit_env import OpenBanditEnvironment
from mab_framework.models.linear_model import OnlineRidgeRegression
from mab_framework.algorithms.stochastic.ucb import UCBAlgorithm
from mab_framework.algorithms.stochastic.thompson_sampling import ThompsonSampling
from mab_framework.experiment.runner import ExperimentRunner


@pytest.fixture(scope="module")
def env():
    return OpenBanditEnvironment(
        behavior_policy="random", campaign="all",
        reward_mode="replay", max_steps=500
    )


def test_constructor(env):
    assert env.n_arms > 0
    assert env.T == 500
    assert env.contexts.shape[0] >= 500
    assert env.logged_actions.shape[0] >= 500
    assert env.rewards.shape[0] >= 500


def test_reset(env):
    env.get_context()
    env.reset()
    assert env.current_idx == 0


def test_get_context_shape(env):
    env.reset()
    ctx = env.get_context()
    assert ctx.ndim == 2
    assert ctx.shape[0] == env.n_arms
    feature_dim = env.contexts.shape[1]
    assert ctx.shape[1] == feature_dim


def test_replay_matching_action(env):
    env.reset()
    logged_action = int(env.logged_actions[0])
    logged_reward = float(env.rewards[0])
    result = env.step(logged_action)
    assert result["instant_reward"] == logged_reward


def test_replay_mismatched_action(env):
    env.reset()
    logged_action = int(env.logged_actions[0])
    wrong_action = (logged_action + 1) % env.n_arms
    result = env.step(wrong_action)
    assert result["instant_reward"] == 0.0
    assert result["optimal_reward"] == 0.0


def test_runner_integration(env):
    env.reset()
    n_arms = env.n_arms

    sample_ctx = env.get_context()
    feature_dim = sample_ctx.shape[1]
    env.reset()

    def algo_factory():
        models = [OnlineRidgeRegression(feature_dim=feature_dim) for _ in range(n_arms)]
        return UCBAlgorithm(n_arms=n_arms, model=models, alpha=1.0)

    runner = ExperimentRunner(
        env=env,
        algorithm_factory=algo_factory,
        steps=100,
        n_runs=1,
    )
    result = runner.run()
    assert "cumulative_regret_mean" in result
    assert len(result["cumulative_regret_mean"]) == 100


def test_zero_reward_mode():
    env = OpenBanditEnvironment(
        behavior_policy="random", campaign="all",
        reward_mode="zero", max_steps=100
    )
    logged_action = int(env.logged_actions[0])
    logged_reward = float(env.rewards[0])
    wrong_action = (logged_action + 1) % env.n_arms
    result = env.step(wrong_action)
    assert result["instant_reward"] == 0.0
    assert result["optimal_reward"] == logged_reward
