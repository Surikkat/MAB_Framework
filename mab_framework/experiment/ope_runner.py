import json
import os
import random
import time
import numpy as np

from mab_framework.experiment.ope_evaluator import OPEEvaluator


class OPERunner:
    def __init__(self, env, algorithm_factory, estimator_names,
                 train_ratio=0.7, n_runs=1, seed=None,
                 save_dir=None, metadata=None):
        self.env = env
        self.algorithm_factory = algorithm_factory
        self.estimator_names = estimator_names if isinstance(estimator_names, list) else [estimator_names]
        self.train_ratio = train_ratio
        self.n_runs = n_runs
        self.seed = seed
        self.save_dir = save_dir
        self.metadata = metadata or {}

    def _split_feedback(self):
        n_total = len(self.env.contexts)
        n_train = int(n_total * self.train_ratio)

        train = {
            "contexts": self.env.contexts[:n_train],
            "actions": self.env.logged_actions[:n_train],
            "rewards": self.env.rewards[:n_train],
        }
        test = {
            "n_rounds": n_total - n_train,
            "n_actions": self.env.n_arms,
            "action": self.env.logged_actions[n_train:],
            "reward": self.env.rewards[n_train:],
            "pscore": self.env.pscore[n_train:],
            "context": self.env.contexts[n_train:],
            "position": np.zeros(n_total - n_train, dtype=int),
        }
        return train, test

    def _train_algorithm(self, algorithm, train_data):
        contexts = train_data["contexts"]
        actions = train_data["actions"]
        rewards = train_data["rewards"]
        n_arms = self.env.n_arms

        for i in range(len(contexts)):
            ctx = np.tile(contexts[i], (n_arms, 1))
            selected = algorithm.select_arm(ctx)
            if selected == int(actions[i]):
                algorithm.update([{
                    "action": int(actions[i]),
                    "reward": float(rewards[i]),
                    "context": ctx,
                }])

    def _predict_action_dist(self, algorithm, test_contexts):
        n_rounds = len(test_contexts)
        n_arms = self.env.n_arms
        action_dist = np.zeros((n_rounds, n_arms, 1))

        for i in range(n_rounds):
            ctx = np.tile(test_contexts[i], (n_arms, 1))
            chosen = algorithm.select_arm(ctx)
            action_dist[i, chosen, 0] = 1.0

        return action_dist

    def _build_reward_estimates(self, train_data, test_feedback):
        from mab_framework.experiment.reward_predictor import RewardPredictor
        predictor = RewardPredictor()
        predictor.fit(
            train_data["contexts"],
            train_data["actions"],
            train_data["rewards"],
        )
        return predictor.predict(test_feedback["context"], self.env.n_arms)

    def run(self):
        train_data, test_feedback = self._split_feedback()
        evaluator = OPEEvaluator(self.estimator_names)

        estimated_rewards = None
        if evaluator.needs_reward_model:
            estimated_rewards = self._build_reward_estimates(train_data, test_feedback)

        all_runs = []
        for run_idx in range(self.n_runs):
            run_seed = (self.seed + run_idx) if self.seed is not None else run_idx
            random.seed(run_seed)
            np.random.seed(run_seed)

            algorithm = self.algorithm_factory()
            start = time.time()
            self._train_algorithm(algorithm, train_data)
            train_time = time.time() - start

            action_dist = self._predict_action_dist(algorithm, test_feedback["context"])
            policy_values = evaluator.evaluate(
                test_feedback, action_dist,
                estimated_rewards_by_reg_model=estimated_rewards,
            )

            run_result = {
                "run_id": run_idx,
                "seed": run_seed,
                "train_time": round(train_time, 4),
                "policy_values": policy_values,
            }
            all_runs.append(run_result)

            if self.save_dir:
                self._save_run(run_result)

        aggregated = {}
        for est_name in all_runs[0]["policy_values"]:
            values = [r["policy_values"][est_name] for r in all_runs]
            aggregated[est_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

        return aggregated

    def _save_run(self, run_result):
        os.makedirs(self.save_dir, exist_ok=True)
        run_data = {**self.metadata, **run_result}
        path = os.path.join(self.save_dir, f"ope_run_{run_result['run_id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2)
