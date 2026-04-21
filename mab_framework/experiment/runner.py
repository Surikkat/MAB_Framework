import time
import json
import random
import os
import numpy as np
from mab_framework.experiment.metrics import MetricsTracker
from mab_framework.experiment.logger import Logger


class ExperimentRunner:
    def __init__(self, env, algorithm_factory, steps: int, n_runs: int = 1,
                 output_file: str = None, seed: int = None,
                 save_dir: str = None, metadata: dict = None):
        self.env = env
        self.algorithm_factory = algorithm_factory
        self.steps = steps
        self.n_runs = n_runs
        self.output_file = output_file
        self.seed = seed
        self.save_dir = save_dir
        self.metadata = metadata or {}

    def _get_run_seed(self, run_idx):
        if self.seed is not None:
            return self.seed + run_idx
        return run_idx

    def run(self):
        all_runs_metrics = {
            "cumulative_regret": [],
            "average_regret": [],
            "regrets": [],
            "rewards": [],
            "times": []
        }

        for run in range(self.n_runs):
            run_seed = self._get_run_seed(run)
            random.seed(run_seed)
            np.random.seed(run_seed)
            try:
                import torch
                torch.manual_seed(run_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(run_seed)
            except ImportError:
                pass

            if hasattr(self.env, 'reset'):
                self.env.reset()

            algorithm = self.algorithm_factory()
            tracker = MetricsTracker()

            for step in range(self.steps):
                start_time = time.time()
                context = self.env.get_context()
                action = algorithm.select_arm(context)
                step_result = self.env.step(action)

                if isinstance(step_result, dict) and "available_rewards" in step_result:
                    available_rewards = step_result["available_rewards"]
                    reward = step_result.get("instant_reward", 0.0)
                    optimal_reward = step_result.get("optimal_reward", reward)
                elif isinstance(step_result, tuple) and len(step_result) == 2:
                    reward, optimal_reward = step_result
                    available_rewards = [{"action": action, "reward": reward, "context": context}]
                else:
                    reward = step_result
                    optimal_reward = reward
                    available_rewards = [{"action": action, "reward": reward, "context": context}]

                algorithm.update(available_rewards)

                iter_time = time.time() - start_time
                regret = optimal_reward - reward
                tracker.add(reward, regret, iter_time)

            metrics = tracker.get_metrics()
            all_runs_metrics["cumulative_regret"].append(metrics["cumulative_regret"])
            all_runs_metrics["average_regret"].append(metrics["average_regret"])
            all_runs_metrics["regrets"].append(metrics["regrets"])
            all_runs_metrics["rewards"].append(metrics["rewards"])
            all_runs_metrics["times"].append(metrics["times"])

            if self.save_dir:
                self._save_run(run, run_seed, metrics)

        aggregated = {}
        for key, value in all_runs_metrics.items():
            arr = np.array(value)
            aggregated[f"{key}_mean"] = np.mean(arr, axis=0).tolist()
            aggregated[f"{key}_std"] = np.std(arr, axis=0).tolist()

        if self.output_file:
            logger = Logger(self.output_file)
            full = dict(aggregated)
            for key, value in all_runs_metrics.items():
                full[f"{key}_raw"] = np.array(value).tolist()
            logger.log(full)

        return aggregated

    def _save_run(self, run_idx, run_seed, metrics):
        os.makedirs(self.save_dir, exist_ok=True)
        run_data = {
            **self.metadata,
            "seed": run_seed,
            "run_id": run_idx,
            "metrics": {
                "cumulative_regret": metrics["cumulative_regret"],
                "average_regret": metrics["average_regret"],
            },
            "runtime": round(sum(metrics["times"]), 4),
        }
        path = os.path.join(self.save_dir, f"run_{run_idx}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(run_data, f, indent=2)