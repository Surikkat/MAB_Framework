import time
import numpy as np
from BanditLab.experiment.metrics import MetricsTracker
from BanditLab.experiment.logger import Logger

class ExperimentRunner:
    def __init__(self, env, algorithm_factory, steps: int, n_runs: int = 1, output_file: str = "results.json"):
        self.env = env
        self.algorithm_factory = algorithm_factory
        self.steps = steps
        self.n_runs = n_runs
        self.logger = Logger(output_file)

    def run(self):
        all_runs_metrics = {
            "cumulative_regret": [],
            "average_regret": [],
            "regrets": [],
            "rewards": [],
            "times": []
        }

        for run in range(self.n_runs):
            np.random.seed(run)
            try:
                import torch
                torch.manual_seed(run)
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

        aggregated_results = {}
        for key, value in all_runs_metrics.items():
            arr = np.array(value)
            aggregated_results[f"{key}_mean"] = np.mean(arr, axis=0).tolist()
            aggregated_results[f"{key}_std"] = np.std(arr, axis=0).tolist()
            aggregated_results[f"{key}_raw"] = arr.tolist()

        self.logger.log(aggregated_results)