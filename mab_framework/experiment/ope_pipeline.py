import os
import json
import numpy as np
from pathlib import Path
from mab_framework.experiment.ope_runner import OPERunner

class OPEPipeline:
    def __init__(self, env, algorithm_factory, candidate_name,
                 estimator_names=["ips", "snips"], train_ratio=0.7, 
                 n_runs=3, seed=None, save_dir=None, metadata=None):
        self.env = env
        self.algorithm_factory = algorithm_factory
        self.candidate_name = candidate_name
        self.estimator_names = estimator_names
        self.train_ratio = train_ratio
        self.n_runs = n_runs
        self.seed = seed
        self.save_dir = save_dir
        self.metadata = metadata or {}

    def run(self) -> dict:
        production_policy_value = float(np.mean(self.env.rewards))

        runner = OPERunner(
            env=self.env,
            algorithm_factory=self.algorithm_factory,
            estimator_names=self.estimator_names,
            train_ratio=self.train_ratio,
            n_runs=self.n_runs,
            seed=self.seed,
            save_dir=None,
        )
        
        aggregated_metrics = runner.run()

        metrics_dict = {
            "production_policy_value": round(production_policy_value, 6)
        }
        
        primary_estimator = list(aggregated_metrics.keys())[0] if aggregated_metrics else None
        candidate_wins = False

        for est_name in aggregated_metrics:
            cand_val = aggregated_metrics[est_name]["mean"]
            cand_std = aggregated_metrics[est_name]["std"]
            metrics_dict[f"candidate_policy_value_{est_name}"] = round(cand_val, 6)
            metrics_dict[f"candidate_std_{est_name}"] = round(cand_std, 6)
            
            if production_policy_value > 0:
                improvement = (cand_val - production_policy_value) / production_policy_value * 100
            else:
                improvement = 0.0
            metrics_dict[f"improvement_{est_name}"] = f"{improvement:+.2f}%"

            if est_name == primary_estimator:
                if cand_val > production_policy_value:
                    candidate_wins = True

        verdict = "candidate_wins" if candidate_wins else "production_wins"

        result = {
            "candidate_name": self.candidate_name,
            "metrics": metrics_dict,
            "verdict": verdict,
            "metadata": self.metadata
        }

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            safe_name = self.candidate_name.replace(" ", "_").lower()
            out_file = os.path.join(self.save_dir, f"{safe_name}_verdict.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)

        return result
