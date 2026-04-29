import yaml
import json
import torch
import numpy as np
import random
import os
import sys
from pathlib import Path
from pydantic import ValidationError

from .config_models import ExperimentConfig
from .runner import ExperimentRunner
import environments
import models
import algorithms

class Experiment:
    def __init__(self, config_dict: dict):
        try:
            self.config = ExperimentConfig(**config_dict)
        except ValidationError as e:
            print("Configuration validation failed:")
            print(e)
            sys.exit(1)

        self.setup_device_and_seed()
        self.setup_environment()
        self.all_results = None
        self.config_path = None

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        if not raw_config:
            raise ValueError(f"Config file {config_path} is empty or invalid.")
        exp = cls(raw_config)
        exp.config_path = config_path
        return exp

    def setup_device_and_seed(self):
        exp_glob = self.config.experiment
        
        # Determine global device
        self.device = torch.device(exp_glob.device)
        
        # Inject seed
        if exp_glob.seed is not None:
            self._set_seed(exp_glob.seed)

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def setup_environment(self):
        env_conf = self.config.environment
        EnvClass = getattr(environments, env_conf.name)
        params = dict(env_conf.params)
        if env_conf.delay:
            delay_dict = env_conf.delay.dict()
            params['delay_config'] = {
                "type": delay_dict["type"],
                **delay_dict.get("params", {})
            }
            
        self.env = EnvClass(**params)
        
        # Resolve dimensions
        self.n_arms = getattr(self.env, 'n_arms', env_conf.n_arms)
        
        self.feature_dim = None
        try:
            sample_context = self.env.get_context()
            self.feature_dim = sample_context.shape[-1]
            self.env.reset()
        except Exception as e:
            print(f"Warning: could not automatically determine feature_dim: {e}")

    def make_algo_factory(self, algo_conf):
        def factory():
            model = None
            if algo_conf.model:
                m_conf = algo_conf.model
                ModelClass = getattr(models, m_conf.name)
                m_params = dict(m_conf.params)
                
                if self.feature_dim is not None:
                    if 'feature_dim' not in m_params or m_params['feature_dim'] == 'auto':
                        m_params['feature_dim'] = self.feature_dim

                import inspect
                sig = inspect.signature(ModelClass.__init__)
                if 'device' in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    m_params['device'] = self.device

                one_model = algo_conf.one_model_per_arm if algo_conf.one_model_per_arm is not None else m_conf.one_model_per_arm
                if one_model:
                    model = [ModelClass(**m_params) for _ in range(self.n_arms)]
                else:
                    model = ModelClass(**m_params)
            
            a_params = dict(algo_conf.params)
            AlgoClass = getattr(algorithms, algo_conf.name)
            a_params['n_arms'] = self.n_arms
            if model is not None:
                a_params['model'] = model
            
            algo_instance = AlgoClass(**a_params)
            return algo_instance
        return factory

    def run(self):
        exp_glob = self.config.experiment
        save_path = Path(self.config.output.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        steps = exp_glob.steps
        env_max_steps = getattr(self.env, 'T', steps)
        if steps > env_max_steps:
            steps = env_max_steps

        all_results = {}
        for algo_conf in self.config.algorithms:
            a_name = algo_conf.display_name or algo_conf.name
            output_file = save_path / f"results_{a_name.replace(' ', '_')}.json"
            
            algo_factory = self.make_algo_factory(algo_conf)
            
            runner = ExperimentRunner(
                env=self.env,
                algorithm_factory=algo_factory,
                steps=steps,
                n_runs=exp_glob.n_runs,
                output_file=str(output_file)
            )
            
            runner.global_seed = exp_glob.seed
            runner.run()
            
            with open(output_file, 'r') as f:
                all_results[a_name] = json.load(f)
                
        self.all_results = all_results
        self.print_leaderboard(all_results)

    def plot(self):
        if not self.all_results:
            raise ValueError("First run the experiment using exp.run() before plotting.")
        save_path = Path(self.config.output.save_path)
        steps = self.config.experiment.steps
        self.plot_results(self.all_results, steps, save_path)

    def plot_results(self, all_results, steps, save_path):
        import matplotlib.pyplot as plt
        metrics_to_plot = self.config.metrics
        t_range = np.arange(1, steps + 1)
        plot_labels = {
            "cumulative_regret": "Cumulative Regret",
            "average_regret": "Average Regret",
            "runtime": "Runtime Metrics"
        }

        for target_metric in metrics_to_plot:
            metric_key = f"{target_metric}_mean"
            plt.figure(figsize=(10, 6))
            
            for a_name, data in all_results.items():
                if metric_key in data:
                    y_data = data[metric_key]
                    plt.plot(t_range[:len(y_data)], y_data, label=a_name, linewidth=2.0)
                    
            plt.xlabel("Round t", fontsize=12)
            ylabel = plot_labels.get(target_metric, target_metric.replace('_', ' ').title())
            plt.ylabel(ylabel, fontsize=12)
            plt.title(f"{self.config.experiment.name} - {ylabel}", fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plot_file = save_path / f"{target_metric}.png"
            plt.savefig(plot_file, dpi=300)
            plt.close()

    def print_leaderboard(self, all_results):
        try:
            from rich.console import Console
            from rich.table import Table
            
            console = Console()
            table = Table(title=f"Experiment Leaderboard: {self.config.experiment.name}")
            table.add_column("Algorithm", justify="left", style="cyan", no_wrap=True)
            table.add_column("Mean Cum. Regret", justify="right", style="magenta")
            table.add_column("Std Dev", justify="right", style="green")
            table.add_column("Avg Time/Step (s)", justify="right", style="yellow")
            
            for a_name, data in all_results.items():
                cum_regret = data.get("cumulative_regret_mean", [])
                cum_regret_std = data.get("cumulative_regret_std", [])
                runtime = data.get("runtime_mean", [])
                
                final_regret = f"{cum_regret[-1]:.3f}" if cum_regret else "N/A"
                final_std = f"{cum_regret_std[-1]:.3f}" if cum_regret_std else "N/A"
                avg_run = f"{np.mean(runtime):.5f}" if runtime else "N/A"
                
                table.add_row(a_name, final_regret, final_std, avg_run)
                
            console.print(table)
        except ImportError:
            import pandas as pd
            records = []
            for a_name, data in all_results.items():
                cum_regret = data.get("cumulative_regret_mean", [])
                cum_regret_std = data.get("cumulative_regret_std", [])
                runtime = data.get("runtime_mean", [])
                
                records.append({
                    "Algorithm": a_name,
                    "Mean Cum. Regret": cum_regret[-1] if cum_regret else None,
                    "Std Dev": cum_regret_std[-1] if cum_regret_std else None,
                    "Avg Time/Step (s)": np.mean(runtime) if runtime else None
                })
            df = pd.DataFrame(records)
            print("\n--- Experiment Leaderboard ---")
            print(df.to_string(index=False))
