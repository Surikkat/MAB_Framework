import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def plot_experiments(results_dir="experiment", save_dir="plots", log_scale=False):
    os.makedirs(save_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(results_dir, "*.json"))

    data_dict = {}
    for f in json_files:
        name = os.path.basename(f).replace("results_", "").replace(".json", "")
        with open(f, 'r') as file:
            try:
                data_dict[name] = json.load(file)
            except:
                continue

    metrics_to_plot = ["cumulative_regret", "average_regret"]

    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 7))

        for name, data in data_dict.items():
            mean_key = f"{metric}_mean"

            if mean_key in data:
                mean = np.array(data[mean_key])
                x = np.arange(len(mean))
                plt.plot(x, mean, label=name, linewidth=2.5)
            elif metric in data:
                val = np.array(data[metric])
                plt.plot(val, label=name, linewidth=2.5)

        plt.title(f"{metric.replace('_', ' ').title()}", fontsize=18, fontweight='bold', pad=20)
        plt.xlabel("Step", fontsize=14)
        plt.ylabel(metric.replace('_', ' ').capitalize(), fontsize=14)

        if log_scale:
            plt.yscale('log')

        plt.legend(frameon=True, shadow=True, borderpad=1, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        file_name = f"{metric}_log.png" if log_scale else f"{metric}.png"
        plt.savefig(os.path.join(save_dir, file_name), dpi=300)
        plt.close()

if __name__ == "__main__":
    plot_experiments(log_scale=False)
    plot_experiments(log_scale=True)