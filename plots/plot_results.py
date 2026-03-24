import json
import os
import glob
import matplotlib.pyplot as plt

def plot_experiments(results_dir="experiment", save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    data_dict = {}
    for f in json_files:
        name = os.path.basename(f).replace("results_", "").replace(".json", "")
        with open(f, 'r') as file:
            data_dict[name] = json.load(file)
            
    plt.figure(figsize=(10, 6))
    for name, data in data_dict.items():
        if "cumulative_regret" in data:
            plt.plot(data["cumulative_regret"], label=name, linewidth=2)
    plt.title("Cumulative Regret", fontsize=14, fontweight='bold')
    plt.xlabel("steps")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cumulative_regret.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    for name, data in data_dict.items():
        if "average_regret" in data:
            plt.plot(data["average_regret"], label=name, linewidth=2)
    plt.title("Average Regret", fontsize=14, fontweight='bold')
    plt.xlabel("steps")
    plt.ylabel("Average Regret")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "average_regret.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_experiments()