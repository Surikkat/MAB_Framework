import sys
import argparse
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from experiment.api import Experiment

def main():
    parser = argparse.ArgumentParser(description="MAB Framework Runner (Legacy)")
    parser.add_argument("config", type=str, help="Path to the experiment YAML config")
    args = parser.parse_args()

    print(f"Starting experiment from {args.config} via legacy run.py...")
    try:
        exp = Experiment.from_config(args.config)
        exp.run()
        exp.plot()
        print("Experiment completed successfully.")
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
