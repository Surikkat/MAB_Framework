import argparse
import sys
from .api import Experiment

def main():
    parser = argparse.ArgumentParser(description="MAB Framework CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    run_parser = subparsers.add_parser("run", help="Run an experiment from a config file")
    run_parser.add_argument("config", type=str, help="Path to the experiment YAML config")
    
    args = parser.parse_args()
    
    if args.command == "run":
        if not args.config:
            print("Error: Config path is required for the 'run' command.")
            sys.exit(1)
        
        print(f"Initializing experiment from {args.config}...")
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
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
