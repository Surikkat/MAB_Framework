"""Console entry point for the BanditLab Streamlit GUI."""
import subprocess
import sys
from pathlib import Path


def main() -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "BanditLab GUI requires Streamlit and Plotly.\n"
            "Install them with:\n\n"
            "    pip install banditlab[gui]\n",
            file=sys.stderr,
        )
        sys.exit(1)

    app_path = Path(__file__).resolve().parent / "app.py"
    if not app_path.exists():
        print(f"Streamlit app not found at {app_path}", file=sys.stderr)
        sys.exit(1)
    sys.exit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    ))


if __name__ == "__main__":
    main()