"""
10 — Launch the Streamlit GUI
==============================

Programmatically launch the BanditLab GUI.

Requires: pip install banditlab[gui]

Run:
    python examples/10_streamlit_launch.py
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Streamlit is not installed. Run: pip install banditlab[gui]")
        sys.exit(1)

    app_path = Path(__file__).resolve().parent.parent / "streamlit_bandit" / "app.py"
    if not app_path.exists():
        print(f"Streamlit app not found at {app_path}")
        sys.exit(1)

    print(f"Launching BanditLab GUI from {app_path}")
    print("Open http://localhost:8501 in your browser.\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=False,
    )


if __name__ == "__main__":
    main()