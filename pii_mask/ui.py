"""`pii-mask-ui` console-script — execs `streamlit run` on _app.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch() -> None:
    """Console-script entry point."""
    app_path = Path(__file__).resolve().parent / "_app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    launch()
