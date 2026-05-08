"""`pii-mask-ui` console-script — execs `streamlit run` on _app.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch() -> None:
    """Console-script entry point.

    Privacy-relevant flags are passed on the command line in addition
    to the .streamlit/config.toml in the repo, since uvx invocations
    may run from a cwd where streamlit's config search misses that file.
    """
    # Importing pii_mask triggers __init__ which sets telemetry-off env
    # vars; the subprocess inherits them.
    import pii_mask  # noqa: F401

    app_path = Path(__file__).resolve().parent / "_app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--browser.gatherUsageStats=false",
        "--server.address=localhost",
        "--server.headless=false",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    launch()
