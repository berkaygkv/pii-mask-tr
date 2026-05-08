"""Turkish PII detection and masking.

Privacy posture: this package never sends document content, span data,
or the HF token anywhere except — when you explicitly run the model
fetcher — to Hugging Face Hub for model download.

We pre-emptively disable framework-level telemetry below before any
dependency imports. These env vars are only set if the user has not
already set them, so opting *into* telemetry remains possible.
"""

from __future__ import annotations

import os as _os


# Hugging Face Hub + Transformers + Datasets all read this on import.
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Honor the consoletools.org anti-telemetry convention.
_os.environ.setdefault("DO_NOT_TRACK", "1")
# Streamlit usage analytics. The .streamlit/config.toml in the repo also
# disables this, but the env var wins regardless of cwd.
_os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
# Suppress transformers' update-check on import.
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


__version__ = "0.1.0"
