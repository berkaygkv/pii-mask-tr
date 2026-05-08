"""Hugging Face model fetch + auth + revision pinning.

The model lives at `berkaygkv/pii-mask-turkish` on Hugging Face. While
it is private, an HF token is required (read scope is enough). When
the model goes public, the same code path works without a token.

Default revision is pinned in `DEFAULT_MODEL_REVISION`. Bumping that
constant in a release ships a new model to all users on next run.
Power users can override with `--model-revision` (CLI) or
`PII_MASK_MODEL_REVISION` (env).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    RepositoryNotFoundError,
)
from huggingface_hub.utils import HfHubHTTPError


DEFAULT_MODEL_REPO = "berkaygkv/pii-mask-turkish"
DEFAULT_MODEL_REVISION = "v5"

_AUTH_HINT = (
    "\nThe Turkish PII model is currently private. To use it:\n"
    "  1. Create a read token at https://huggingface.co/settings/tokens\n"
    "  2. Request access at https://huggingface.co/{repo}\n"
    "  3. Export it: export HF_TOKEN=hf_xxx   (or run: huggingface-cli login)\n"
    "\nThe model will be made public in a future release; this step will go away.\n"
)


def _cache_dir() -> Path:
    base = os.getenv("PII_MASK_CACHE") or os.path.expanduser("~/.cache/pii-mask-tr")
    path = Path(base) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_model_revision(override: str | None = None) -> str:
    return (
        override
        or os.getenv("PII_MASK_MODEL_REVISION")
        or DEFAULT_MODEL_REVISION
    )


def resolve_model_repo(override: str | None = None) -> str:
    return (
        override
        or os.getenv("PII_MASK_MODEL_REPO")
        or DEFAULT_MODEL_REPO
    )


def fetch_model(
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    refresh: bool = False,
    quiet: bool = False,
) -> Path:
    """Download (or reuse cached) model snapshot. Returns local path.

    `refresh=True` forces a re-download even if the revision is cached.

    `PII_MASK_LOCAL_CHECKPOINT` (env) bypasses HF entirely and returns
    the given local path. Useful for development against an unpublished
    checkpoint.
    """
    local_override = os.getenv("PII_MASK_LOCAL_CHECKPOINT")
    if local_override:
        path = Path(local_override).expanduser().resolve()
        if not path.exists():
            sys.stderr.write(
                f"PII_MASK_LOCAL_CHECKPOINT points to a missing path: {path}\n"
            )
            sys.exit(2)
        if not quiet:
            print(f"  using local checkpoint: {path}", file=sys.stderr)
        return path

    repo = resolve_model_repo(repo_id)
    rev = resolve_model_revision(revision)
    target = _cache_dir() / repo.replace("/", "__") / rev
    token = os.getenv("HF_TOKEN")

    if not refresh and target.exists() and any(target.iterdir()):
        if not quiet:
            print(f"  using cached model: {repo}@{rev}", file=sys.stderr)
        return target

    if not quiet:
        print(f"  downloading model: {repo}@{rev} ...", file=sys.stderr)
    try:
        path = snapshot_download(
            repo_id=repo,
            revision=rev,
            local_dir=str(target),
            token=token,
            force_download=refresh,
        )
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        sys.stderr.write(_AUTH_HINT.format(repo=repo))
        sys.stderr.write(f"\n(underlying error: {exc.__class__.__name__})\n")
        sys.exit(2)
    except HfHubHTTPError as exc:
        msg = str(exc)
        if "401" in msg or "403" in msg:
            sys.stderr.write(_AUTH_HINT.format(repo=repo))
            sys.exit(2)
        sys.stderr.write(f"error fetching {repo}@{rev}: {exc}\n")
        sys.exit(1)
    return Path(path)
