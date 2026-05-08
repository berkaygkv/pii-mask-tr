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


class ModelLoadError(Exception):
    """Base for fetch_model failures. `hint` is a multi-line user
    message; `kind` is `"not_found" | "gated" | "unauthorized" | "fetch"`."""

    def __init__(self, kind: str, hint: str, *, repo: str, revision: str) -> None:
        super().__init__(hint)
        self.kind = kind
        self.hint = hint
        self.repo = repo
        self.revision = revision


_NOT_FOUND_HINT = (
    "Model repo not found at {repo}@{rev}.\n"
    "  - Check the repo + revision exist: https://huggingface.co/{repo}\n"
    "  - If the model hasn't been pushed yet, bypass the Hub by setting\n"
    "    PII_MASK_LOCAL_CHECKPOINT to a local checkpoint directory."
)

_GATED_HINT = (
    "The model at {repo}@{rev} is private/gated.\n"
    "  1. Request access at https://huggingface.co/{repo}\n"
    "  2. Create a read token at https://huggingface.co/settings/tokens\n"
    "  3. Set HF_TOKEN in your shell:\n"
    "       bash:        export HF_TOKEN=hf_xxx\n"
    "       powershell:  $env:HF_TOKEN = \"hf_xxx\"\n"
    "     or run: huggingface-cli login\n"
    "  4. Restart this app — env vars are captured at process start."
)

_AUTH_HINT = (
    "Not authorized to access {repo}@{rev}.\n"
    "  - Confirm your HF_TOKEN has read access to this repo.\n"
    "  - Generate a new token at https://huggingface.co/settings/tokens\n"
    "  - Restart this app after setting the token."
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

    Raises `ModelLoadError` on failure — callers (CLI, UI) decide how
    to render the hint.

    `refresh=True` forces a re-download even if the revision is cached.

    `PII_MASK_LOCAL_CHECKPOINT` (env) bypasses HF entirely and returns
    the given local path. Useful for development against an unpublished
    checkpoint.
    """
    local_override = os.getenv("PII_MASK_LOCAL_CHECKPOINT")
    if local_override:
        path = Path(local_override).expanduser().resolve()
        if not path.exists():
            raise ModelLoadError(
                "fetch",
                f"PII_MASK_LOCAL_CHECKPOINT points to a missing path: {path}",
                repo=str(path),
                revision="local",
            )
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
    except RepositoryNotFoundError as exc:
        raise ModelLoadError(
            "not_found",
            _NOT_FOUND_HINT.format(repo=repo, rev=rev),
            repo=repo, revision=rev,
        ) from exc
    except GatedRepoError as exc:
        raise ModelLoadError(
            "gated",
            _GATED_HINT.format(repo=repo, rev=rev),
            repo=repo, revision=rev,
        ) from exc
    except HfHubHTTPError as exc:
        msg = str(exc)
        if "404" in msg:
            raise ModelLoadError(
                "not_found",
                _NOT_FOUND_HINT.format(repo=repo, rev=rev),
                repo=repo, revision=rev,
            ) from exc
        if "401" in msg or "403" in msg:
            raise ModelLoadError(
                "unauthorized",
                _AUTH_HINT.format(repo=repo, rev=rev),
                repo=repo, revision=rev,
            ) from exc
        raise ModelLoadError(
            "fetch",
            f"error fetching {repo}@{rev}: {exc}",
            repo=repo, revision=rev,
        ) from exc
    return Path(path)
