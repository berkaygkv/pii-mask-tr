"""Hugging Face model fetch + auth + revision pinning.

The Turkish PII model is published as one HF repo per version, named
`berkaygkv/pii-model-turkish-<revision>` — `…-v4`, `…-v5`, etc. The
package keeps a `KNOWN_REVISIONS` list ordered newest-first; on every
run we try the newest cached revision first, then fall through to
older ones if it isn't cached and not yet pushed to the Hub.

Each `pii-mask-tr` release bumps `KNOWN_REVISIONS` to advertise newer
checkpoints. End users get the latest available without any flags;
power users can still pin via `--model-revision` / `PII_MASK_MODEL_REVISION`
or override the repo with `PII_MASK_MODEL_REPO`.
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


# Newest revision first. Bump on each release that ships a newer model.
# Keep this list to revisions that have actually been published to the Hub —
# entries that don't exist make every run pay an extra 404 round-trip and
# can confuse users when HF returns 401 instead of 404 for missing private
# repos.
KNOWN_REVISIONS: list[str] = ["v4"]
DEFAULT_REPO_PATTERN = "berkaygkv/pii-model-turkish-{revision}"

# In auto-resolve mode (no explicit pin), fall through these error kinds
# to the next-older revision. `unauthorized` is included because HF can
# return 401 for private repos that don't exist (privacy by obscurity);
# without this, a missing newer revision would mask a perfectly fine
# older one.
_AUTO_FALLBACK_KINDS = frozenset({"not_found", "unauthorized"})


class ModelLoadError(Exception):
    """Raised by `fetch_model` when no usable checkpoint can be obtained.

    `kind` is `"not_found" | "gated" | "unauthorized" | "fetch"`.
    `hint` is a multi-line, user-facing message.
    """

    def __init__(self, kind: str, hint: str, *, repo: str, revision: str) -> None:
        super().__init__(hint)
        self.kind = kind
        self.hint = hint
        self.repo = repo
        self.revision = revision


_NOT_FOUND_HINT = (
    "Model not found at any known revision: {tried}.\n"
    "  - The model may not have been pushed yet to its expected repo.\n"
    "  - You can pin a specific repo via PII_MASK_MODEL_REPO + PII_MASK_MODEL_REVISION.\n"
    "  - Or bypass HF entirely by setting PII_MASK_LOCAL_CHECKPOINT to a local path."
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


def _repo_for_revision(rev: str) -> str:
    return DEFAULT_REPO_PATTERN.format(revision=rev)


def _target_for(repo: str, rev: str) -> Path:
    return _cache_dir() / repo.replace("/", "__") / rev


def _is_cached(target: Path) -> bool:
    return target.exists() and any(target.iterdir())


def _resolve_explicit(
    repo_id: str | None, revision: str | None,
) -> tuple[str, str] | None:
    """If the user has pinned a specific repo / revision, return it.
    Otherwise return None to engage the auto-resolve path."""
    explicit_rev = revision or os.getenv("PII_MASK_MODEL_REVISION")
    explicit_repo = repo_id or os.getenv("PII_MASK_MODEL_REPO")
    if explicit_rev or explicit_repo:
        rev = explicit_rev or KNOWN_REVISIONS[0]
        repo = explicit_repo or _repo_for_revision(rev)
        return repo, rev
    return None


def _download(repo: str, rev: str, *, refresh: bool, quiet: bool) -> Path:
    target = _target_for(repo, rev)
    token = os.getenv("HF_TOKEN")
    if not refresh and _is_cached(target):
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
            _NOT_FOUND_HINT.format(tried=f"{repo}@{rev}"),
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
                _NOT_FOUND_HINT.format(tried=f"{repo}@{rev}"),
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


def fetch_model(
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    refresh: bool = False,
    quiet: bool = False,
) -> Path:
    """Resolve a usable checkpoint and return its local path.

    Resolution order:
      1. If `PII_MASK_LOCAL_CHECKPOINT` is set, use it. No HF.
      2. If the caller pins repo / revision (CLI flag or env var),
         fetch exactly that. Fail loudly on miss.
      3. Otherwise: walk `KNOWN_REVISIONS` newest-first.
         - First, return any cached revision (skip the network entirely).
         - If nothing is cached, try each revision against the Hub,
           falling through `not_found` only — auth / fetch errors raise.
    """
    if local := os.getenv("PII_MASK_LOCAL_CHECKPOINT"):
        path = Path(local).expanduser().resolve()
        if not path.exists():
            raise ModelLoadError(
                "fetch",
                f"PII_MASK_LOCAL_CHECKPOINT points to a missing path: {path}",
                repo=str(path), revision="local",
            )
        if not quiet:
            print(f"  using local checkpoint: {path}", file=sys.stderr)
        return path

    explicit = _resolve_explicit(repo_id, revision)
    if explicit is not None:
        repo, rev = explicit
        return _download(repo, rev, refresh=refresh, quiet=quiet)

    # Auto-resolve. Cached wins regardless of network state.
    if not refresh:
        for rev in KNOWN_REVISIONS:
            repo = _repo_for_revision(rev)
            target = _target_for(repo, rev)
            if _is_cached(target):
                if not quiet:
                    print(f"  using cached model: {repo}@{rev}", file=sys.stderr)
                return target

    last_fallback_err: ModelLoadError | None = None
    for rev in KNOWN_REVISIONS:
        repo = _repo_for_revision(rev)
        try:
            return _download(repo, rev, refresh=refresh, quiet=quiet)
        except ModelLoadError as exc:
            if exc.kind in _AUTO_FALLBACK_KINDS:
                last_fallback_err = exc
                if not quiet:
                    print(
                        f"  {repo}@{rev} unavailable ({exc.kind}), "
                        "trying older revision …",
                        file=sys.stderr,
                    )
                continue
            raise  # gated / fetch errors surface immediately

    # All known revisions failed. Surface the most recent error so the
    # user has the actionable hint (e.g. unauthorized → fix token).
    if last_fallback_err is not None:
        raise last_fallback_err
    tried = ", ".join(
        _repo_for_revision(r) + "@" + r for r in KNOWN_REVISIONS
    )
    raise ModelLoadError(
        "not_found",
        _NOT_FOUND_HINT.format(tried=tried),
        repo=_repo_for_revision(KNOWN_REVISIONS[0]),
        revision=KNOWN_REVISIONS[0],
    )
