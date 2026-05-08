"""Hugging Face model fetch + auth + revision pinning.

The Turkish PII model is published as one HF repo per version, named
`berkaygkv/pii-model-turkish-<revision>` — `…-v4`, `…-v5`, etc.

Resolution strategy when no pin is set:

1. **Cache-first short-circuit** — if any known revision is already on
   disk, use it. Zero network. Keeps startup instant and offline-safe.
2. **Live discovery** — when we *do* hit the network (cache miss or
   ``--refresh``), call ``discover_latest_revision_on_hub()`` and use
   the highest published ``vN`` as the primary candidate. This is the
   key seam that lets a freshly-published model reach end users
   *without* a `pii-mask-tr` release: the new repo on HF is enough.
3. **Fallback to KNOWN_REVISIONS** — discovery returns ``None`` when
   the Hub is unreachable, so the offline list still drives the walk.

Each `pii-mask-tr` release may still bump ``KNOWN_REVISIONS`` to widen
the offline fallback set, but a model bump alone no longer requires
one. End users get the latest model with::

    uvx --refresh --from git+https://github.com/berkaygkv/pii-mask-tr.git \\
        pii-mask warm --refresh

Power users can pin via ``--model-revision`` /
``PII_MASK_MODEL_REVISION`` or override the repo with
``PII_MASK_MODEL_REPO``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    RepositoryNotFoundError,
)
from huggingface_hub.utils import HfHubHTTPError


# Offline fallback set, newest first. Used when ``list_models`` on the
# Hub is unreachable (no network, HF down, ``HF_HUB_OFFLINE=1``). On a
# healthy network the loader prefers ``discover_latest_revision_on_hub``
# instead — this list does *not* need to be bumped on every model push.
KNOWN_REVISIONS: list[str] = ["v6", "v4"]
DEFAULT_REPO_PATTERN = "berkaygkv/pii-model-turkish-{revision}"
_REPO_NAME_RE = re.compile(r"^berkaygkv/pii-model-turkish-(v\d+)$")

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


def discover_latest_revision_on_hub() -> str | None:
    """Query HF for the highest published ``vN`` of the PII model.

    Returns the revision label (e.g. ``"v7"``) or ``None`` if the Hub
    is unreachable, the user is offline, or no matching repo exists.

    This is the seam that lets a freshly pushed model reach end users
    without a `pii-mask-tr` release: ``fetch_model`` calls this before
    walking ``KNOWN_REVISIONS`` and uses the result as the primary
    candidate when present.

    All exceptions are swallowed by design — the caller treats failure
    as "fall back to ``KNOWN_REVISIONS``", which preserves the
    existing offline path. Authentication is honoured via ``HF_TOKEN``
    so private/gated repos are visible to authorised users.
    """

    if os.getenv("HF_HUB_OFFLINE") == "1":
        return None
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.getenv("HF_TOKEN"))
        # ``search`` is a substring match against the model card; using
        # it scopes the listing tightly so we are not paginating the
        # whole author. The regex above does the strict filtering.
        results = api.list_models(
            author="berkaygkv",
            search="pii-model-turkish",
        )
        best: tuple[int, str] | None = None
        for model in results:
            match = _REPO_NAME_RE.match(model.id)
            if match is None:
                continue
            label = match.group(1)
            n = int(label[1:])
            if best is None or n > best[0]:
                best = (n, label)
        return best[1] if best else None
    except Exception:  # noqa: BLE001
        return None


def _candidate_revisions(*, prefer_hub: bool) -> list[str]:
    """Build the ordered list of revisions to try on the Hub.

    With ``prefer_hub=True``, prepend the discovered latest revision
    (de-duped against ``KNOWN_REVISIONS``). Discovery failure or
    ``prefer_hub=False`` returns ``KNOWN_REVISIONS`` unchanged.

    This is the only place the two sources are merged; everything else
    in ``fetch_model`` walks the resulting list.
    """

    if not prefer_hub:
        return list(KNOWN_REVISIONS)
    discovered = discover_latest_revision_on_hub()
    if discovered is None:
        return list(KNOWN_REVISIONS)
    out = [discovered]
    for rev in KNOWN_REVISIONS:
        if rev != discovered:
            out.append(rev)
    return out


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


def _primary_revision_for_ui() -> str:
    """The revision a UI should target for progress display.

    Discovery wins when the Hub is reachable so the progress bar polls
    the directory ``fetch_model`` will actually populate; falls back to
    ``KNOWN_REVISIONS[0]`` offline. Same merge rule as
    ``_candidate_revisions(prefer_hub=True)[0]``.
    """

    return _candidate_revisions(prefer_hub=True)[0]


def expected_cache_target() -> Path:
    """Where the auto-resolve path will write the downloaded model.

    Useful for UIs that want to poll the directory for live download
    progress before `fetch_model` returns. Tracks the discovered
    latest revision when the Hub is reachable.
    """
    label = _primary_revision_for_ui()
    return _target_for(_repo_for_revision(label), label)


def estimate_download_size_bytes(*, default_mb: int = 500) -> int:
    """Best-effort total download size of the auto-resolve revision.

    One HF API call (`model_info` with files metadata). Falls back to
    `default_mb` MB if the API is unreachable or any field is missing.
    Useful for sizing a progress bar before `snapshot_download` runs.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.getenv("HF_TOKEN"))
        label = _primary_revision_for_ui()
        repo = _repo_for_revision(label)
        info = api.model_info(repo, files_metadata=True)
        total = 0
        for sib in info.siblings or []:
            size = getattr(sib, "size", None) or getattr(sib, "lfs", None)
            if isinstance(size, int):
                total += size
            elif size and hasattr(size, "size"):
                total += size.size or 0
        if total > 0:
            return total
    except Exception:  # noqa: BLE001
        pass
    return default_mb * 1024 * 1024


def _resolve_explicit(
    repo_id: str | None, revision: str | None,
) -> tuple[str, str, str] | None:
    """If the user has pinned a specific repo / revision, return
    `(repo, label, git_rev)`. Otherwise None — engages auto-resolve.

    `label` is what we display + cache as. `git_rev` is what we ask
    HF for. They diverge for the per-version-repo pattern, where the
    version is in the repo name and the actual revision is `main`.
    """
    explicit_rev = revision or os.getenv("PII_MASK_MODEL_REVISION")
    explicit_repo = repo_id or os.getenv("PII_MASK_MODEL_REPO")
    if not (explicit_rev or explicit_repo):
        return None
    if explicit_repo:
        # User pinned a custom repo. Their revision is the git ref.
        rev = explicit_rev or "main"
        return explicit_repo, rev, rev
    # User pinned only a model version → per-version-repo, files on main.
    rev = explicit_rev or KNOWN_REVISIONS[0]
    return _repo_for_revision(rev), rev, "main"


def _download(
    repo: str, label: str, git_rev: str,
    *, refresh: bool, quiet: bool,
) -> Path:
    """Fetch (or reuse cached) `repo` at `git_rev`. Cache + display
    use `label`."""
    target = _target_for(repo, label)
    token = os.getenv("HF_TOKEN")
    if not refresh and _is_cached(target):
        if not quiet:
            print(f"  using cached model: {repo}@{label}", file=sys.stderr)
        return target
    if not quiet:
        print(f"  downloading model: {repo}@{label} ...", file=sys.stderr)
    try:
        path = snapshot_download(
            repo_id=repo,
            revision=git_rev,
            local_dir=str(target),
            token=token,
            force_download=refresh,
        )
    except RepositoryNotFoundError as exc:
        raise ModelLoadError(
            "not_found",
            _NOT_FOUND_HINT.format(tried=f"{repo}@{label}"),
            repo=repo, revision=label,
        ) from exc
    except GatedRepoError as exc:
        raise ModelLoadError(
            "gated",
            _GATED_HINT.format(repo=repo, rev=label),
            repo=repo, revision=label,
        ) from exc
    except HfHubHTTPError as exc:
        msg = str(exc)
        if "404" in msg:
            raise ModelLoadError(
                "not_found",
                _NOT_FOUND_HINT.format(tried=f"{repo}@{label}"),
                repo=repo, revision=label,
            ) from exc
        if "401" in msg or "403" in msg:
            raise ModelLoadError(
                "unauthorized",
                _AUTH_HINT.format(repo=repo, rev=label),
                repo=repo, revision=label,
            ) from exc
        raise ModelLoadError(
            "fetch",
            f"error fetching {repo}@{label}: {exc}",
            repo=repo, revision=label,
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
        repo, label, git_rev = explicit
        return _download(repo, label, git_rev, refresh=refresh, quiet=quiet)

    # Auto-resolve. Cached wins regardless of network state — keeps
    # offline runs fast and predictable. ``--refresh`` forces both a
    # discovery query and a fresh download, which is the upgrade path.
    if not refresh:
        for label in KNOWN_REVISIONS:
            repo = _repo_for_revision(label)
            target = _target_for(repo, label)
            if _is_cached(target):
                if not quiet:
                    print(f"  using cached model: {repo}@{label}", file=sys.stderr)
                return target

    # We are going to hit the network. Ask the Hub which version is
    # newest so a freshly pushed model is picked up without waiting
    # for a pii-mask-tr release. Discovery failure (offline / HF down)
    # silently falls back to KNOWN_REVISIONS so the cached/offline
    # path still works.
    candidates = _candidate_revisions(prefer_hub=True)
    if not quiet and candidates and candidates[0] not in KNOWN_REVISIONS:
        print(
            f"  discovered newer revision on Hub: {candidates[0]} "
            f"(KNOWN_REVISIONS={KNOWN_REVISIONS})",
            file=sys.stderr,
        )

    last_fallback_err: ModelLoadError | None = None
    for label in candidates:
        repo = _repo_for_revision(label)
        try:
            # Per-version repo: version is in the repo name; files on `main`.
            return _download(repo, label, "main", refresh=refresh, quiet=quiet)
        except ModelLoadError as exc:
            if exc.kind in _AUTO_FALLBACK_KINDS:
                last_fallback_err = exc
                if not quiet:
                    print(
                        f"  {repo}@{label} unavailable ({exc.kind}), "
                        "trying older revision …",
                        file=sys.stderr,
                    )
                continue
            raise  # gated / fetch errors surface immediately

    # All candidates failed. Surface the most recent error so the user
    # has the actionable hint (e.g. unauthorized → fix token).
    if last_fallback_err is not None:
        raise last_fallback_err
    tried = ", ".join(
        _repo_for_revision(r) + "@" + r for r in candidates
    )
    head = candidates[0] if candidates else KNOWN_REVISIONS[0]
    raise ModelLoadError(
        "not_found",
        _NOT_FOUND_HINT.format(tried=tried),
        repo=_repo_for_revision(head),
        revision=head,
    )
