"""Live-discovery and candidate-resolution tests for ``model_loader``.

The point of discovery is decoupling: a freshly published
``berkaygkv/pii-model-turkish-vN`` repo on HF must reach end users
without a ``pii-mask-tr`` release. Without these tests, a future
refactor could quietly revert to the old hand-maintained
``KNOWN_REVISIONS`` walk and the upgrade path would silently break
again.

All tests stub the HF client — no live network calls.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from pii_mask import model_loader as ml


class _FakeModel:
    """Minimal stand-in for ``huggingface_hub.ModelInfo`` (just .id)."""

    def __init__(self, repo_id: str) -> None:
        self.id = repo_id


class DiscoverLatestRevisionTest(unittest.TestCase):
    """``discover_latest_revision_on_hub`` picks the highest published vN."""

    def _patch_list_models(self, repo_ids: list[str]):
        fake_api = mock.MagicMock()
        fake_api.list_models.return_value = [_FakeModel(rid) for rid in repo_ids]
        return mock.patch("huggingface_hub.HfApi", return_value=fake_api)

    def test_picks_highest_version(self) -> None:
        with self._patch_list_models([
            "berkaygkv/pii-model-turkish-v4",
            "berkaygkv/pii-model-turkish-v6",
            "berkaygkv/pii-model-turkish-v5",
        ]):
            self.assertEqual("v6", ml.discover_latest_revision_on_hub())

    def test_handles_double_digit_versions(self) -> None:
        """Lexicographic compare would put 'v9' > 'v12'. Numeric must win."""

        with self._patch_list_models([
            "berkaygkv/pii-model-turkish-v9",
            "berkaygkv/pii-model-turkish-v12",
            "berkaygkv/pii-model-turkish-v6",
        ]):
            self.assertEqual("v12", ml.discover_latest_revision_on_hub())

    def test_ignores_non_matching_repos(self) -> None:
        """Other repos under the same author don't poison the list."""

        with self._patch_list_models([
            "berkaygkv/some-other-model",
            "berkaygkv/pii-model-turkish-v6",
            "berkaygkv/pii-model-turkish-experimental",  # missing -vN
            "berkaygkv/pii-model-turkish-v6-rerun",      # extra suffix
        ]):
            self.assertEqual("v6", ml.discover_latest_revision_on_hub())

    def test_returns_none_on_no_matches(self) -> None:
        with self._patch_list_models(["berkaygkv/some-other-model"]):
            self.assertIsNone(ml.discover_latest_revision_on_hub())

    def test_returns_none_on_network_error(self) -> None:
        """HF unreachable / DNS failure / token rejected — discovery
        must swallow and let the caller fall back to KNOWN_REVISIONS.
        """

        fake_api = mock.MagicMock()
        fake_api.list_models.side_effect = OSError("network down")
        with mock.patch("huggingface_hub.HfApi", return_value=fake_api):
            self.assertIsNone(ml.discover_latest_revision_on_hub())

    def test_offline_env_var_short_circuits(self) -> None:
        """``HF_HUB_OFFLINE=1`` must skip the network call entirely so
        the user's offline mode is respected without a wrapped retry.
        """

        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            with mock.patch("huggingface_hub.HfApi") as factory:
                self.assertIsNone(ml.discover_latest_revision_on_hub())
                factory.assert_not_called()


class CandidateRevisionsTest(unittest.TestCase):
    """``_candidate_revisions`` merges discovery with KNOWN_REVISIONS."""

    def test_prefer_hub_false_returns_known(self) -> None:
        self.assertEqual(
            list(ml.KNOWN_REVISIONS),
            ml._candidate_revisions(prefer_hub=False),
        )

    def test_discovery_returning_known_value_does_not_duplicate(self) -> None:
        """Discovery returning ``v6`` (already in KNOWN_REVISIONS) must
        not duplicate it in the candidate list.
        """

        with mock.patch.object(
            ml, "discover_latest_revision_on_hub", return_value="v6",
        ):
            candidates = ml._candidate_revisions(prefer_hub=True)
            self.assertEqual(candidates.count("v6"), 1)
            self.assertEqual("v6", candidates[0])

    def test_discovered_newer_revision_is_prepended(self) -> None:
        """A v9 discovered from HF (not yet in KNOWN_REVISIONS) must be
        the first candidate, with the offline fallbacks following.
        """

        with mock.patch.object(
            ml, "discover_latest_revision_on_hub", return_value="v9",
        ):
            candidates = ml._candidate_revisions(prefer_hub=True)
        self.assertEqual("v9", candidates[0])
        for rev in ml.KNOWN_REVISIONS:
            self.assertIn(rev, candidates)

    def test_discovery_failure_falls_back_to_known(self) -> None:
        """Offline / Hub down → behave exactly like the old loader."""

        with mock.patch.object(
            ml, "discover_latest_revision_on_hub", return_value=None,
        ):
            self.assertEqual(
                list(ml.KNOWN_REVISIONS),
                ml._candidate_revisions(prefer_hub=True),
            )


class FetchModelUsesDiscoveryTest(unittest.TestCase):
    """``fetch_model`` must consult discovery on every network path.

    Cache-first short-circuit is preserved — no network when something
    is already on disk and ``refresh`` is False — but as soon as we
    reach the download walk, the discovered revision must lead.
    """

    def setUp(self) -> None:
        # Force every cache lookup to return False so the loader always
        # walks the candidate list.
        self._uncached = mock.patch.object(ml, "_is_cached", return_value=False)
        self._uncached.start()
        # Track which revisions were attempted; succeed on the first.
        self._attempts: list[str] = []
        def fake_download(repo, label, git_rev, *, refresh, quiet):  # noqa: ARG001
            self._attempts.append(label)
            return ml.Path(f"/tmp/__fake_cache__/{label}")
        self._download = mock.patch.object(ml, "_download", side_effect=fake_download)
        self._download.start()

    def tearDown(self) -> None:
        self._uncached.stop()
        self._download.stop()

    def test_refresh_path_tries_discovered_revision_first(self) -> None:
        with mock.patch.object(
            ml, "discover_latest_revision_on_hub", return_value="v9",
        ):
            ml.fetch_model(refresh=True, quiet=True)
        self.assertEqual("v9", self._attempts[0])

    def test_first_run_with_empty_cache_tries_discovered_first(self) -> None:
        """Cache short-circuit fails (everything uncached), discovery
        runs, and the discovered revision wins. This is the
        ``uvx --from git+...`` fresh-install scenario.
        """

        with mock.patch.object(
            ml, "discover_latest_revision_on_hub", return_value="v9",
        ):
            ml.fetch_model(refresh=False, quiet=True)
        self.assertEqual("v9", self._attempts[0])

    def test_offline_falls_back_to_known(self) -> None:
        """Discovery returns None → walk KNOWN_REVISIONS only, in order."""

        with mock.patch.object(
            ml, "discover_latest_revision_on_hub", return_value=None,
        ):
            ml.fetch_model(refresh=True, quiet=True)
        self.assertEqual(ml.KNOWN_REVISIONS[0], self._attempts[0])


class ExpectedCacheTargetTrackingTest(unittest.TestCase):
    """``expected_cache_target`` (used by the UI progress bar) must
    track the same revision ``fetch_model`` will actually populate.

    Without this alignment, the UI would poll the v6 directory while
    ``fetch_model`` writes to v7, and the progress bar would never move.
    """

    def test_target_follows_discovery(self) -> None:
        with mock.patch.object(
            ml, "discover_latest_revision_on_hub", return_value="v9",
        ):
            target = ml.expected_cache_target()
        self.assertIn("v9", str(target))


if __name__ == "__main__":
    unittest.main()
