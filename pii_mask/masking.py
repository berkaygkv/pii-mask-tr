"""Mask detected PII as round-trippable placeholders, and unmask back.

Placeholders use guillemets `«LABEL_N»` — single-codepoint sentinels
that survive cloud-LLM round-trips and don't collide with markdown.
Per-entity (not per-occurrence) indexing means identical values share
the same placeholder so the LLM can track coreference.
"""

from __future__ import annotations

import re
import unicodedata


_PLACEHOLDER_OPEN = "«"
_PLACEHOLDER_CLOSE = "»"


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().casefold()


def _placeholder(label: str, index: int) -> str:
    return f"{_PLACEHOLDER_OPEN}{label}_{index}{_PLACEHOLDER_CLOSE}"


def _check_no_overlap(spans: list[dict]) -> None:
    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    for prev, nxt in zip(ordered, ordered[1:]):
        if nxt["start"] < prev["end"]:
            raise ValueError(
                f"Overlapping spans are not supported by the masking layer: "
                f"{prev} overlaps {nxt}"
            )


def mask_text(text: str, spans: list[dict]) -> tuple[str, dict[str, str]]:
    """Replace each PII span with a `«LABEL_N»` placeholder.

    Returns `(masked_text, mapping)` where `mapping` is
    `{placeholder: original_text}`.
    """
    if not spans:
        return text, {}

    _check_no_overlap(spans)

    entity_to_placeholder: dict[tuple[str, str], str] = {}
    mapping: dict[str, str] = {}
    label_counters: dict[str, int] = {}

    for span in sorted(spans, key=lambda s: s["start"]):
        label = span["label"]
        original = text[span["start"]:span["end"]]
        key = (label, _normalize(original))
        if key not in entity_to_placeholder:
            label_counters[label] = label_counters.get(label, 0) + 1
            placeholder = _placeholder(label, label_counters[label])
            entity_to_placeholder[key] = placeholder
            mapping[placeholder] = original

    out = text
    for span in sorted(spans, key=lambda s: s["start"], reverse=True):
        original = text[span["start"]:span["end"]]
        key = (span["label"], _normalize(original))
        placeholder = entity_to_placeholder[key]
        out = out[:span["start"]] + placeholder + out[span["end"]:]

    return out, mapping


def unmask_text(masked: str, mapping: dict[str, str]) -> str:
    """Reverse `mask_text`. Replace each placeholder with the original."""
    if not mapping:
        return masked

    placeholders = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(p) for p in placeholders))
    return pattern.sub(lambda m: mapping[m.group(0)], masked)
