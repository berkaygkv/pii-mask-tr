"""Format-only structural validators for Turkish PII identifiers.

Used as a precision filter on model predictions: a span predicted with
label X must structurally match X to be kept. Drops false positives
only — never adds spans, never reclassifies.

Format-only by design: false negatives leak PII (worse) while strict
checksums improve precision but cost recall on placeholders, typos, and
partial exports.
"""

from __future__ import annotations

import re
from typing import Iterable


_TCKN_RE = re.compile(r"^[1-9][0-9]{10}$")
_VKN_RE = re.compile(r"^[0-9]{10}$")
_IBAN_TR_RE = re.compile(r"^TR[0-9]{24}$")
_VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$", re.IGNORECASE)
_PLAKA_RE = re.compile(
    r"^(?:0[1-9]|[1-7][0-9]|8[01])[\s-]?[A-PR-VYZ]{1,3}[\s-]?[0-9]{2,4}$",
    re.IGNORECASE,
)


def is_valid_tckn(value: str) -> bool:
    return bool(_TCKN_RE.match(value))


def is_valid_vkn(value: str) -> bool:
    return bool(_VKN_RE.match(value))


def is_valid_iban_tr(value: str) -> bool:
    return bool(_IBAN_TR_RE.match(value.replace(" ", "").upper()))


def is_valid_sasi_no(value: str) -> bool:
    return bool(_VIN_RE.match(value))


def is_valid_plaka(value: str) -> bool:
    return bool(_PLAKA_RE.match(value.strip()))


_VALIDATORS = {
    "TCKN": is_valid_tckn,
    "VKN": is_valid_vkn,
    "IBAN": is_valid_iban_tr,
    "SASI_NO": is_valid_sasi_no,
    "PLAKA": is_valid_plaka,
}


def is_structurally_valid(label: str, span_text: str) -> bool:
    validator = _VALIDATORS.get(label)
    if validator is None:
        return True
    return validator(span_text)


def filter_invalid_spans(spans: Iterable[dict], text: str) -> list[dict]:
    kept: list[dict] = []
    for span in spans:
        span_text = text[span["start"]:span["end"]]
        if is_structurally_valid(span["label"], span_text):
            kept.append(span)
    return kept
