"""End-to-end: PDF / text → detect → mask → write outputs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pii_mask.inference import load_pii_model, predict_spans
from pii_mask.masking import mask_text
from pii_mask.preview import render_preview_html


@dataclass
class PipelineResult:
    source_path: Path
    text: str
    spans: list[dict]
    masked_text: str
    mapping: dict[str, str]
    masked_path: Path
    mapping_path: Path
    preview_path: Path
    elapsed_s: float


def _load_text(path: Path, converter) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pii_mask.pdf_ingest import parse_pdf_path
        return parse_pdf_path(converter, path)
    return path.read_text(encoding="utf-8")


def process_file(
    path: Path,
    *,
    model,
    tokenizer,
    converter=None,
    out_dir: Optional[Path] = None,
) -> PipelineResult:
    """Run the full pipeline on a single file. Writes 3 sibling files."""
    started = time.perf_counter()
    text = _load_text(path, converter)
    spans = predict_spans(model, tokenizer, text)
    masked, mapping = mask_text(text, spans)

    target_dir = out_dir or path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    masked_path = target_dir / f"{stem}.masked.txt"
    mapping_path = target_dir / f"{stem}.mapping.json"
    preview_path = target_dir / f"{stem}.preview.html"

    masked_path.write_text(masked, encoding="utf-8")
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    preview_path.write_text(
        render_preview_html(text, spans, source=path.name),
        encoding="utf-8",
    )

    return PipelineResult(
        source_path=path,
        text=text,
        spans=spans,
        masked_text=masked,
        mapping=mapping,
        masked_path=masked_path,
        mapping_path=mapping_path,
        preview_path=preview_path,
        elapsed_s=time.perf_counter() - started,
    )
