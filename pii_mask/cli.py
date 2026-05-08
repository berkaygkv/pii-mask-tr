"""`pii-mask` — Turkish PII detection + masking CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pii_mask import __version__
from pii_mask.inference import load_pii_model
from pii_mask.model_loader import fetch_model
from pii_mask.pipeline import process_file


def _print(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pii-mask",
        description=(
            "Detect and mask Turkish PII in PDFs and text files. "
            "Outputs three files next to each input: "
            "<name>.masked.txt, <name>.mapping.json, <name>.preview.html"
        ),
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="PDF or .txt files")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write outputs here instead of next to each input",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="HF revision (tag/branch/sha). Default: pinned production version.",
    )
    parser.add_argument(
        "--refresh-model",
        action="store_true",
        help="Re-download the model even if cached",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"pii-mask-tr {__version__}",
    )
    args = parser.parse_args(argv)

    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        for p in missing:
            _print(f"error: not found: {p}")
        return 2

    _print("[1/3] preparing model …")
    checkpoint = fetch_model(
        revision=args.model_revision,
        refresh=args.refresh_model,
    )

    _print("[2/3] loading model …")
    model, tokenizer = load_pii_model(checkpoint)
    device = next(model.parameters()).device
    _print(f"      ready on {device}")

    needs_pdf = any(p.suffix.lower() == ".pdf" for p in args.inputs)
    converter = None
    if needs_pdf:
        _print("      initialising PDF reader (Docling) …")
        from pii_mask.pdf_ingest import build_converter
        converter = build_converter()

    _print("[3/3] processing …")
    total_entities = 0
    for path in args.inputs:
        result = process_file(
            path,
            model=model,
            tokenizer=tokenizer,
            converter=converter,
            out_dir=args.out_dir,
        )
        total_entities += len(result.spans)
        _print(
            f"  ✓ {path.name}: {len(result.spans)} entities "
            f"({result.elapsed_s:.1f}s)"
        )
        _print(f"      → {result.masked_path}")
        _print(f"      → {result.mapping_path}")
        _print(f"      → {result.preview_path}")

    _print(
        f"\ndone — {total_entities} entities across {len(args.inputs)} file(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
