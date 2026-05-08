"""`pii-mask` — Turkish PII detection + masking CLI.

Subcommands:
  pii-mask warm                         — pre-download all models (one-time)
  pii-mask <file> [<file> ...] [opts]  — process files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pii_mask import __version__
from pii_mask.inference import load_pii_model
from pii_mask.model_loader import ModelLoadError, fetch_model
from pii_mask.pipeline import process_file


_USAGE = """\
usage:
  pii-mask warm                          pre-download all models (one-time)
  pii-mask <file> [<file> ...] [opts]    process PDFs / text files
  pii-mask --help                        full options
"""


def _print(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------- warm subcommand ----------------

def _cmd_warm(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pii-mask warm",
        description=(
            "Pre-download every model the tool needs at runtime. Run once "
            "after install; subsequent `pii-mask` and `pii-mask-ui` calls "
            "skip all downloads and start instantly. Use --refresh to "
            "upgrade to the latest published model — the loader queries "
            "Hugging Face for the highest published vN, so a freshly "
            "released model arrives without a `pii-mask-tr` release."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Upgrade path: query HF for the latest model version, "
            "re-download even if an older version is cached."
        ),
    )
    args = parser.parse_args(argv)

    _print("Pre-downloading all models — this is a one-time setup.")
    _print("")

    # 1. PII model from HF.
    _print("[1/3] PII detection model (BERTurk + CRF, ~500 MB)")
    try:
        checkpoint = fetch_model(refresh=args.refresh, quiet=False)
    except ModelLoadError as exc:
        _print("")
        _print(exc.hint)
        return 2
    _print(f"      cached at {checkpoint}")

    # 2. Docling layout + table models.
    _print("")
    _print("[2/3] PDF layout + table reader (Docling, ~1.5 GB)")
    _print("      this is the big one — sit tight")
    try:
        from pii_mask.pdf_ingest import build_converter
        build_converter()  # triggers Docling's internal HF downloads
        _print("      ready")
    except Exception as exc:  # noqa: BLE001
        _print(f"      warning: docling setup failed: {exc}")
        _print("      (docling models will retry on first PDF upload)")

    # 3. EasyOCR weights for Turkish + English.
    _print("")
    _print("[3/3] OCR weights (EasyOCR — Turkish + English, ~95 MB)")
    try:
        import easyocr
        easyocr.Reader(
            ["tr", "en"],
            gpu=False,
            download_enabled=True,
            verbose=False,
        )
        _print("      ready")
    except Exception as exc:  # noqa: BLE001
        _print(f"      warning: easyocr setup failed: {exc}")
        _print("      (OCR weights will retry on first scanned PDF)")

    _print("")
    _print("✓ All models cached. You can now run:")
    _print("    pii-mask document.pdf      (CLI)")
    _print("    pii-mask-ui                (browser UI)")
    return 0


# ---------------- process subcommand ----------------

def _cmd_process(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pii-mask",
        description=(
            "Detect and mask Turkish PII in PDFs and text files. "
            "Outputs three files next to each input: "
            "<name>.masked.txt, <name>.mapping.json, <name>.preview.html"
        ),
        usage=_USAGE,
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
        "--offline",
        action="store_true",
        help=(
            "Refuse all network access. Requires the model to already be "
            "cached locally. Sets HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"pii-mask-tr {__version__}",
    )
    args = parser.parse_args(argv)

    if args.offline:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        for p in missing:
            _print(f"error: not found: {p}")
        return 2

    _print("[1/3] fetching model (≈500 MB on first run, then cached) …")
    _print("      tip: run `pii-mask warm` once to pre-download all models")
    try:
        checkpoint = fetch_model(
            revision=args.model_revision,
            refresh=args.refresh_model,
        )
    except ModelLoadError as exc:
        _print("")
        _print(exc.hint)
        return 2

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


# ---------------- dispatch ----------------

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if argv and argv[0] in ("warm", "download", "init"):
        return _cmd_warm(argv[1:])
    if argv and argv[0] in ("-h", "--help"):
        print(_USAGE, file=sys.stderr)
        return 0
    return _cmd_process(argv)


if __name__ == "__main__":
    raise SystemExit(main())
