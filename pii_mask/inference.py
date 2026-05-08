"""Sliding-window inference + span post-processing."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertForTokenClassification

from pii_mask.berturk_crf import (
    BertCrfForTokenClassification,
    is_crf_checkpoint,
)
from pii_mask.identifier_validators import filter_invalid_spans


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def decode_bio_to_spans(
    offsets: Sequence[tuple[int, int]],
    pred_ids: Sequence[int],
    id_to_label: dict[int, str],
) -> list[dict]:
    spans: list[dict] = []
    current: dict | None = None
    for (token_start, token_end), pred_id in zip(offsets, pred_ids):
        if token_start == token_end:
            if current is not None:
                spans.append(current)
                current = None
            continue
        label = id_to_label[pred_id]
        if label == "O":
            if current is not None:
                spans.append(current)
                current = None
            continue
        prefix, _, tag = label.partition("-")
        if prefix == "B":
            if current is not None:
                spans.append(current)
            current = {"label": tag, "start": token_start, "end": token_end}
        elif prefix == "I":
            if current is not None and current["label"] == tag:
                current["end"] = token_end
            else:
                current = {"label": tag, "start": token_start, "end": token_end}
    if current is not None:
        spans.append(current)
    return spans


_ADJACENT_JOIN_CHARS = frozenset({"@", ".", "-", "_", "/"})


def merge_adjacent_same_label_spans(spans: list[dict], text: str) -> list[dict]:
    if not spans:
        return spans
    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    merged: list[dict] = [dict(ordered[0])]
    for span in ordered[1:]:
        prev = merged[-1]
        gap = span["start"] - prev["end"]
        if span["label"] == prev["label"] and 0 <= gap <= 1:
            if gap == 0 or text[prev["end"]:span["start"]] in _ADJACENT_JOIN_CHARS:
                if span["end"] > prev["end"]:
                    prev["end"] = span["end"]
                    if "text" in prev:
                        prev["text"] = text[prev["start"]:prev["end"]]
                continue
        merged.append(dict(span))
    return merged


def merge_window_spans(window_spans: list[dict], text: str) -> list[dict]:
    if not window_spans:
        return []
    ordered = sorted(
        window_spans,
        key=lambda s: (s["start"], -(s["end"] - s["start"]), s["label"]),
    )
    merged: list[dict] = []
    for span in ordered:
        if merged and span["start"] < merged[-1]["end"]:
            if span["label"] == merged[-1]["label"]:
                if span["end"] > merged[-1]["end"]:
                    merged[-1]["end"] = span["end"]
                continue
            cur_len = merged[-1]["end"] - merged[-1]["start"]
            new_len = span["end"] - span["start"]
            if new_len > cur_len:
                merged[-1] = {
                    "label": span["label"],
                    "start": span["start"],
                    "end": span["end"],
                }
            continue
        merged.append({
            "label": span["label"],
            "start": span["start"],
            "end": span["end"],
        })
    for s in merged:
        s["text"] = text[s["start"]:s["end"]]
    return merged


def load_pii_model(checkpoint_path: Path):
    """Load the model from a local checkpoint dir. No network.

    `local_files_only=True` is enforced — even if a tokenizer file is
    missing, transformers will error out instead of silently fetching
    from the Hub. Model files have already been downloaded once via
    `pii_mask.model_loader.fetch_model`.
    """
    checkpoint_path = Path(checkpoint_path)
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    if is_crf_checkpoint(checkpoint_path):
        model = BertCrfForTokenClassification.from_pretrained(checkpoint_path)
    else:
        model = BertForTokenClassification.from_pretrained(checkpoint_path, local_files_only=True)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def predict_spans(
    model,
    tokenizer,
    text: str,
    *,
    max_length: int = 512,
    stride: int = 128,
) -> list[dict]:
    if not text:
        return []
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    id_to_label = {int(k): v for k, v in model.config.id2label.items()}
    device = next(model.parameters()).device

    all_spans: list[dict] = []
    with torch.no_grad():
        for window_idx in range(len(encoded["input_ids"])):
            input_ids = torch.tensor(
                [encoded["input_ids"][window_idx]], dtype=torch.long, device=device,
            )
            attention_mask = torch.tensor(
                [encoded["attention_mask"][window_idx]], dtype=torch.long, device=device,
            )
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(model, "predict"):
                preds = model.predict(outputs.logits, attention_mask)[0].cpu().tolist()
            else:
                preds = outputs.logits.argmax(dim=-1)[0].cpu().tolist()
            offsets = encoded["offset_mapping"][window_idx]
            all_spans.extend(decode_bio_to_spans(offsets, preds, id_to_label))

    merged = merge_window_spans(all_spans, text)
    glued = merge_adjacent_same_label_spans(merged, text)
    return filter_invalid_spans(glued, text)
