"""BERTurk + CRF token classifier.

Wrapper around `bert-base-turkish-128k-cased` with a linear head and a
CRF layer that constrains BIO transitions. The CRF layer tightens
precision on rare labels and eliminates orphan I- tags mid-entity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoConfig, AutoModel


MODEL_TYPE_MARKER = "bert_crf"
HEAD_CRF_FILENAME = "head_crf.pt"
MODEL_TYPE_FILENAME = "model_type.json"


@dataclass
class BertCrfOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor


class BertCrfForTokenClassification(nn.Module):
    def __init__(
        self,
        bert: nn.Module,
        config,
        num_labels: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.bert = bert
        self.config = config
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> BertCrfOutput:
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(bert_out.last_hidden_state)
        logits = self.classifier(hidden)

        loss: torch.Tensor | None = None
        if labels is not None:
            crf_mask = attention_mask.bool() & (labels != -100)
            crf_mask[:, 0] = True
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            loss = -self.crf(logits, crf_labels, mask=crf_mask, reduction="mean")
        return BertCrfOutput(loss=loss, logits=logits)

    def predict(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.bool()
        decoded = self.crf.decode(logits, mask=mask)
        result = torch.zeros(
            logits.shape[:-1], dtype=torch.long, device=logits.device,
        )
        for batch_idx, path in enumerate(decoded):
            length = min(len(path), result.shape[1])
            for pos in range(length):
                result[batch_idx, pos] = path[pos]
        return result

    @classmethod
    def from_pretrained(
        cls,
        save_directory,
        *,
        local_files_only: bool = True,
        dropout: float = 0.1,
    ) -> "BertCrfForTokenClassification":
        save_directory = Path(save_directory)
        config = AutoConfig.from_pretrained(save_directory, local_files_only=local_files_only)
        bert = AutoModel.from_pretrained(save_directory, config=config, local_files_only=local_files_only)
        instance = cls(bert=bert, config=config, num_labels=config.num_labels, dropout=dropout)
        head_crf = torch.load(
            save_directory / HEAD_CRF_FILENAME,
            map_location="cpu",
            weights_only=True,
        )
        instance.classifier.load_state_dict(head_crf["classifier"])
        instance.crf.load_state_dict(head_crf["crf"])
        return instance


def is_crf_checkpoint(path) -> bool:
    marker = Path(path) / MODEL_TYPE_FILENAME
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("type") == MODEL_TYPE_MARKER
