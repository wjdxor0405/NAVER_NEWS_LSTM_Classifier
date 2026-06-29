"""저장된 모델을 불러와 새 기사 문장을 분류하는 모듈."""

from __future__ import annotations

import pickle
from typing import Dict, Tuple

import torch

from app.config import Config
from app.model import TextLSTMClassifier
from app.preprocess import clean_text, pad_sequences, texts_to_sequences


def load_artifacts(config: Config) -> Tuple[TextLSTMClassifier, Dict[str, object]]:
    """저장된 모델 가중치와 전처리 메타데이터를 불러온다."""

    meta_path = config.model_path.replace(".pt", "_meta.pkl")
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    saved_config = metadata["config"]
    model = TextLSTMClassifier(
        vocab_size  = len(metadata["vocab"]),
        embed_dim   = saved_config.embed_dim,
        hidden_dim  = saved_config.hidden_dim,
        num_classes = len(metadata["label_to_id"]),
        num_layers  = saved_config.num_layers,
        dropout     = saved_config.dropout,
    )
    model.load_state_dict(torch.load(config.model_path, map_location="cpu"))
    model.eval()
    return model, metadata


def predict_text(
    text:     str,
    model:    TextLSTMClassifier,
    metadata: Dict[str, object],
    config:   Config,
) -> str:
    """새 기사 한 문장을 입력받아 예측된 카테고리명을 반환한다."""

    cleaned  = clean_text(text)
    sequence = texts_to_sequences([cleaned], metadata["vocab"])[0]
    padded   = pad_sequences([sequence], config.max_len)
    x        = torch.tensor(padded, dtype=torch.long)

    with torch.no_grad():
        logits   = model(x)
        pred_id  = int(torch.argmax(logits, dim=1).item())

    return metadata["id_to_label"][pred_id]


def predict_with_proba(
    text:     str,
    model:    TextLSTMClassifier,
    metadata: Dict[str, object],
    config:   Config,
) -> Dict[str, float]:
    """새 기사 한 문장에 대해 각 카테고리의 확률을 딕셔너리로 반환한다."""

    import torch.nn.functional as F

    cleaned  = clean_text(text)
    sequence = texts_to_sequences([cleaned], metadata["vocab"])[0]
    padded   = pad_sequences([sequence], config.max_len)
    x        = torch.tensor(padded, dtype=torch.long)

    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=1).squeeze(0)

    return {
        metadata["id_to_label"][i]: round(float(probs[i]), 4)
        for i in range(len(metadata["id_to_label"]))
    }
