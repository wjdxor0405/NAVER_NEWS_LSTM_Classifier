"""저장된 모델을 불러와 새 기사 문장을 분류하는 모듈."""

from __future__ import annotations
# 타입 힌트를 지금 평가하지 말고 나중에 평가하라는 의미의 모듈임
# 아직 만들어 지지 않은 클래스나 함수를 먼저 사용해도 에러를 표시하지 마라는 기능임

import pickle
from typing import Dict, Tuple

import torch

from app.config import Config
from app.model import TextLSTMClassifier
from app.preprocess import clean_text, pad_sequences, texts_to_sequences


def load_artifacts(config: Config) -> Tuple[TextLSTMClassifier, Dict[str, object]]:
    """저장된 모델 가중치와 전처리 메타데이터를 불러온다."""

    meta_path = config.model_path.replace(".pt", "_meta.pkl")                 # 모델과 함께 저장된 메타데이터 파일 경로를 만든다.
    with open(meta_path, "rb") as f:                                           # 메타데이터 파일을 바이너리 읽기 모드로 연다.
        metadata = pickle.load(f)                                              # 단어 사전과 라벨 사전을 읽어온다.
    model = TextLSTMClassifier(                                                # 저장 당시와 같은 구조의 모델 객체를 생성한다.
        vocab_size=len(metadata["vocab"]),
        embed_dim=metadata["config"].embed_dim,
        hidden_dim=metadata["config"].hidden_dim,
        num_classes=len(metadata["label_to_id"]),
    )
    model.load_state_dict(torch.load(config.model_path, map_location="cpu"))   # 저장된 가중치를 현재 모델에 적용한다.
    model.eval()                                                               # 예측용 평가 모드로 전환한다.
    return model, metadata                                                     # 모델과 메타데이터를 반환한다.


def predict_text(text: str, model: TextLSTMClassifier, metadata: Dict[str, object], config: Config) -> str:
    """새 기사 한 문장을 입력받아 예측된 BBC 카테고리명을 반환한다."""

    cleaned = clean_text(text)                                                 # 입력 문장도 학습 데이터와 동일한 규칙으로 정제한다.
    sequence = texts_to_sequences([cleaned], metadata["vocab"])[0]             # 정제된 문장을 정수 토큰으로 변환한다.
    padded = pad_sequences([sequence], config.max_len)                         # 모델 입력 길이에 맞게 패딩한다.
    x = torch.tensor(padded, dtype=torch.long)                                  # NumPy 배열을 PyTorch 텐서로 변환한다.
    with torch.no_grad():                                                       # 예측 시에는 기울기 계산을 하지 않는다.
        logits = model(x)                                                       # 모델이 각 카테고리 점수를 계산한다.
        pred_id = int(torch.argmax(logits, dim=1).item())                       # 가장 높은 점수의 클래스 ID를 추출한다.
    return metadata["id_to_label"][pred_id]                                    # 클래스 ID를 사람이 읽을 수 있는 라벨명으로 변환한다.
