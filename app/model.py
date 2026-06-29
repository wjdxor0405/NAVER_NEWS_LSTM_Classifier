"""PyTorch 기반 LSTM 기사 분류 모델 정의 모듈."""

from __future__ import annotations

import torch
from torch import nn


class TextLSTMClassifier(nn.Module):
    """Embedding, LSTM, Dropout, Linear 계층으로 구성된 텍스트 분류 모델."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int) -> None:
        """모델에 필요한 계층을 생성한다."""

        super().__init__()                                                      # nn.Module의 초기화 로직을 실행한다.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)     # 정수 토큰을 밀집 임베딩 벡터로 변환한다.
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)            # 문장 순서를 고려하여 정보를 누적하는 LSTM 계층이다.
        self.dropout = nn.Dropout(p=0.5)                                        # 과적합을 줄이기 위해 일부 뉴런 출력을 무작위로 0으로 만든다.
        self.fc = nn.Linear(hidden_dim, num_classes)                            # LSTM 결과를 카테고리 개수만큼의 점수로 변환한다.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 토큰 배열을 받아 각 카테고리에 대한 예측 점수를 반환한다."""

        embedded = self.embedding(x)                                            # [배치, 문장길이]를 [배치, 문장길이, 임베딩차원]으로 변환한다.
        _, (hidden, _) = self.lstm(embedded)                                    # LSTM을 통과시키고 마지막 은닉 상태를 얻는다.
        last_hidden = hidden[-1]                                                # 마지막 LSTM 층의 은닉 상태만 분류에 사용한다.
        dropped = self.dropout(last_hidden)                                     # 과적합 방지를 위해 Dropout을 적용한다.
        logits = self.fc(dropped)                                               # 각 클래스에 대한 원시 점수(logit)를 계산한다.
        return logits                                                           # CrossEntropyLoss에 입력할 점수를 반환한다.
