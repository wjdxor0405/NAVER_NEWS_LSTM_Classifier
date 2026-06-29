"""PyTorch 기반 Bidirectional LSTM + Attention 기사 분류 모델 정의 모듈."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AttentionLayer(nn.Module):
    """LSTM 전체 시퀀스 출력에 소프트맥스 가중치를 적용하는 셀프 어텐션 계층.

    마지막 은닉 상태만 사용하는 기존 방식과 달리, 모든 타임스텝 출력을
    가중 합산하여 문장의 핵심 단어를 강조한다.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # 각 타임스텝의 중요도 점수를 계산하는 선형 계층이다.
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """LSTM 전체 출력에서 가중 평균 컨텍스트 벡터를 계산한다.

        Args:
            lstm_out: [배치, 시퀀스길이, hidden_dim] 형태의 LSTM 출력.

        Returns:
            [배치, hidden_dim] 형태의 어텐션 가중 평균 벡터.
        """
        scores = self.attn(lstm_out).squeeze(-1)        # [배치, 시퀀스길이]
        weights = F.softmax(scores, dim=1).unsqueeze(2) # [배치, 시퀀스길이, 1]
        context = (lstm_out * weights).sum(dim=1)       # [배치, hidden_dim]
        return context


class TextLSTMClassifier(nn.Module):
    """Bidirectional 2-layer LSTM + Attention + LayerNorm 텍스트 분류 모델.

    기존 단방향 단일층 LSTM 대비 개선 사항:
    - Bidirectional: 앞→뒤, 뒤→앞 양방향으로 문맥을 파악한다.
    - 2-layer LSTM: 더 깊은 특징 추출이 가능하다.
    - Attention: 분류에 중요한 단어에 가중치를 부여한다.
    - LayerNorm: 배치 크기가 작은 환경에서 BatchNorm보다 안정적이다.
    - Dropout 0.3: 소량 데이터에서 0.5보다 완화된 정규화를 적용한다.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # 정수 토큰을 밀집 임베딩 벡터로 변환한다. PAD 토큰(0)은 학습하지 않는다.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 양방향 2층 LSTM이다. 출력 hidden_dim은 forward+backward가 합쳐져 2배가 된다.
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 양방향이므로 hidden_dim * 2 차원의 출력에 어텐션을 적용한다.
        self.attention = AttentionLayer(hidden_dim * 2)

        # 어텐션 출력을 정규화하여 학습을 안정시킨다.
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # 분류 전 드롭아웃으로 과적합을 방지한다.
        self.dropout = nn.Dropout(p=dropout)

        # 중간 은닉층을 추가하여 표현력을 높인다.
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 토큰 배열을 받아 각 카테고리에 대한 예측 점수를 반환한다."""

        embedded = self.dropout(self.embedding(x))       # 임베딩에도 드롭아웃 적용
        lstm_out, _ = self.lstm(embedded)                # [배치, 시퀀스, hidden*2]
        context = self.attention(lstm_out)               # [배치, hidden*2]
        normed = self.layer_norm(context)                # LayerNorm 정규화
        hidden = F.relu(self.fc1(self.dropout(normed)))  # 은닉층 + ReLU 활성화
        logits = self.fc2(self.dropout(hidden))          # 최종 분류 점수
        return logits
