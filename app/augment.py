"""소량 한국어 뉴스 제목 데이터를 증강하는 모듈.

45개 샘플만으로는 LSTM 학습이 어렵기 때문에,
원본 제목을 변형하여 학습 데이터를 n배로 확대한다.

지원 증강 기법:
  1. 랜덤 단어 삭제 (Random Deletion)
  2. 랜덤 단어 교환 (Random Swap)
  3. 역방향 토큰 나열 (Token Reverse) – 어순 변화 학습
"""

from __future__ import annotations

import random
from typing import List, Tuple


def _random_deletion(tokens: List[str], p: float = 0.15) -> List[str]:
    """각 토큰을 p 확률로 삭제한다. 토큰이 1개면 그대로 반환한다."""
    if len(tokens) <= 1:
        return tokens
    result = [w for w in tokens if random.random() > p]
    return result if result else [random.choice(tokens)]  # 전부 삭제되면 1개 복원


def _random_swap(tokens: List[str], n: int = 1) -> List[str]:
    """n 쌍의 무작위 위치를 서로 교환한다."""
    tokens = tokens.copy()
    length = len(tokens)
    if length < 2:
        return tokens
    for _ in range(n):
        i, j = random.sample(range(length), 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return tokens


def _token_reverse(tokens: List[str]) -> List[str]:
    """토큰 순서를 역방향으로 나열한다."""
    return tokens[::-1]


def augment_data(
    texts: List[str],
    labels: List[str],
    n: int = 4,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """원본 데이터를 n배로 증강하여 반환한다.

    원본 데이터를 보존하고 증강 샘플을 추가한다.
    각 샘플마다 n개의 변형 버전을 생성한다.

    Args:
        texts:  정제된(clean_text 적용 후) 기사 제목 목록.
        labels: 각 제목에 대응하는 카테고리 라벨 목록.
        n:      원본 1건당 생성할 증강 샘플 수.
        seed:   재현성을 위한 난수 고정값.

    Returns:
        (증강 포함 전체 texts, 증강 포함 전체 labels) 튜플.
    """
    random.seed(seed)

    aug_texts: List[str]  = []
    aug_labels: List[str] = []

    # 증강 함수 풀 – 각 샘플에 순환 적용한다.
    techniques = [
        lambda t: _random_deletion(t, p=0.15),
        lambda t: _random_swap(t, n=1),
        lambda t: _random_deletion(_random_swap(t, n=1), p=0.10),
        lambda t: _token_reverse(t),
    ]

    for text, label in zip(texts, labels):
        tokens = text.split()
        for i in range(n):
            fn = techniques[i % len(techniques)]
            new_tokens = fn(tokens)
            if new_tokens:                         # 빈 결과 방지
                aug_texts.append(" ".join(new_tokens))
                aug_labels.append(label)

    # 원본 + 증강 순서로 합친다.
    all_texts  = texts  + aug_texts
    all_labels = labels + aug_labels
    return all_texts, all_labels
