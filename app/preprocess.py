"""텍스트 정제, 토큰화, 패딩, 라벨 인코딩을 담당하는 모듈.

한국어 데이터 전처리를 지원하도록 기존 영문 전처리 함수를 수정했다.
- clean_text()  : 한국어 형태소 분석(konlpy 선택적) 또는 공백 분리 방식 지원
- STOP_WORDS    : 한국어 불용어 목록으로 교체
- 그 외 함수들은 언어 독립적으로 동작하므로 변경 없이 재사용한다.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 한국어 불용어 목록
#   조사·어미·접속사·지시어처럼 분류에 도움이 되지 않는 단어를 제거한다.
# ---------------------------------------------------------------------------
STOP_WORDS: set[str] = {
    # 조사
    "이", "가", "을", "를", "은", "는", "의", "에", "에서", "로", "으로",
    "와", "과", "도", "만", "까지", "부터", "에게", "한테", "께", "이나",
    "나", "이랑", "랑", "조차", "마저", "밖에", "뿐",
    # 접속사·부사
    "그리고", "그러나", "하지만", "또한", "그래서", "따라서", "즉", "또",
    "그런데", "그러므로", "그래도", "왜냐하면", "만약", "비록", "아직",
    # 지시어
    "이것", "그것", "저것", "이런", "그런", "저런", "이번", "그간",
    # 동사·형용사 일반어미(형태소 미분리 환경에서 걸러지는 단어들)
    "있다", "없다", "하다", "되다", "이다", "아니다", "같다", "위해",
    "통해", "대한", "관련", "경우", "이후", "현재", "지난", "올해",
    # 수량·단위 (분류에 불필요)
    "가지", "개", "명", "번", "여", "약",
    # 기타
    "및", "등", "중", "내", "간", "전", "후", "더", "함께", "이상",
}


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """한국어(및 영문 혼용) 기사 문장에서 불필요한 문자와 불용어를 제거한다.

    konlpy가 설치된 환경이면 Okt 형태소 분석기로 명사·동사·형용사만 추출한다.
    설치되지 않은 경우 공백 기준 분리 방식으로 폴백하여 동작한다.

    Args:
        text: 원본 기사 제목 또는 본문 문장.
        remove_stopwords: True이면 STOP_WORDS에 포함된 단어를 제거한다.

    Returns:
        정제된 토큰들을 공백으로 이어 붙인 문자열.
    """
    # 1) 특수문자·이모지 제거 (한글·영문·숫자·공백만 남김)
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", " ", text)

    # 2) 연속 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    # 3) 형태소 분석 (선택적 의존성)
    try:
        from konlpy.tag import Okt  # type: ignore
        okt = Okt()
        # 명사·동사·형용사·외국어만 추출 (조사·어미 자동 제거)
        pos_tags = okt.pos(text, norm=True, stem=True)
        tokens = [
            word for word, pos in pos_tags
            if pos in {"Noun", "Verb", "Adjective", "Foreign", "Alpha"}
            and len(word) >= 2
        ]
    except Exception:
        # konlpy 미설치 → 공백 분리 후 2글자 미만 토큰 제거
        tokens = [w for w in text.split() if len(w) >= 2]

    # 4) 불용어 제거
    if remove_stopwords:
        tokens = [w for w in tokens if w not in STOP_WORDS]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 이하 함수들은 언어 독립적으로 동작하므로 원본 로직을 유지한다.
# ---------------------------------------------------------------------------

def build_vocab(texts: Sequence[str], max_vocab: int) -> Dict[str, int]:
    """학습 데이터에서 자주 등장한 단어를 정수 인덱스로 매핑하는 사전을 만든다."""

    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(text.split())
    most_common = counter.most_common(max_vocab - 2)   # PAD·OOV 자리 제외
    vocab: Dict[str, int] = {"<PAD>": 0, "<OOV>": 1}
    for index, (word, _) in enumerate(most_common, start=2):
        vocab[word] = index
    return vocab


def texts_to_sequences(texts: Sequence[str], vocab: Dict[str, int]) -> List[List[int]]:
    """문장 목록을 정수 토큰 시퀀스 목록으로 변환한다."""

    sequences: List[List[int]] = []
    for text in texts:
        seq = [vocab.get(word, vocab["<OOV>"]) for word in text.split()]
        sequences.append(seq)
    return sequences


def pad_sequences(sequences: Sequence[Sequence[int]], max_len: int) -> np.ndarray:
    """서로 다른 길이의 정수 시퀀스를 동일한 길이의 2차원 배열로 맞춘다."""

    padded = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        truncated = list(seq)[-max_len:]
        padded[i, -len(truncated):] = truncated if truncated else []
    return padded


def encode_labels(labels: Sequence[str]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """문자열 라벨을 정수 라벨로 변환하고 양방향 라벨 사전을 반환한다."""

    label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    encoded = np.array([label_to_id[label] for label in labels], dtype=np.int64)
    return encoded, label_to_id, id_to_label
