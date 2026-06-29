"""네이버 뉴스 기사 분류 RNN 프로젝트 설정 파일."""

from dataclasses import dataclass, field


@dataclass
class Config:
    """학습과 예측에 공통으로 사용하는 하이퍼파라미터를 한 곳에서 관리하는 클래스."""

    # ── 전처리 ──────────────────────────────────────────────────────────────
    max_vocab: int   = 5000     # 토큰화에 사용할 최대 단어 수이다.
    max_len:   int   = 30       # 한국어 뉴스 제목 기준 최대 토큰 길이이다.

    # ── 모델 구조 ────────────────────────────────────────────────────────────
    embed_dim:  int   = 128     # 단어 임베딩 차원 (64→128: 표현력 향상).
    hidden_dim: int   = 128     # LSTM 은닉 상태 차원 (64→128).
    num_layers: int   = 2       # LSTM 스택 층수 (Bidirectional이므로 실제 파라미터 4배).
    dropout:    float = 0.3     # 소량 데이터에서 0.5→0.3으로 완화.

    # ── 학습 ────────────────────────────────────────────────────────────────
    batch_size:    int   = 8       # 미니배치 크기.
    epochs:        int   = 60      # 소량 데이터에서는 epoch를 충분히 늘린다.
    learning_rate: float = 5e-4    # AdamW 초기 학습률.
    weight_decay:  float = 1e-4    # AdamW L2 정규화 강도.
    label_smoothing: float = 0.1   # Label Smoothing: 과자신감 방지.
    patience:      int   = 15      # EarlyStopping 인내 epoch 수.

    # ── 데이터 ──────────────────────────────────────────────────────────────
    test_size:    float = 0.2   # 평가 데이터 비율.
    augment_n:    int   = 4     # 데이터 증강 배수 (원본 1 + 증강 n).
    random_state: int   = 42    # 재현성을 위한 난수 고정값.

    # ── 경로 ────────────────────────────────────────────────────────────────
    model_path: str = "../models/naver_lstm_model.pt"   # 모델 저장 경로.
    plot_dir:   str = "../models"                        # 시각화 이미지 저장 폴더.
