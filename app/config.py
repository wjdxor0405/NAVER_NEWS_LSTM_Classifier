"""네이버 뉴스 기사 분류 RNN 프로젝트 설정 파일."""

from dataclasses import dataclass


@dataclass
class Config:
    """학습과 예측에 공통으로 사용하는 하이퍼파라미터를 한 곳에서 관리하는 클래스."""

    max_vocab: int = 5000           # 토큰화에 사용할 최대 단어 수이다.
    max_len: int = 30               # 한국어 뉴스 제목은 영문보다 짧으므로 80→30으로 줄였다.
    embed_dim: int = 64             # 각 단어 정수를 몇 차원의 임베딩 벡터로 바꿀지 정한다.
    hidden_dim: int = 64            # LSTM 내부 은닉 상태의 차원 수이다.
    batch_size: int = 8             # 한 번의 학습 단계에서 모델에 넣을 샘플 개수이다.
    epochs: int = 10                # 전체 학습 데이터를 몇 번 반복해서 학습할지 정한다.
    learning_rate: float = 0.001    # Adam 최적화 알고리즘의 학습률이다.
    test_size: float = 0.2          # 전체 데이터 중 평가 데이터로 사용할 비율이다.
    random_state: int = 42          # 실험 결과를 재현하기 위한 난수 고정값이다.
    model_path: str = "../models/naver_lstm_model.pt"  # 학습된 모델을 저장할 경로이다.
