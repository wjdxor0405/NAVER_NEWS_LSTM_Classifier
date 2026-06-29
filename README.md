
## 프로젝트 구조

```
naver_rnn_project/
├── main.py                  # 실행 진입점
├── requirements.txt
├── app/
│   ├── config.py            # 하이퍼파라미터 설정
│   ├── data.py              # 네이버 뉴스 크롤링 + 샘플 데이터
│   ├── preprocess.py        # 한국어 전처리 (konlpy 선택적)
│   ├── model.py             # LSTM 모델 정의
│   ├── train.py             # 학습·평가·저장
│   └── predict.py           # 추론
└── models/                  # 저장된 모델 파일
```

## 한국어 전처리 (`app/preprocess.py`)

| 기능 | 영문 원본 | 한국어 수정본 |
|------|----------|-------------|
| 정규식 | `[^a-z\s]` 제거 | `[^\uAC00-\uD7A3a-zA-Z0-9\s]` 제거 |
| 토크나이저 | 공백 분리 | konlpy Okt (선택) / 공백 분리 (기본) |
| 불용어 | 영문 관사·전치사 | 한국어 조사·접속사·지시어 (~60개) |
| 최소 토큰 길이 | — | 2글자 미만 제거 |
| max_len | 80 | 30 (제목 길이 맞춤) |

## 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# (선택) konlpy 형태소 분석기 설치 (Java 11 이상 필요)
pip install konlpy

# 학습 및 예측
python main.py
```
