# 네이버 뉴스 LSTM 기사 분류 프로젝트

BBC RNN Classifier를 기반으로, **네이버 뉴스 한국어 기사 제목**을 세 가지 카테고리로 분류하는 LSTM 모델 실습 프로젝트입니다.

## 분류 카테고리

| 라벨 | 네이버 섹션 | 섹션 ID |
|------|-----------|---------|
| `it` | IT/과학 | 105 |
| `sports` | 스포츠 | 107 |
| `entertainment` | 연예 | 106 |

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

## 데이터 수집 전략

`app/data.py`는 **방법 1·2 하이브리드** 방식으로 동작합니다.

### 방법 2 (우선): 네이버 뉴스 실시간 크롤링
```python
from app.data import get_news_it, get_news_sports, get_news_entertainment

it_data    = get_news_it()            # [(제목, "it"), ...]
sports_data = get_news_sports()       # [(제목, "sports"), ...]
ent_data   = get_news_entertainment() # [(제목, "entertainment"), ...]
```
- 접근 URL: `https://news.naver.com/section/{105|107|106}`
- 네트워크 오류·파싱 실패 시 방법 1로 자동 폴백

### 방법 1 (폴백): 내장 샘플 데이터
각 카테고리별 15건의 실제 뉴스 제목 형식 샘플이 모듈에 내장되어 있습니다.

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

## 개별 함수 테스트

```python
from app.data import get_news_it, get_news_sports, get_news_entertainment, load_sample_data
from app.preprocess import clean_text

# 크롤링 테스트
print(get_news_it()[:3])

# 전처리 테스트
print(clean_text("삼성전자, AI 반도체 칩 출시…글로벌 시장 공략"))
# 출력 예: "삼성전자 반도체 출시 글로벌 시장 공략"
```
