"""네이버 뉴스 카테고리별 기사 제목 수집 및 샘플 데이터 모듈.

방법 2: 네이버 뉴스 각 카테고리 페이지에 접속하여 뉴스 제목을 가져온다.
        네트워크 오류 또는 파싱 실패 시 내장 샘플 데이터로 자동 폴백한다.
"""

from __future__ import annotations

import time
from typing import List, Tuple

try:
    import requests
    from bs4 import BeautifulSoup
    _CRAWL_AVAILABLE = True
except ImportError:
    _CRAWL_AVAILABLE = False


# ---------------------------------------------------------------------------
# 네이버 뉴스 카테고리 URL 매핑
#   IT/과학  : sid=105   스포츠 : sid=107   연예 : sid=106
# ---------------------------------------------------------------------------
_NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.naver.com/",
    "Accept-Language": "ko-KR,ko;q=0.9",
}

_NAVER_URLS = {
    "it":            "https://news.naver.com/section/105",
    "sports":        "https://news.naver.com/section/107",
    "entertainment": "https://news.naver.com/section/106",
}

# CSS 셀렉터 우선순위 목록 (네이버 HTML 구조가 변경될 경우를 대비)
_TITLE_SELECTORS = [
    "strong.sa_text_strong",          # 섹션 메인 헤드라인
    "div.sa_text strong",             # 섹션 서브 카드
    "a.nclicks\\(cls_art\\) strong",  # 구형 레이아웃
    "dt.photo a",                     # 일부 뷰 모드
    "ul.type06_headline li dt a",     # 리스트 모드
    "ul.type06 li dt a",
]

_REQUEST_TIMEOUT = 8   # 초
_MIN_TITLES = 5        # 크롤링 성공 판단 최소 제목 수


def _fetch_titles(url: str, label: str) -> List[Tuple[str, str]]:
    """네이버 뉴스 페이지에서 기사 제목을 크롤링하여 (제목, 라벨) 튜플 목록으로 반환한다."""
    if not _CRAWL_AVAILABLE:
        raise ImportError("requests 또는 beautifulsoup4 패키지가 설치되어 있지 않습니다.")

    response = requests.get(url, headers=_NAVER_HEADERS, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()
    response.encoding = "utf-8"

    soup = BeautifulSoup(response.text, "html.parser")
    titles: List[str] = []

    for selector in _TITLE_SELECTORS:
        elements = soup.select(selector)
        for el in elements:
            text = el.get_text(strip=True)
            if text and len(text) >= 8:   # 너무 짧은 텍스트(버튼·레이블 등) 제외
                titles.append(text)
        if len(titles) >= _MIN_TITLES:
            break

    # 중복 제거 (순서 유지)
    seen: set[str] = set()
    unique: List[str] = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return [(title, label) for title in unique]


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def get_news_it() -> List[Tuple[str, str]]:
    """네이버 뉴스 IT/과학 섹션에서 기사 제목을 가져온다.

    Returns:
        List[Tuple[str, str]]: (기사 제목, "it") 형태의 튜플 목록.
        크롤링 실패 시 내장 샘플 데이터를 반환한다.
    """
    try:
        results = _fetch_titles(_NAVER_URLS["it"], "it")
        if len(results) >= _MIN_TITLES:
            print(f"[data] IT 뉴스 {len(results)}건 크롤링 완료")
            return results
        raise ValueError(f"수집된 IT 제목이 {len(results)}건으로 부족합니다.")
    except Exception as exc:
        print(f"[data] IT 크롤링 실패 ({exc}) → 샘플 데이터 사용")
        return _SAMPLE_IT


def get_news_sports() -> List[Tuple[str, str]]:
    """네이버 뉴스 스포츠 섹션에서 기사 제목을 가져온다.

    Returns:
        List[Tuple[str, str]]: (기사 제목, "sports") 형태의 튜플 목록.
        크롤링 실패 시 내장 샘플 데이터를 반환한다.
    """
    try:
        results = _fetch_titles(_NAVER_URLS["sports"], "sports")
        if len(results) >= _MIN_TITLES:
            print(f"[data] 스포츠 뉴스 {len(results)}건 크롤링 완료")
            return results
        raise ValueError(f"수집된 스포츠 제목이 {len(results)}건으로 부족합니다.")
    except Exception as exc:
        print(f"[data] 스포츠 크롤링 실패 ({exc}) → 샘플 데이터 사용")
        return _SAMPLE_SPORTS


def get_news_entertainment() -> List[Tuple[str, str]]:
    """네이버 뉴스 연예 섹션에서 기사 제목을 가져온다.

    Returns:
        List[Tuple[str, str]]: (기사 제목, "entertainment") 형태의 튜플 목록.
        크롤링 실패 시 내장 샘플 데이터를 반환한다.
    """
    try:
        results = _fetch_titles(_NAVER_URLS["entertainment"], "entertainment")
        if len(results) >= _MIN_TITLES:
            print(f"[data] 연예 뉴스 {len(results)}건 크롤링 완료")
            return results
        raise ValueError(f"수집된 연예 제목이 {len(results)}건으로 부족합니다.")
    except Exception as exc:
        print(f"[data] 연예 크롤링 실패 ({exc}) → 샘플 데이터 사용")
        return _SAMPLE_ENTERTAINMENT


def load_sample_data() -> Tuple[List[str], List[str]]:
    """세 카테고리 뉴스 제목 전체를 합쳐 기사 목록과 라벨 목록으로 분리해서 반환한다.

    크롤링이 가능한 환경이면 실시간 네이버 뉴스 제목을, 아니면 내장 샘플을 사용한다.
    """
    all_data: List[Tuple[str, str]] = []
    all_data.extend(get_news_it())
    time.sleep(0.3)           # 연속 요청 간격 (크롤링 예절)
    all_data.extend(get_news_sports())
    time.sleep(0.3)
    all_data.extend(get_news_entertainment())

    texts  = [text  for text, _     in all_data]
    labels = [label for _,    label in all_data]
    return texts, labels


# ---------------------------------------------------------------------------
# 내장 샘플 데이터 (크롤링 폴백용) — 실제 네이버 뉴스 제목 형식 반영
# ---------------------------------------------------------------------------

_SAMPLE_IT: List[Tuple[str, str]] = [
    ("애플, 새 아이폰에 AI 기반 번역 기능 탑재 예정", "it"),
    ("삼성전자 갤럭시 폴더블폰 신형 출시 임박…스펙 유출", "it"),
    ("구글 딥마인드, 단백질 구조 예측 모델 정확도 대폭 향상", "it"),
    ("네이버 클라우드, 생성형 AI 서비스 기업 고객 대상 확대", "it"),
    ("카카오, 멀티모달 AI 모델 공개…이미지·텍스트 동시 처리", "it"),
    ("국내 스타트업, 반도체 설계 자동화 소프트웨어로 해외 진출", "it"),
    ("메타, 증강현실 안경 신제품 발표…실시간 번역 지원", "it"),
    ("LG전자 OLED TV, 글로벌 시장점유율 역대 최고 기록", "it"),
    ("KT·SKT, 6G 통신 기술 공동 연구 협약 체결", "it"),
    ("인텔, 차세대 AI 가속 칩 아키텍처 공개…엔비디아 맞대결", "it"),
    ("오픈AI, GPT 최신 버전 멀티에이전트 기능 강화", "it"),
    ("토스, 금융 데이터 분석 AI 서비스 출시…개인 맞춤 자산 관리", "it"),
    ("현대차, 자율주행 레벨4 소프트웨어 도심 실증 완료", "it"),
    ("클라우드플레어, 한국 데이터센터 용량 두 배 확장", "it"),
    ("MS 코파일럿, 오피스 365 전 제품에 기본 탑재 결정", "it"),
]

_SAMPLE_SPORTS: List[Tuple[str, str]] = [
    ("손흥민, 리그 10호 골 폭발…토트넘 3연승 견인", "sports"),
    ("류현진, 복귀전 6이닝 무실점…팀 승리 이끌어", "sports"),
    ("김민재, 분데스리가 베스트 XI 선정…시즌 활약 인정", "sports"),
    ("한국 여자 배구 대표팀, 세계선수권 8강 진출 확정", "sports"),
    ("KBO 리그 선두 경쟁 치열…SSG·LG 1게임 차 접전", "sports"),
    ("이강인, 파리 생제르맹 선발 출격…멀티 어시스트 맹활약", "sports"),
    ("황선우, 세계수영선수권 자유형 결선 진출…메달 도전", "sports"),
    ("NBA 플레이오프 4강 대진 확정…두 경기 동시 개막", "sports"),
    ("국내 골프 신예, LPGA 투어 데뷔전 공동 2위 기록", "sports"),
    ("KIA 타이거즈, 9회말 끝내기 홈런으로 극적 역전승", "sports"),
    ("대한민국 U-23 축구, 아시안게임 조별리그 전승 통과", "sports"),
    ("안세영, 배드민턴 세계랭킹 1위 굳히기…시즌 7번째 우승", "sports"),
    ("박태준, 탁구 세계선수권 단식 4강 진출 쾌거", "sports"),
    ("프리미어리그 득점왕 경쟁 막판 역전…최종전이 관건", "sports"),
    ("한국 남자 농구, 아시아컵 예선 4연승으로 본선 조기 확정", "sports"),
]

_SAMPLE_ENTERTAINMENT: List[Tuple[str, str]] = [
    ("BTS 슈가, 솔로 앨범 발매 첫 주 빌보드 200 진입", "entertainment"),
    ("영화 '서울의 봄', 누적 관객 1300만 돌파 기념 특별 상영", "entertainment"),
    ("블랙핑크 제니, 할리우드 영화 주연 데뷔 확정", "entertainment"),
    ("tvN 새 드라마, 방영 첫 날 자체 최고 시청률 경신", "entertainment"),
    ("아이유, 전국 투어 콘서트 전석 매진…추가 공연 검토", "entertainment"),
    ("봉준호 감독 신작, 칸 영화제 경쟁 부문 공식 초청", "entertainment"),
    ("뉴진스, 일본 데뷔 싱글 오리콘 차트 1위 등극", "entertainment"),
    ("배우 마동석, 마블 시리즈 시즌2 출연 계약 완료", "entertainment"),
    ("세븐틴, 미국 스타디움 투어 성황리 마무리…현지 언론 극찬", "entertainment"),
    ("MBC 예능 프로그램, 넷플릭스 글로벌 동시 방영 결정", "entertainment"),
    ("임영웅, 콘서트 티켓 오픈 10분 만에 전석 매진 기록", "entertainment"),
    ("스트레이키즈, 美 타임지 선정 '올해의 아티스트' 후보 등재", "entertainment"),
    ("배우 송강호, 베니스 영화제 심사위원장 위촉", "entertainment"),
    ("오징어게임 시즌3, 넷플릭스 전 세계 동시 공개 일정 발표", "entertainment"),
    ("가수 싸이, 흠뻑쇼 전국 투어 출동 선언…20개 도시 순회", "entertainment"),
]
