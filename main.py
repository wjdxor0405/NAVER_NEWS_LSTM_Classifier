"""PyCharm에서 바로 실행할 수 있는 네이버 뉴스 RNN 분류 프로젝트 진입점."""

from app.config import Config
from app.predict import load_artifacts, predict_text
from app.train import train_model


if __name__ == "__main__":
    # 프로젝트 전역 설정 객체를 생성한다.
    config = Config()

    # 네이버 뉴스 기사 데이터로 LSTM 분류 모델을 학습하고 모델 파일을 저장한다.
    train_model(config)

    # 저장된 모델과 단어 사전, 라벨 사전을 다시 불러와 실제 서비스 예측 흐름을 확인한다.
    model, metadata = load_artifacts(config)

    # 새롭게 분류할 테스트 기사 제목을 준비한다.
    sample_news = "삼성전자, 차세대 AI 반도체 칩 양산 돌입…글로벌 시장 공략"

    # 테스트 기사 제목을 모델에 입력하여 예측 카테고리를 얻는다.
    predicted_label = predict_text(sample_news, model, metadata, config)

    # 최종 예측 결과를 화면에 출력한다.
    print("\n새 기사:", sample_news)
    print("예측 카테고리:", predicted_label)
