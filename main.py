"""PyCharm에서 바로 실행할 수 있는 네이버 뉴스 RNN 분류 프로젝트 진입점."""

from app.config import Config
from app.predict import load_artifacts, predict_text, predict_with_proba
from app.train import train_model


if __name__ == "__main__":
    config = Config()

    # ── 학습 + K-Fold 검증 + 시각화 ─────────────────────────────────────
    model, metadata = train_model(config, run_kfold=True)

    # ── 저장된 모델로 추론 테스트 ─────────────────────────────────────────
    model, metadata = load_artifacts(config)

    test_headlines = [
        "삼성전자, 차세대 AI 반도체 칩 양산 돌입…글로벌 시장 공략",
        "손흥민 결승골로 토트넘 챔피언스리그 16강 진출 확정",
        "BTS 지민, 솔로 앨범 빌보드 1위 등극…새 역사 썼다",
    ]

    print("\n" + "=" * 55)
    print("  🔍 신규 기사 분류 예측")
    print("=" * 55)
    for headline in test_headlines:
        label = predict_text(headline, model, metadata, config)
        proba = predict_with_proba(headline, model, metadata, config)
        proba_str = "  |  ".join(f"{k}: {v:.3f}" for k, v in sorted(proba.items()))
        print(f"\n  기사: {headline}")
        print(f"  예측: [{label}]")
        print(f"  확률: {proba_str}")
