"""학습 결과 시각화 및 평가지표 출력 모듈.

생성하는 차트:
  1. train_curve.png   – Epoch별 Train Loss / Val Loss / Val Accuracy 곡선
  2. confusion_matrix.png – 혼동 행렬 히트맵
  3. metrics_bar.png   – 카테고리별 Precision / Recall / F1 막대 차트
  4. kfold_box.png     – K-Fold Accuracy 분포 박스플롯 (K-Fold 실행 시)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # GUI 없는 서버 환경에서 파일로 저장한다.
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────
def _set_korean_font() -> None:
    """시스템에 설치된 한글 폰트를 matplotlib에 적용한다."""
    import glob

    # Linux 나눔 폰트 ttf 파일을 직접 등록한다 (폰트 캐시 미반영 문제 우회).
    nanum_paths = glob.glob("/usr/share/fonts/truetype/nanum/*.ttf")
    for p in nanum_paths:
        try:
            fm.fontManager.addfont(p)
        except Exception:
            pass

    candidates = [
        "NanumBarunGothic", "NanumGothic", "NanumSquare",
        "Malgun Gothic", "AppleGothic", "UnDotum", "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

_set_korean_font()

# ── 카테고리 한글 표시명 ────────────────────────────────────────────────────
_LABEL_KO = {
    "entertainment": "연예",
    "it":            "IT/과학",
    "sports":        "스포츠",
}


def _label_ko(name: str) -> str:
    return _LABEL_KO.get(name, name)


# ── 1. 학습 곡선 ────────────────────────────────────────────────────────────

def plot_train_curve(
    train_losses: List[float],
    val_losses:   List[float],
    val_accs:     List[float],
    save_path: str,
    best_epoch: Optional[int] = None,
) -> None:
    """Train Loss, Val Loss, Val Accuracy 세 곡선을 하나의 그림에 그린다."""

    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("학습 곡선 (Training Curve)", fontsize=14, fontweight="bold")

    # Loss 그래프
    ax1.plot(epochs, train_losses, "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, val_losses,   "r-s", markersize=3, label="Val Loss")
    if best_epoch is not None:
        ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7,
                    label=f"Best epoch ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("손실 곡선")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy 그래프
    ax2.plot(epochs, val_accs, "g-^", markersize=3, label="Val Accuracy")
    if best_epoch is not None:
        ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7,
                    label=f"Best epoch ({best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("검증 정확도 곡선")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[시각화] 학습 곡선 저장 → {save_path}")


# ── 2. 혼동 행렬 ────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    id_to_label: Dict[int, str],
    save_path: str,
) -> None:
    """혼동 행렬 히트맵을 그린다. 대각선이 정답, 나머지가 오분류이다."""

    label_names = [id_to_label[i] for i in range(len(id_to_label))]
    ko_names    = [_label_ko(n) for n in label_names]
    cm = confusion_matrix(y_true, y_pred)

    # 행 합계로 나눠 비율(행 정규화)로 표현한다.
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("혼동 행렬 (Confusion Matrix)", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm,      cm_norm],
        ["건수",   "비율 (행 정규화)"],
        ["d",      ".2f"],
    ):
        im = ax.imshow(data, interpolation="nearest",
                       cmap="Blues", vmin=0, vmax=data.max())
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(ko_names))); ax.set_xticklabels(ko_names)
        ax.set_yticks(range(len(ko_names))); ax.set_yticklabels(ko_names)
        ax.set_xlabel("예측 라벨")
        ax.set_ylabel("실제 라벨")
        ax.set_title(title)
        thresh = data.max() / 2.0
        for i in range(len(ko_names)):
            for j in range(len(ko_names)):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black",
                        fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[시각화] 혼동 행렬 저장 → {save_path}")


# ── 3. 카테고리별 평가지표 막대 차트 ────────────────────────────────────────

def plot_metrics_bar(
    y_true: List[int],
    y_pred: List[int],
    id_to_label: Dict[int, str],
    save_path: str,
) -> None:
    """카테고리별 Precision / Recall / F1-Score 막대 차트를 그린다."""

    label_names = [id_to_label[i] for i in range(len(id_to_label))]
    ko_names    = [_label_ko(n) for n in label_names]

    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall    = recall_score   (y_true, y_pred, average=None, zero_division=0)
    f1        = f1_score       (y_true, y_pred, average=None, zero_division=0)
    macro_f1  = f1_score       (y_true, y_pred, average="macro", zero_division=0)
    accuracy  = accuracy_score (y_true, y_pred)

    x      = np.arange(len(ko_names))
    width  = 0.25
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, precision, width, label="Precision", color=colors[0], alpha=0.85)
    ax.bar(x,         recall,    width, label="Recall",    color=colors[1], alpha=0.85)
    ax.bar(x + width, f1,        width, label="F1-Score",  color=colors[2], alpha=0.85)

    # 값 레이블 표시
    for bars in [
        ax.bar(x - width, precision, width),
        ax.bar(x,         recall,    width),
        ax.bar(x + width, f1,        width),
    ]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(ko_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(
        f"카테고리별 평가지표\n"
        f"Accuracy: {accuracy:.4f}  |  Macro F1: {macro_f1:.4f}",
        fontsize=12,
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[시각화] 평가지표 막대 차트 저장 → {save_path}")


# ── 4. K-Fold 정확도 박스플롯 ───────────────────────────────────────────────

def plot_kfold_box(
    fold_accs: List[float],
    save_path: str,
) -> None:
    """K-Fold 각 폴드별 정확도 분포를 박스플롯으로 그린다."""

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(fold_accs, patch_artist=True,
                    boxprops=dict(facecolor="#AEC6E8", color="navy"),
                    medianprops=dict(color="red", linewidth=2))
    for i, acc in enumerate(fold_accs, 1):
        ax.scatter(1, acc, zorder=5, color="navy", s=40)
        ax.annotate(f"Fold {i}\n{acc:.3f}",
                    (1.07, acc), fontsize=8, va="center")

    mean_acc = np.mean(fold_accs)
    ax.axhline(mean_acc, color="orange", linestyle="--",
               label=f"Mean: {mean_acc:.4f}")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"K-Fold 교차검증 정확도\n"
        f"Mean: {mean_acc:.4f} ± {np.std(fold_accs):.4f}",
        fontsize=12,
    )
    ax.set_xticks([1]); ax.set_xticklabels(["K-Fold"])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[시각화] K-Fold 박스플롯 저장 → {save_path}")


# ── 5. 콘솔 평가지표 출력 ───────────────────────────────────────────────────

def print_metrics(
    y_true: List[int],
    y_pred: List[int],
    id_to_label: Dict[int, str],
    fold_accs: Optional[List[float]] = None,
) -> None:
    """정확도·Macro F1·카테고리별 리포트를 콘솔에 출력한다."""

    label_names = [id_to_label[i] for i in range(len(id_to_label))]
    accuracy  = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 55)
    print("  📊 최종 평가 지표 요약")
    print("=" * 55)
    print(f"  정확도   (Accuracy)         : {accuracy:.4f}")
    print(f"  Macro F1-Score             : {macro_f1:.4f}")
    print(f"  Weighted F1-Score          : {weighted_f1:.4f}")
    if fold_accs:
        print(f"  K-Fold Mean Accuracy       : {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print("-" * 55)
    print(classification_report(
        y_true, y_pred,
        target_names=label_names,
        zero_division=0,
    ))
    print("=" * 55)
