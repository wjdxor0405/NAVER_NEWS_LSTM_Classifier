"""데이터 전처리·증강, 모델 학습·평가·저장, 시각화를 수행하는 모듈."""

from __future__ import annotations

import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.augment import augment_data
from app.config import Config
from app.data import load_sample_data
from app.model import TextLSTMClassifier
from app.preprocess import (
    build_vocab, clean_text, encode_labels,
    pad_sequences, texts_to_sequences,
)
from app.visualize import (
    plot_confusion_matrix, plot_kfold_box,
    plot_metrics_bar, plot_train_curve, print_metrics,
)


# ── 유틸 ────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """학습 결과가 최대한 동일하게 재현되도록 난수를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """NumPy 배열을 DataLoader로 변환한다."""
    ds = TensorDataset(torch.tensor(x, dtype=torch.long),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _build_model(vocab_size: int, num_classes: int, config: Config) -> TextLSTMClassifier:
    """설정값에 따라 TextLSTMClassifier를 생성한다."""
    return TextLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_classes=num_classes,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )


# ── 단일 Epoch 학습 ─────────────────────────────────────────────────────────

def _train_epoch(
    model: TextLSTMClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """한 epoch 동안 학습하고 평균 loss를 반환한다."""
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── 단일 Epoch 평가 ─────────────────────────────────────────────────────────

def _eval_epoch(
    model: TextLSTMClassifier,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, List[int], List[int]]:
    """한 epoch 동안 평가하고 (평균 loss, 예측값, 정답값)을 반환한다."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            total_loss += criterion(logits, batch_y).item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.tolist())
            all_targets.extend(batch_y.tolist())
    return total_loss / len(loader), all_preds, all_targets


# ── 전체 학습 루프 (EarlyStopping + ReduceLR + 최적 모델 저장) ──────────────

def _fit(
    model: TextLSTMClassifier,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    config:       Config,
    verbose:      bool = True,
) -> Tuple[List[float], List[float], List[float], int]:
    """EarlyStopping이 적용된 학습 루프.

    Returns:
        (train_losses, val_losses, val_accs, best_epoch)
    """
    # Label Smoothing CrossEntropy: 과자신감으로 인한 과적합을 줄인다.
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    # AdamW: Adam + L2 weight decay가 분리 적용되어 정규화 효과가 더 정확하다.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # ReduceLROnPlateau: val_loss가 개선되지 않으면 학습률을 절반으로 낮춘다.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5,
    )

    train_losses, val_losses, val_accs = [], [], []
    best_val_loss = float("inf")
    best_state    = None
    best_epoch    = 1
    no_improve    = 0

    for epoch in range(1, config.epochs + 1):
        tr_loss = _train_epoch(model, train_loader, criterion, optimizer)
        val_loss, preds, targets = _eval_epoch(model, val_loader, criterion)
        val_acc  = accuracy_score(targets, preds)
        scheduler.step(val_loss)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Best 모델 갱신
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch
            no_improve    = 0
        else:
            no_improve += 1

        if verbose:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d}/{config.epochs} | "
                f"TrainLoss: {tr_loss:.4f} | "
                f"ValLoss: {val_loss:.4f} | "
                f"ValAcc: {val_acc:.4f} | "
                f"LR: {lr_now:.2e}"
            )

        # EarlyStopping
        if no_improve >= config.patience:
            print(f"\n[EarlyStopping] {config.patience} epoch 동안 개선 없음 → epoch {epoch}에서 중단")
            break

    # 최적 가중치 복원
    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses, val_accs, best_epoch


# ── K-Fold 교차검증 ─────────────────────────────────────────────────────────

def _kfold_cv(
    x: np.ndarray,
    y: np.ndarray,
    vocab_size: int,
    num_classes: int,
    config: Config,
    k: int = 5,
) -> List[float]:
    """K-Fold 교차검증으로 각 폴드의 정확도를 반환한다."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.random_state)
    fold_accs: List[float] = []
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    print(f"\n[K-Fold] {k}-Fold 교차검증 시작...")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(x, y), 1):
        set_seed(config.random_state + fold)
        model = _build_model(vocab_size, num_classes, config)
        tr_ld = _make_loader(x[tr_idx], y[tr_idx], config.batch_size, shuffle=True)
        va_ld = _make_loader(x[val_idx], y[val_idx], config.batch_size, shuffle=False)
        _fit(model, tr_ld, va_ld, config, verbose=False)
        _, preds, targets = _eval_epoch(model, va_ld, criterion)
        acc = accuracy_score(targets, preds)
        fold_accs.append(acc)
        print(f"  Fold {fold}/{k} → Accuracy: {acc:.4f}")

    print(f"[K-Fold] Mean: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}\n")
    return fold_accs


# ── 메인 학습 함수 ───────────────────────────────────────────────────────────

def train_model(
    config: Config,
    run_kfold: bool = True,
) -> Tuple[TextLSTMClassifier, Dict[str, object]]:
    """네이버 뉴스 기사 데이터를 사용해 고도화된 LSTM 분류 모델을 학습한다.

    개선 사항:
      - 데이터 증강으로 학습 샘플 확대
      - Bidirectional 2-layer LSTM + Attention
      - AdamW + Label Smoothing + Gradient Clipping
      - ReduceLROnPlateau + EarlyStopping
      - K-Fold 교차검증으로 안정적인 성능 추정
      - 4종 시각화 자동 저장

    Args:
        config:    하이퍼파라미터 설정 객체.
        run_kfold: True이면 K-Fold 교차검증을 추가 실행한다.

    Returns:
        (학습된 모델, 메타데이터 딕셔너리)
    """
    set_seed(config.random_state)

    # ── 1. 데이터 수집 및 전처리 ──────────────────────────────────────────
    print("=" * 55)
    print("  [1/5] 데이터 로딩 및 전처리")
    print("=" * 55)
    raw_texts, labels = load_sample_data()
    cleaned_texts = [clean_text(text) for text in raw_texts]
    print(f"  원본 샘플 수: {len(cleaned_texts)}")

    # ── 2. 데이터 증강 ────────────────────────────────────────────────────
    print(f"\n  [2/5] 데이터 증강 (×{config.augment_n + 1})")
    aug_texts, aug_labels = augment_data(
        cleaned_texts, labels, n=config.augment_n, seed=config.random_state
    )
    print(f"  증강 후 샘플 수: {len(aug_texts)}")

    # ── 3. 어휘·시퀀스 변환 ───────────────────────────────────────────────
    vocab    = build_vocab(aug_texts, config.max_vocab)
    seqs     = texts_to_sequences(aug_texts, vocab)
    x        = pad_sequences(seqs, config.max_len)
    y_full, label_to_id, id_to_label = encode_labels(aug_labels)
    num_classes = len(label_to_id)
    vocab_size  = len(vocab)
    print(f"  어휘 사전 크기: {vocab_size}")

    # ── 4. Train / Test 분할 ──────────────────────────────────────────────
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_full,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y_full,
    )
    print(f"  학습: {len(x_train)}  평가: {len(x_test)}")

    # ── 5. K-Fold 교차검증 (선택) ─────────────────────────────────────────
    fold_accs: Optional[List[float]] = None
    if run_kfold:
        print("\n  [3/5] K-Fold 교차검증")
        fold_accs = _kfold_cv(x_train, y_train, vocab_size, num_classes, config, k=5)

    # ── 6. 최종 모델 학습 ─────────────────────────────────────────────────
    print("\n  [4/5] 최종 모델 학습 (Train/Val split)")
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train,
        test_size=0.15,
        random_state=config.random_state,
        stratify=y_train,
    )
    train_loader = _make_loader(x_tr,  y_tr,  config.batch_size, shuffle=True)
    val_loader   = _make_loader(x_val, y_val, config.batch_size, shuffle=False)
    test_loader  = _make_loader(x_test, y_test, config.batch_size, shuffle=False)

    model = _build_model(vocab_size, num_classes, config)
    train_losses, val_losses, val_accs, best_epoch = _fit(
        model, train_loader, val_loader, config, verbose=True
    )

    # ── 7. 테스트셋 최종 평가 ─────────────────────────────────────────────
    print("\n  [5/5] 테스트셋 최종 평가 및 시각화")
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    _, final_preds, final_targets = _eval_epoch(model, test_loader, criterion)
    accuracy = accuracy_score(final_targets, final_preds)

    print_metrics(final_targets, final_preds, id_to_label, fold_accs)

    # ── 8. 시각화 저장 ────────────────────────────────────────────────────
    os.makedirs(config.plot_dir, exist_ok=True)

    plot_train_curve(
        train_losses, val_losses, val_accs,
        save_path=os.path.join(config.plot_dir, "train_curve.png"),
        best_epoch=best_epoch,
    )
    plot_confusion_matrix(
        final_targets, final_preds, id_to_label,
        save_path=os.path.join(config.plot_dir, "confusion_matrix.png"),
    )
    plot_metrics_bar(
        final_targets, final_preds, id_to_label,
        save_path=os.path.join(config.plot_dir, "metrics_bar.png"),
    )
    if fold_accs:
        plot_kfold_box(
            fold_accs,
            save_path=os.path.join(config.plot_dir, "kfold_box.png"),
        )

    # ── 9. 모델 저장 ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(config.model_path)), exist_ok=True)
    torch.save(model.state_dict(), config.model_path)
    meta = {
        "vocab":        vocab,
        "label_to_id":  label_to_id,
        "id_to_label":  id_to_label,
        "config":       config,
    }
    with open(config.model_path.replace(".pt", "_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"\n모델 저장 완료 → {config.model_path}")

    return model, {**meta, "accuracy": accuracy, "fold_accs": fold_accs}
