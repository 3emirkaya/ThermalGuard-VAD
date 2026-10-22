"""
Değerlendirme (Evaluation) ve Ablasyon Analizi Modülü.

Saha Değerlendirme ve Performans Metrikleri:
--------------------------------------------
1. **Frame-Level AUROC**: Videodaki her kare için anomali olup olmadığını ayırt etme başarısı.
2. **Frame-Level PR-AUC / F1-Score**: Dengesiz veri setlerinde (anomali nadirdir)
   gerçekçi performans göstergesi.
3. **Pixel-Level AUROC**: Anomalinin karenin neresinde olduğunu (uzamsal lokalizasyon)
   hata haritası üzerinden ölçer.
4. **Termal Ön İşleme Ablasyonu**:
   * Ham (Raw) Normalizasyon vs.
   * Yalnızca CLAHE vs.
   * Tam Termal Hat (AGC + FPN + Bilateral + CLAHE)
   arasındaki AUROC ve Gürültü Dayanımı (Robustness) farkı.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn import metrics


def compute_frame_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Kare düzeyinde AUROC, PR-AUC ve optimal F1 skorunu hesaplar.
    Etiketi -1 olan (belirsiz) kareler değerlendirmeye dahil edilmez.
    """
    valid_mask = labels >= 0
    y_true = labels[valid_mask]
    y_score = scores[valid_mask]

    if len(np.unique(y_true)) < 2:
        return {
            "auroc": 0.5,
            "pr_auc": float(np.mean(y_true)),
            "best_f1": 0.0,
            "best_threshold": 0.5,
        }

    # 1. AUROC (Receiver Operating Characteristic)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auroc = float(metrics.auc(fpr, tpr))

    # 2. PR-AUC (Precision-Recall)
    precision, recall, pr_thresh = metrics.precision_recall_curve(y_true, y_score)
    pr_auc = float(metrics.auc(recall, precision))

    # 3. Optimal Eşik & En İyi F1 Skoru
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.nanargmax(f1_scores)
    best_f1 = float(f1_scores[best_idx])
    best_threshold = float(pr_thresh[min(best_idx, len(pr_thresh) - 1)])

    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
    }


def compute_pixel_auroc(
    error_maps: np.ndarray,  # (N, H, W)
    ground_truth_masks: np.ndarray,  # (N, H, W) ikili maske {0, 1}
    sample_stride: int = 4,  # Hız için pikselleri alt-örnekle
) -> float:
    """
    Piksel bazında lokalizasyon AUROC değerini hesaplar.
    """
    sub_maps = error_maps[:, ::sample_stride, ::sample_stride].ravel()
    sub_gt = ground_truth_masks[:, ::sample_stride, ::sample_stride].ravel()

    if len(np.unique(sub_gt)) < 2:
        return 0.5

    fpr, tpr, _ = metrics.roc_curve(sub_gt, sub_maps)
    return float(metrics.auc(fpr, tpr))


def calibrate_normal_threshold(
    normal_scores: np.ndarray,
    false_alarm_rate: float = 0.05,
) -> float:
    """
    Sadece normal durumlardan alınan skorlara göre hedef yanlış alarm oranını
    (örn. %5) sağlayacak eşik değerini belirler.
    """
    assert len(normal_scores) > 0
    # Eşik = normal skorların (100 - FAR) persentili
    p = 100.0 * (1.0 - false_alarm_rate)
    return float(np.percentile(normal_scores, p))
