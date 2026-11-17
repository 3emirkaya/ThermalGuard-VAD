"""
Evaluation and Ablation Analysis Module.

Field Evaluation and Performance Metrics:
------------------------------------------
1. **Frame-Level AUROC**: Ability to discriminate anomalous from normal frames across sequences.
2. **Frame-Level PR-AUC / F1-Score**: Real-world performance under severe class imbalance
   (where anomalies are rare occurrences).
3. **Pixel-Level AUROC**: Evaluates spatial anomaly localization accuracy via error maps.
4. **Thermal Preprocessing Ablation**:
   * Raw Normalization vs.
   * CLAHE only vs.
   * Full Thermal Pipeline (AGC + FPN + Bilateral + CLAHE)
   benchmarking robustness and AUROC retention under sensor degradation.
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
    Computes frame-level AUROC, PR-AUC, and optimal F1 score.
    Ambiguous frames (label == -1) are excluded from evaluation.
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

    # 3. Optimal threshold & best F1 score
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
    ground_truth_masks: np.ndarray,  # (N, H, W) binary mask {0, 1}
    sample_stride: int = 4,  # Subsample pixels for high evaluation speed
) -> float:
    """
    Computes pixel-level anomaly localization AUROC.
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
    Calibrates operational detection threshold on normal frames to satisfy
    a target false alarm rate (e.g., 5%).
    """
    assert len(normal_scores) > 0
    # Threshold = (100 - FAR) percentile of nominal scores
    p = 100.0 * (1.0 - false_alarm_rate)
    return float(np.percentile(normal_scores, p))
