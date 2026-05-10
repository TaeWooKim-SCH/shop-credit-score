from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


@dataclass
class ThresholdAnalysis:
    f1_optimal: float
    youden_optimal: float
    cost_optimal: float
    pr_curve: pd.DataFrame  # threshold, precision, recall, f1
    cost_summary: pd.DataFrame  # threshold, cost, tp, fp, fn, tn
    f1_at_default: float
    f1_at_f1_optimal: float


class ThresholdOptimizer:
    """확률 → 이진 분류 임계값 최적화. F1, Youden's J, 비용 기반 3가지 방법.

    비용 모델: 그로몽 펀드 투자 의사결정 관점
      - FP (성장 예측 → 실제 정체): 투자 손실 (cost = 5)
      - FN (정체 예측 → 실제 성장): 기회 비용 (cost = 1)
    """

    DEFAULT_THRESHOLD = 0.5
    FP_COST = 5.0
    FN_COST = 1.0

    def analyze(self, y_true: pd.Series, y_proba: np.ndarray) -> ThresholdAnalysis:
        # PR-curve based F1
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        # PR curve의 마지막 점은 임계값 정의 안 됨 → 슬라이스
        thr_pr = thresholds
        f1_pr = f1[:-1]
        precision_pr = precision[:-1]
        recall_pr = recall[:-1]

        f1_optimal = float(thr_pr[np.argmax(f1_pr)])

        # ROC-based Youden's J
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_proba)
        j = tpr - fpr
        youden_optimal = float(thresholds_roc[np.argmax(j)])

        # Cost-based
        unique_thr = np.unique(np.concatenate([thr_pr, thresholds_roc]))
        unique_thr = unique_thr[(unique_thr > 0) & (unique_thr < 1)]
        cost_records = []
        for t in unique_thr:
            pred = (y_proba >= t).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            tn = int(((pred == 0) & (y_true == 0)).sum())
            cost = self.FP_COST * fp + self.FN_COST * fn
            cost_records.append({
                "threshold": float(t), "cost": float(cost),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })
        cost_df = pd.DataFrame(cost_records).sort_values("cost").reset_index(drop=True)
        cost_optimal = float(cost_df.iloc[0]["threshold"])

        f1_at_default = self._f1_at(y_true, y_proba, self.DEFAULT_THRESHOLD)
        f1_at_optimal = self._f1_at(y_true, y_proba, f1_optimal)

        pr_df = pd.DataFrame({
            "threshold": thr_pr,
            "precision": precision_pr,
            "recall": recall_pr,
            "f1": f1_pr,
        })

        return ThresholdAnalysis(
            f1_optimal=f1_optimal,
            youden_optimal=youden_optimal,
            cost_optimal=cost_optimal,
            pr_curve=pr_df,
            cost_summary=cost_df,
            f1_at_default=f1_at_default,
            f1_at_f1_optimal=f1_at_optimal,
        )

    @staticmethod
    def _f1_at(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> float:
        from sklearn.metrics import f1_score
        pred = (y_proba >= threshold).astype(int)
        return float(f1_score(y_true, pred, zero_division=0))
