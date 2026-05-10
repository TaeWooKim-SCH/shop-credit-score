from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..config import DEFAULT_WEIGHTS, Paths
from ..modeling import EvaluationResult, ShapAnalysis


class ModelPlotter:
    """모델 시각화: ROC 곡선 / SHAP 중요도 / 가중치 비교."""

    GROUP_AXIS_LABELS = ["RRI\n(응답개선)", "OPI\n(주문성과)",
                         "SRI\n(감성개선)", "RSI\n(운영안정)"]

    def __init__(self, paths: Paths):
        self.paths = paths

    def plot_roc(self, eval_result: EvaluationResult) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(eval_result.fpr, eval_result.tpr, color="tomato", lw=2,
                label=f"XGBoost (AUC={eval_result.auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(alpha=0.3)
        self._save(fig, "roc_curve.png")

    def plot_shap_importance(self, analysis: ShapAnalysis, top_n: int = 15) -> None:
        top = analysis.importance_df.head(top_n)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="steelblue")
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(f"피처 중요도 (SHAP, Top {top_n})")
        self._save(fig, "shap_importance.png")

    def plot_weight_comparison(self, analysis: ShapAnalysis) -> None:
        keys = list(DEFAULT_WEIGHTS.keys())
        x_pos = np.arange(len(keys))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x_pos - 0.2,
               [analysis.group_weights_normalized[k] for k in keys],
               0.35, label="SHAP 최적화 가중치", color="tomato", alpha=0.8)
        ax.bar(x_pos + 0.2,
               [DEFAULT_WEIGHTS[k] for k in keys],
               0.35, label="기본 가중치", color="steelblue", alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.GROUP_AXIS_LABELS)
        ax.set_ylabel("가중치")
        ax.set_title("지수별 가중치 비교 (SHAP vs 기본값)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        self._save(fig, "weight_comparison.png")

    def _save(self, fig, fname: str) -> None:
        fig.tight_layout()
        fig.savefig(self.paths.output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
