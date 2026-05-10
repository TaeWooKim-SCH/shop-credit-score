from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier


@dataclass
class ShapAnalysis:
    importance_df: pd.DataFrame
    group_weights_raw: dict[str, float]
    group_weights_normalized: dict[str, float]


class ShapAnalyzer:
    """SHAP 기반 피처 중요도 + 4개 그룹 가중치 산출."""

    def analyze(
        self,
        model: XGBClassifier,
        X_test: pd.DataFrame,
        feature_groups: dict[str, list[str]],
    ) -> ShapAnalysis:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        feature_cols = list(X_test.columns)

        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        raw_w = self._group_raw_weights(shap_vals, feature_cols, feature_groups)
        total = sum(raw_w.values()) or 1.0
        normalized = {k: v / total for k, v in raw_w.items()}
        return ShapAnalysis(importance_df, raw_w, normalized)

    def _group_raw_weights(
        self,
        shap_vals: np.ndarray,
        feature_cols: list[str],
        feature_groups: dict[str, list[str]],
    ) -> dict[str, float]:
        weights: dict[str, float] = {}
        for name, cols in feature_groups.items():
            cols_in = [c for c in cols if c in feature_cols]
            if cols_in:
                idx_list = [feature_cols.index(c) for c in cols_in]
                weights[name] = float(np.abs(shap_vals[:, idx_list]).mean())
            else:
                weights[name] = 0.0
        return weights
