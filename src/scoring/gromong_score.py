from __future__ import annotations

import pandas as pd

from ..config import DEFAULT_WEIGHTS, INDEX_KEY_MAP


class GromongScoreCalculator:
    """4개 지수 + 가중치(SHAP/기본) → 그로몽 스코어 (0~100)."""

    SCALE = 100
    PRECISION = 2

    def compute(
        self,
        df: pd.DataFrame,
        shap_weights: dict[str, float],
    ) -> pd.DataFrame:
        out = df.copy()
        out["gromong_score"] = self._weighted_sum(out, shap_weights).round(self.PRECISION)
        out["gromong_score_default"] = self._weighted_sum(out, DEFAULT_WEIGHTS).round(self.PRECISION)
        return out

    def _weighted_sum(self, df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
        total = sum(df[idx_col] * weights[group_name]
                    for idx_col, group_name in INDEX_KEY_MAP.items())
        return total * self.SCALE
