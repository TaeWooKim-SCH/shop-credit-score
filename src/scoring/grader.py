from __future__ import annotations

import pandas as pd

from ..config import GRADE_BINS, GRADE_LABELS


class GradeAssigner:
    """그로몽 스코어 → D/C/B/A/S 등급 + 평점 보정 규칙."""

    NEW_SHOP_GRADE = "C"
    LOW_RATING_THRESHOLD = 3.5
    DEMOTION_CHAIN = [("S", "A"), ("A", "B"), ("B", "C"), ("C", "D")]

    def __init__(
        self,
        bins: list[int] | None = None,
        labels: list[str] | None = None,
    ):
        self.bins = bins or GRADE_BINS
        self.labels = labels or GRADE_LABELS

    def assign(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["grade"] = pd.cut(
            out["gromong_score"], bins=self.bins,
            labels=self.labels, include_lowest=True,
        ).astype(str)
        out = self._apply_no_review_rule(out)
        out = self._apply_low_rating_demotion(out)
        return out

    def _apply_no_review_rule(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df["avg_rating"] == 0
        df.loc[mask, "grade"] = self.NEW_SHOP_GRADE
        return df

    def _apply_low_rating_demotion(self, df: pd.DataFrame) -> pd.DataFrame:
        mask_low = (df["avg_rating"] > 0) & (df["avg_rating"] < self.LOW_RATING_THRESHOLD)
        for g_from, g_to in self.DEMOTION_CHAIN:
            df.loc[mask_low & (df["grade"] == g_from), "grade"] = g_to
        return df
