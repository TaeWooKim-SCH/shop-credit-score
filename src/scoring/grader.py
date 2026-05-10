from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    GRADE_BIN_METHOD,
    GRADE_BINS,
    GRADE_LABELS,
    GRADE_TARGET_RATIOS,
)


class GradeAssigner:
    """그로몽 스코어 → D/C/B/A/S 등급 + 평점 보정 규칙.

    bin 산출 방식:
      - method="fixed"    : ``bins`` 인자(또는 config.GRADE_BINS) 그대로 사용
      - method="quantile" : 데이터 분포 + 목표 비율(``target_ratios``)로 동적 산출
                            calibration 등으로 분포가 이동해도 등급 비율이 일정
    """

    NEW_SHOP_GRADE = "C"
    LOW_RATING_THRESHOLD = 3.5
    DEMOTION_CHAIN = [("S", "A"), ("A", "B"), ("B", "C"), ("C", "D")]
    SCORE_LOWER_BOUND = -1.0       # 최소 스코어보다 작은 값
    SCORE_UPPER_BOUND = 1000.0     # 최대 스코어보다 큰 값

    def __init__(
        self,
        bins: list[float] | None = None,
        labels: list[str] | None = None,
        method: str | None = None,
        target_ratios: dict[str, float] | None = None,
    ):
        self.method = method or GRADE_BIN_METHOD
        self.labels = labels or GRADE_LABELS
        self.target_ratios = target_ratios or GRADE_TARGET_RATIOS
        self.bins = bins if bins is not None else GRADE_BINS

    def assign(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        bins = self._resolve_bins(out["gromong_score"])
        out["grade"] = pd.cut(
            out["gromong_score"], bins=bins,
            labels=self.labels, include_lowest=True,
        ).astype(str)
        out = self._apply_no_review_rule(out)
        out = self._apply_low_rating_demotion(out)
        return out

    def _resolve_bins(self, score: pd.Series) -> list[float]:
        if self.method == "fixed":
            return list(self.bins)
        if self.method == "quantile":
            return self._compute_quantile_bins(score)
        raise ValueError(f"Unknown grade bin method: {self.method}")

    def _compute_quantile_bins(self, score: pd.Series) -> list[float]:
        # 누적 비율을 따라 분위수 컷 포인트 산출.
        # labels 순서가 D, C, B, A, S 이므로 누적 비율도 동일 순서로 계산.
        cum = 0.0
        cuts: list[float] = []
        for label in self.labels[:-1]:
            cum += self.target_ratios.get(label, 0.0)
            cuts.append(float(score.quantile(cum)))
        # 단조 증가 보장 (분포가 매우 치우친 경우 동일 값 발생 → 미세 보정)
        bins = [self.SCORE_LOWER_BOUND] + cuts + [self.SCORE_UPPER_BOUND]
        for i in range(1, len(bins)):
            if bins[i] <= bins[i - 1]:
                bins[i] = bins[i - 1] + 1e-6
        return bins

    def _apply_no_review_rule(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df["avg_rating"] == 0
        df.loc[mask, "grade"] = self.NEW_SHOP_GRADE
        return df

    def _apply_low_rating_demotion(self, df: pd.DataFrame) -> pd.DataFrame:
        mask_low = (df["avg_rating"] > 0) & (df["avg_rating"] < self.LOW_RATING_THRESHOLD)
        for g_from, g_to in self.DEMOTION_CHAIN:
            df.loc[mask_low & (df["grade"] == g_from), "grade"] = g_to
        return df
