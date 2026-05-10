from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class IndexCalculator(ABC):
    """단일 지수 계산을 위한 Strategy 인터페이스. 산출은 0~1 clipped Series."""

    name: str = ""
    column: str = ""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series: ...

    @staticmethod
    def _get_or_default(df: pd.DataFrame, col: str, default: float) -> pd.Series:
        return df.get(col, pd.Series(default, index=df.index))


class RRICalculator(IndexCalculator):
    """RRI: 리뷰 응답 개선 여지 — 응답 안 할수록 ↑.

    표본 크기 보정을 위해 shrunk 컬럼 우선 사용 (없으면 raw로 폴백).
    리뷰 1건 5점 매장이 만점 받는 등 통계적 유의성 없는 케이스 방지.
    """

    name = "RRI (응답개선)"
    column = "idx_RRI"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        reply_rate = self._first_available(df, "any_reply_rate_shrunk", "any_reply_rate")
        neg_ratio = self._first_available(df, "negative_review_ratio_shrunk", "negative_review_ratio")
        rating = self._first_available(df, "avg_rating_shrunk", "avg_rating")
        return (
            (1 - reply_rate.clip(0, 1)) * 0.50
            + neg_ratio.clip(0, 1) * 0.30
            + (1 - rating.clip(0, 5) / 5.0) * 0.20
        ).clip(0, 1)

    @staticmethod
    def _first_available(df: pd.DataFrame, *cols: str) -> pd.Series:
        for c in cols:
            if c in df.columns:
                return df[c]
        return pd.Series(0, index=df.index)


class OPICalculator(IndexCalculator):
    """OPI: 주문 체력 + 성장 잠재력."""

    name = "OPI (주문성과)"
    column = "idx_OPI"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        sales_growth = self._get_or_default(df, "sales_growth_mom", 0).clip(-1, 3)
        return (
            df["growth_probability"].clip(0, 1) * 0.50
            + sales_growth.apply(lambda x: (x + 1) / 4) * 0.30
            + self._get_or_default(df, "sales_rank_in_brand", 0.5).clip(0, 1) * 0.20
        ).clip(0, 1)


class SRICalculator(IndexCalculator):
    """SRI: 감성 개선 여지 — shrunk 평점/부정비율 사용으로 표본 크기 보정."""

    name = "SRI (감성개선)"
    column = "idx_SRI"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        neg_ratio = RRICalculator._first_available(
            df, "negative_review_ratio_shrunk", "negative_review_ratio",
        )
        rating = RRICalculator._first_available(df, "avg_rating_shrunk", "avg_rating")
        return (
            neg_ratio.clip(0, 1) * 0.50
            + (1 - rating.clip(0, 5) / 5.0) * 0.30
            + self._get_or_default(df, "rating_std", 0).clip(0, 2) / 2.0 * 0.20
        ).clip(0, 1)


class RSICalculator(IndexCalculator):
    """RSI: 운영 안정성 — 메뉴 활성도 + 주문 안정성 + 메뉴 다양성."""

    name = "RSI (운영안정)"
    column = "idx_RSI"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        act = self._get_or_default(df, "active_menu_ratio", 0.5)
        cv = self._get_or_default(df, "order_cv", 0.3)
        mc = self._get_or_default(df, "menu_count", 1)
        max_mc = max(mc.max(), 1)
        return (
            act.clip(0, 1) * 0.40
            + (1 - cv.clip(0, 2) / 2.0) * 0.40
            + (mc / max_mc).clip(0, 1) * 0.20
        ).clip(0, 1)


class IndexCalculatorRegistry:
    """4개 지수 계산기를 보관하고 일괄 적용."""

    def __init__(self, calculators: list[IndexCalculator] | None = None):
        self.calculators = calculators or [
            RRICalculator(), OPICalculator(),
            SRICalculator(), RSICalculator(),
        ]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for calc in self.calculators:
            out[calc.column] = calc.compute(out)
        return out

    def name_for_column(self) -> dict[str, str]:
        return {c.column: c.name for c in self.calculators}
