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

    응답률은 3유형(사장님 직접 + AI + 마케팅) 합산인 ``any_reply_rate`` 사용.
    ``owner_reply_rate``(사장님 직접만)는 매우 희소(~0.1%)라 분리 후엔 단독 사용 부적합.
    """

    name = "RRI (응답개선)"
    column = "idx_RRI"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        reply_rate = self._get_or_default(df, "any_reply_rate", df.get("owner_reply_rate", 0))
        return (
            (1 - reply_rate.clip(0, 1)) * 0.50
            + self._get_or_default(df, "negative_review_ratio", 0.2).clip(0, 1) * 0.30
            + (1 - df["avg_rating"].clip(0, 5) / 5.0) * 0.20
        ).clip(0, 1)


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
    """SRI: 감성 개선 여지."""

    name = "SRI (감성개선)"
    column = "idx_SRI"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return (
            self._get_or_default(df, "negative_review_ratio", 0.2).clip(0, 1) * 0.50
            + (1 - df["avg_rating"].clip(0, 5) / 5.0) * 0.30
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
