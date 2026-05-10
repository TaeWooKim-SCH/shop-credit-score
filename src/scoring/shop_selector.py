from __future__ import annotations

import pandas as pd

from ..config import MIN_MONTHS_FOR_SCORING


class LatestShopSelector:
    """6개월 이상 데이터 보유 매장 필터 + 매장당 최신 1행 추출 + growth_probability 부여."""

    def __init__(self, min_months: int = MIN_MONTHS_FOR_SCORING):
        self.min_months = min_months

    def select(
        self,
        master: pd.DataFrame,
        model_df: pd.DataFrame,
        verbose: bool = True,
    ) -> pd.DataFrame:
        valid_shops = self._valid_shops(master)
        if verbose:
            print(f"  스코어링 대상: {len(valid_shops)}개 매장 ({self.min_months}개월 이상 데이터 보유)")

        latest = (
            master[master["platform_shop_id"].isin(valid_shops)]
            .sort_values(["platform_shop_id", "year_month_dt"])
            .groupby("platform_shop_id").tail(1).copy()
        )

        latest_prob = (
            model_df.sort_values(["platform_shop_id", "year_month_dt"])
            .groupby("platform_shop_id")["growth_probability"].last()
        )
        latest["growth_probability"] = latest["platform_shop_id"].map(latest_prob).fillna(0.0)
        return latest

    def _valid_shops(self, master: pd.DataFrame):
        cnt = master.groupby("platform_shop_id")["year_month"].nunique()
        return cnt[cnt >= self.min_months].index
