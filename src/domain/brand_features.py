from __future__ import annotations

import numpy as np
import pandas as pd


class BrandFeatureBuilder:
    """브랜드 상대 성과 + 로그 변환 + 리뷰/주문 비율 파생."""

    def add(self, master: pd.DataFrame) -> pd.DataFrame:
        m = master.copy()
        m = self._add_brand_relative(m)
        m = self._add_ratios_and_logs(m)
        return m

    def _add_brand_relative(self, m: pd.DataFrame) -> pd.DataFrame:
        if "brand_name" in m.columns:
            bm_mean = m.groupby(["brand_name", "year_month"])["monthly_sales"].transform("mean")
            m["sales_vs_brand_avg"] = np.where(bm_mean > 0, m["monthly_sales"] / bm_mean, 1.0)
            m["sales_rank_in_brand"] = (
                m.groupby(["brand_name", "year_month"])["monthly_sales"].rank(pct=True)
            )
        else:
            m["sales_vs_brand_avg"] = 1.0
            m["sales_rank_in_brand"] = 0.5
        return m

    def _add_ratios_and_logs(self, m: pd.DataFrame) -> pd.DataFrame:
        m["review_per_order"] = np.where(
            m["monthly_order_count"] > 0,
            m["monthly_review_count"] / m["monthly_order_count"], 0,
        )
        m["sales_log"] = np.log1p(m["monthly_sales"])
        m["orders_log"] = np.log1p(m["monthly_order_count"])
        m["reviews_log"] = np.log1p(m["monthly_review_count"])
        return m
