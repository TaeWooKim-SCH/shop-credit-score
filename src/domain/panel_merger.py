from __future__ import annotations

import numpy as np
import pandas as pd


class PanelMerger:
    """월별 패널 + 메타 테이블을 master로 병합 + 결측 처리."""

    def merge(
        self,
        order_monthly: pd.DataFrame,
        review_monthly: pd.DataFrame,
        shop_meta: pd.DataFrame,
        treat_meta: pd.DataFrame,
        control_meta: pd.DataFrame,
        address_meta: pd.DataFrame,
    ) -> pd.DataFrame:
        m = pd.merge(
            order_monthly, review_monthly,
            on=["platform_shop_id", "year_month"], how="outer",
        )
        m = pd.merge(m, shop_meta, on="platform_shop_id", how="left")
        m = pd.merge(m, treat_meta, on="platform_shop_id", how="left")
        if not control_meta.empty and "platform_shop_id" in control_meta.columns:
            m = pd.merge(m, control_meta, on="platform_shop_id",
                         how="left", suffixes=("", "_ctrl"))
        if not address_meta.empty and "platform_shop_id" in address_meta.columns:
            m = pd.merge(m, address_meta, on="platform_shop_id",
                         how="left", suffixes=("", "_addr"))
        return self._fill_missing(m)

    def _fill_missing(self, m: pd.DataFrame) -> pd.DataFrame:
        for col in m.select_dtypes(include=[np.number]).columns:
            m[col] = m[col].fillna(0)
        for col in m.select_dtypes(include=["object"]).columns:
            m[col] = m[col].fillna("unknown")
        return m
