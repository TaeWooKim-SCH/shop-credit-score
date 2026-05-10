from __future__ import annotations

import numpy as np
import pandas as pd


class TreatmentPanelBuilder:
    """처치 시점 + 패널 인덱스 + 처치/통제 라벨 + 사전 기준선 부여."""

    EXPERIMENT_KEYWORD = "실험"

    def add(self, master: pd.DataFrame) -> pd.DataFrame:
        m = master.copy()
        m["year_month_dt"] = pd.to_datetime(m["year_month"] + "-01", errors="coerce")
        m = m.sort_values(["platform_shop_id", "year_month_dt"]).reset_index(drop=True)

        if "treat_month" in m.columns:
            m["treat_month_dt"] = pd.to_datetime(
                m["treat_month"].replace("unknown", np.nan) + "-01", errors="coerce"
            )
            m["months_since_treatment"] = (
                (m["year_month_dt"].dt.year - m["treat_month_dt"].dt.year) * 12
                + (m["year_month_dt"].dt.month - m["treat_month_dt"].dt.month)
            )
            m["post_treatment"] = (m["months_since_treatment"] >= 0).astype(int)
        else:
            m["months_since_treatment"] = -999
            m["post_treatment"] = 0

        m["treated"] = (
            m["group_type"].astype(str).str.contains(self.EXPERIMENT_KEYWORD, na=False).astype(int)
            if "group_type" in m.columns else 0
        )
        m["treated_x_post"] = m["treated"] * m["post_treatment"]

        m = self._add_pretreat_baselines(m)
        m = self._add_order_cv(m)
        return m

    def _add_pretreat_baselines(self, m: pd.DataFrame) -> pd.DataFrame:
        pre_mask = m["post_treatment"] == 0
        pre_sales = m[pre_mask].groupby("platform_shop_id")["monthly_sales"].mean()
        pre_orders = m[pre_mask].groupby("platform_shop_id")["monthly_order_count"].mean()

        m["pre_treat_avg_sales"] = m["platform_shop_id"].map(pre_sales).fillna(
            m.groupby("platform_shop_id")["monthly_sales"].transform("mean")
        )
        m["pre_treat_avg_orders"] = m["platform_shop_id"].map(pre_orders).fillna(
            m.groupby("platform_shop_id")["monthly_order_count"].transform("mean")
        )
        return m

    def _add_order_cv(self, m: pd.DataFrame) -> pd.DataFrame:
        order_cv = (
            m.groupby("platform_shop_id")["monthly_order_count"].std()
            / m.groupby("platform_shop_id")["monthly_order_count"].mean().replace(0, np.nan)
        ).fillna(0)
        m["order_cv"] = m["platform_shop_id"].map(order_cv)
        return m
