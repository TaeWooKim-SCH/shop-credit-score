from __future__ import annotations

import numpy as np
import pandas as pd


class OrderAggregator:
    """주문 raw → 매장×월 패널 + 월간 성장률."""

    PEAK_HOURS = list(range(11, 14)) + list(range(17, 21))

    def transform(self, order_df: pd.DataFrame) -> pd.DataFrame:
        df = order_df.copy()
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

        for col in ["price", "quantity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["sales"] = (
            df["price"] * df["quantity"] if "quantity" in df.columns
            else df.get("price", np.nan)
        )
        df["order_hour"] = df["order_date"].dt.hour
        df["is_weekend"] = df["order_date"].dt.dayofweek.isin([5, 6]).astype(int)
        df["is_peak"] = df["order_hour"].isin(self.PEAK_HOURS).astype(int)
        return df

    def aggregate_monthly(self, order_df: pd.DataFrame) -> pd.DataFrame:
        df = self.transform(order_df)

        agg_dict = {
            "monthly_order_count": ("platform_shop_id", "size"),
            "monthly_sales": ("sales", "sum"),
            "avg_order_value": ("sales", "mean"),
            "weekend_ratio": ("is_weekend", "mean"),
            "peak_ratio": ("is_peak", "mean"),
        }
        if "quantity" in df.columns:
            agg_dict["monthly_quantity_sum"] = ("quantity", "sum")

        monthly = (
            df.groupby(["platform_shop_id", "year_month"], as_index=False)
            .agg(**agg_dict)
            .sort_values(["platform_shop_id", "year_month"])
        )
        monthly["sales_growth_mom"] = (
            monthly.groupby("platform_shop_id")["monthly_sales"].pct_change()
        )
        monthly["order_growth_mom"] = (
            monthly.groupby("platform_shop_id")["monthly_order_count"].pct_change()
        )
        return monthly
