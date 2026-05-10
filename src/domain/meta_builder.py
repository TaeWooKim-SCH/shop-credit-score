from __future__ import annotations

import pandas as pd

from .raw_dataset import RawDataset


class MetaBuilder:
    """매장/처치/통제/주소 메타 테이블 구성 (각 매장당 1행)."""

    SHOP_META_COLS = [
        "platform_shop_id", "shop_name", "category_name", "shop_address",
        "brand_name", "promo_flag", "persona_flag", "group_type",
    ]
    TREAT_META_COLS = [
        "platform_shop_id", "user_id", "service_term_agree_date", "treat_month",
    ]

    def shop_meta(self, raw: RawDataset) -> pd.DataFrame:
        keep = [c for c in self.SHOP_META_COLS if c in raw.shop.columns]
        return raw.shop[keep].drop_duplicates("platform_shop_id")

    def treat_meta(self, raw: RawDataset) -> pd.DataFrame:
        df = raw.treat.copy()
        df["service_term_agree_date"] = pd.to_datetime(
            df["service_term_agree_date"], errors="coerce"
        )
        df["treat_month"] = df["service_term_agree_date"].dt.to_period("M").astype(str)
        keep = [c for c in self.TREAT_META_COLS if c in df.columns]
        return df[keep].drop_duplicates("platform_shop_id")

    def control_meta(self, raw: RawDataset) -> pd.DataFrame:
        df = raw.control.copy()
        if "platform_shop_id" not in df.columns:
            return pd.DataFrame()

        for col in ["delivery_price", "pickup_price"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["is_active_menu"] = (
            (df["menu_status"].astype(str) == "ACTIVE").astype(int)
            if "menu_status" in df.columns else 0
        )

        agg_ctrl: dict[str, tuple[str, str]] = {
            "active_menu_ratio": ("is_active_menu", "mean"),
        }
        if "menu_id" in df.columns:
            agg_ctrl["menu_count"] = ("menu_id", "nunique")
        if "delivery_price" in df.columns:
            agg_ctrl["avg_delivery_price"] = ("delivery_price", "mean")
        if "pickup_price" in df.columns:
            agg_ctrl["avg_pickup_price"] = ("pickup_price", "mean")
        if "business_number" in df.columns:
            agg_ctrl["business_number"] = ("business_number", "first")

        return df.groupby("platform_shop_id", as_index=False).agg(**agg_ctrl)

    def address_meta(self, raw: RawDataset) -> pd.DataFrame:
        if "platform_shop_id" not in raw.address.columns:
            return pd.DataFrame()
        return raw.address.drop_duplicates("platform_shop_id")
