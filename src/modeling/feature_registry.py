from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FeatureRegistry:
    """4개 지수에 매핑되는 피처 그룹 정의 + 카테고리 인코딩 책임."""

    FEAT_OPI = [
        "monthly_order_count", "monthly_sales", "avg_order_value",
        "sales_growth_mom", "order_growth_mom",
        "sales_log", "orders_log", "weekend_ratio", "peak_ratio",
    ]
    FEAT_SRI = [
        "monthly_review_count", "avg_rating", "rating_std",
        "reviews_log", "review_per_order", "negative_review_ratio",
        "sentiment_neg_kw", "sentiment_pos_kw",
        "rating_text_inconsist_rate", "text_length_mean",
    ]
    FEAT_RRI = [
        "owner_reply_rate", "ai_reply_rate", "marketing_reply_rate", "any_reply_rate",
        "months_since_treatment", "post_treatment", "treated", "treated_x_post",
    ]
    FEAT_RSI = [
        "menu_count", "active_menu_ratio", "avg_delivery_price", "avg_pickup_price",
        "order_cv", "sales_vs_brand_avg", "sales_rank_in_brand",
    ]
    CATEGORY_COLS = ["category_name", "brand_name", "group_type"]

    GROUP_KEY_MAP = {
        "RRI (응답개선)": FEAT_RRI,
        "OPI (주문성과)": FEAT_OPI,
        "SRI (감성개선)": FEAT_SRI,
        "RSI (운영안정)": FEAT_RSI,
    }

    def __init__(self):
        self.groups: dict[str, list[str]] = {}
        self.feature_cols: list[str] = []
        self.cat_cols: list[str] = []

    def fit(self, master: pd.DataFrame) -> "FeatureRegistry":
        present = lambda cols: [c for c in cols if c in master.columns]
        self.groups = {name: present(cols) for name, cols in self.GROUP_KEY_MAP.items()}
        ordered: list[str] = []
        for name in ["OPI (주문성과)", "SRI (감성개선)",
                     "RRI (응답개선)", "RSI (운영안정)"]:
            ordered.extend(self.groups[name])
        self.feature_cols = list(dict.fromkeys(ordered))
        self.cat_cols = [c for c in self.CATEGORY_COLS if c in master.columns]
        return self

    def encode_categories(self, master: pd.DataFrame) -> pd.DataFrame:
        out = master.copy()
        for col in self.cat_cols:
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].astype(str))
            if col not in self.feature_cols:
                self.feature_cols.append(col)
        self.feature_cols = list(dict.fromkeys(self.feature_cols))
        return out
