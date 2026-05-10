from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import GROWTH_DECLINE_THRESHOLD, GROWTH_HORIZON_MONTHS, GROWTH_THRESHOLD


class GrowthLabeler:
    """미래 N개월 주문 증가율 기반 growth_label 부여 (data leakage 방지).

    두 가지 라벨을 동시에 부여:
      - growth_label    (binary)  : 1=성장(+thresh) / 0=유지·하락
      - growth_label_3c (3-class) : 1=성장 / 0=유지 / -1=하락(-decline_thresh)
    """

    def __init__(
        self,
        horizon_months: int = GROWTH_HORIZON_MONTHS,
        threshold: float = GROWTH_THRESHOLD,
        decline_threshold: float = GROWTH_DECLINE_THRESHOLD,
    ):
        self.horizon = horizon_months
        self.threshold = threshold
        self.decline_threshold = decline_threshold

    def add(self, master: pd.DataFrame) -> pd.DataFrame:
        m = master.copy()
        future_col_name = f"future_order_{self.horizon}m"
        m[future_col_name] = (
            m.groupby("platform_shop_id")["monthly_order_count"].shift(-self.horizon)
        )
        future = m[future_col_name]
        cur = m["monthly_order_count"]
        growth_rate = (future - cur) / cur

        valid = (~future.isna()) & (cur > 0)

        m["growth_label"] = np.where(
            future.isna(), np.nan,
            np.where(valid & (growth_rate >= self.threshold), 1, 0),
        )

        m["growth_label_3c"] = np.where(
            future.isna(), np.nan,
            np.where(
                valid & (growth_rate >= self.threshold), 1,
                np.where(valid & (growth_rate <= self.decline_threshold), -1, 0),
            ),
        )
        return m

    def labeled_subset(self, master: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        labeled = master.dropna(subset=["growth_label"]).copy()
        labeled["growth_label"] = labeled["growth_label"].astype(int)
        labeled["growth_label_3c"] = labeled["growth_label_3c"].astype(int)

        if verbose:
            print(f"\n[growth_label 분포 — 미래 {self.horizon}개월 기준]")
            print(labeled["growth_label"].value_counts())
            print(f"양성 비율: {labeled['growth_label'].mean():.1%}")

            print(f"\n[growth_label_3c 분포 (성장/유지/하락)]")
            cnt = labeled["growth_label_3c"].value_counts().reindex([1, 0, -1])
            print(cnt.to_string())
            print(f"  성장 비율: {(labeled['growth_label_3c']==1).mean():.1%} | "
                  f"유지: {(labeled['growth_label_3c']==0).mean():.1%} | "
                  f"하락: {(labeled['growth_label_3c']==-1).mean():.1%}")
        return labeled
