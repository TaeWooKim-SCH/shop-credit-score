from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Paths


class EdaPlotter:
    """EDA 시각화: 월별 추세 / 별점 분포 / 브랜드별 매출 분포."""

    GROUP_COLORS = {0: "steelblue", 1: "tomato"}
    GROUP_LABELS = {0: "통제군", 1: "실험군"}

    def __init__(self, paths: Paths):
        self.paths = paths

    def plot_monthly_trend(self, master: pd.DataFrame) -> None:
        trend = (
            master.groupby(["year_month_dt", "treated"])["monthly_order_count"]
            .mean().unstack(fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        for col in trend.columns:
            ax.plot(trend.index, trend[col],
                    label=self.GROUP_LABELS.get(col, str(col)),
                    color=self.GROUP_COLORS.get(col, "gray"), linewidth=2)
        ax.set_title("실험군 vs 통제군 — 월별 평균 주문 건수 추세")
        ax.set_xlabel("월")
        ax.set_ylabel("평균 주문 건수")
        ax.legend()
        ax.grid(alpha=0.3)
        self._save(fig, "eda_monthly_trend.png")

    def plot_rating_dist(self, master: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, (grp, lbl, clr) in enumerate(
            [(1, "실험군", "tomato"), (0, "통제군", "steelblue")]
        ):
            d = master[master["treated"] == grp]["avg_rating"].replace(0, np.nan).dropna()
            axes[i].hist(d, bins=20, color=clr, edgecolor="white", alpha=0.8)
            axes[i].set_title(f"{lbl} 별점 분포 (평균: {d.mean():.2f})")
            axes[i].set_xlabel("평균 별점")
            axes[i].set_ylabel("빈도")
        self._save(fig, "eda_rating_dist.png")

    def plot_brand_sales_dist(self, master: pd.DataFrame) -> None:
        if "brand_name" not in master.columns:
            return
        brand_perf = master.groupby("platform_shop_id").agg(
            avg_sales=("monthly_sales", "mean"),
            brand=("brand_name", "first"),
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        unique_brands = brand_perf["brand"].unique()
        cmap = plt.cm.get_cmap("tab10", len(unique_brands))
        for i, b in enumerate(unique_brands):
            d = brand_perf[brand_perf["brand"] == b]["avg_sales"]
            ax.hist(d, bins=30, alpha=0.6, label=str(b), color=cmap(i))
        ax.set_title("브랜드별 매장 평균 매출 분포")
        ax.set_xlabel("평균 매출")
        ax.legend()
        self._save(fig, "eda_brand_sales_dist.png")

    def _save(self, fig, fname: str) -> None:
        fig.tight_layout()
        fig.savefig(self.paths.output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
