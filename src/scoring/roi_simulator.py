"""ROI 시뮬레이션 — 매장별 댓글몽 도입 시 기대 수익 추정.

핵심 식:
    월별 추가 매출 = CATE × avg_order_value
    누적 순이익(N개월) = 월별 추가 매출 × 마진율 × N - 구독료 × N
    ROI = 누적 순이익 / (구독료 × N)
    Payback months = 구독료 / (월별 추가 매출 × 마진율)

CATE 결측 매장은 전체 ATT를 폴백 효과로 사용.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (
    ROI_COMMENT_MONG_MONTHLY_FEE,
    ROI_HORIZON_MONTHS,
    ROI_MARGIN_RATE,
)


@dataclass
class ROIParams:
    margin_rate: float = ROI_MARGIN_RATE
    monthly_fee: float = ROI_COMMENT_MONG_MONTHLY_FEE
    horizon_months: int = ROI_HORIZON_MONTHS
    fallback_effect: float | None = None  # CATE 결측 시 폴백 효과 (건/월)


class ROISimulator:
    """매장별 CATE → ROI/Payback 산출.

    필요한 입력 컬럼:
      - predicted_treatment_effect (또는 fallback_effect)
      - avg_order_value
    """

    OUT_COLS = [
        "expected_orders_per_month",
        "expected_revenue_per_month",
        "expected_profit_per_month",
        "expected_cumulative_profit",
        "expected_roi_12m",
        "payback_months",
    ]

    def __init__(self, params: ROIParams | None = None):
        self.params = params or ROIParams()

    def compute(
        self,
        df: pd.DataFrame,
        params: ROIParams | None = None,
    ) -> pd.DataFrame:
        p = params or self.params
        out = df.copy()

        # CATE 결측 시 폴백 처리
        cate = out.get("predicted_treatment_effect")
        if cate is None:
            if p.fallback_effect is None:
                # 컬럼이 아예 없으면 0으로 설정 (ROI 계산 가능)
                cate = pd.Series(0.0, index=out.index)
            else:
                cate = pd.Series(p.fallback_effect, index=out.index)
        else:
            if p.fallback_effect is not None:
                cate = cate.fillna(p.fallback_effect)
            else:
                cate = cate.fillna(0.0)

        # 음수 효과는 보수적으로 0으로 clip (ROI 계산 목적)
        effect = cate.clip(lower=0)

        aov = out.get("avg_order_value", pd.Series(0.0, index=out.index)).fillna(0)

        out["expected_orders_per_month"] = effect
        out["expected_revenue_per_month"] = effect * aov
        out["expected_profit_per_month"] = (
            out["expected_revenue_per_month"] * p.margin_rate - p.monthly_fee
        )
        out["expected_cumulative_profit"] = (
            out["expected_profit_per_month"] * p.horizon_months
        )

        # ROI = 누적 순이익 / 총 구독료
        total_fee = p.monthly_fee * p.horizon_months
        out["expected_roi_12m"] = np.where(
            total_fee > 0,
            out["expected_cumulative_profit"] / total_fee,
            0.0,
        )

        # Payback months = 구독료 / 월 순이익 (월 순이익 ≤ 0이면 NaN)
        monthly_revenue_profit = out["expected_revenue_per_month"] * p.margin_rate
        out["payback_months"] = np.where(
            monthly_revenue_profit > p.monthly_fee,
            p.monthly_fee / (monthly_revenue_profit - p.monthly_fee + p.monthly_fee),
            # 위 식 정리: 도입 비용 회수에 걸리는 월 수
            np.nan,
        )
        # Payback 식 단순화: 구독료 / 월 매출 마진 (그 매장이 구독료를 회수하는 월수)
        # = monthly_fee / (cate * aov * margin)
        denom = effect * aov * p.margin_rate
        out["payback_months"] = np.where(
            denom > 0,
            p.monthly_fee / denom,
            np.nan,
        )
        # 너무 오래 걸리는 경우 캡 (예: 120개월 초과 → NaN으로 표시)
        out["payback_months"] = np.where(
            out["payback_months"] > 120,
            np.nan,
            out["payback_months"],
        )
        out["payback_months"] = out["payback_months"].round(2)
        out["expected_roi_12m"] = out["expected_roi_12m"].round(4)
        out["expected_revenue_per_month"] = out["expected_revenue_per_month"].round(0)
        out["expected_profit_per_month"] = out["expected_profit_per_month"].round(0)
        out["expected_cumulative_profit"] = out["expected_cumulative_profit"].round(0)
        out["expected_orders_per_month"] = out["expected_orders_per_month"].round(2)

        return out

    def cumulative_curve(
        self,
        cate: float,
        aov: float,
        months: int | None = None,
        params: ROIParams | None = None,
    ) -> pd.DataFrame:
        """단일 매장의 1~N개월 누적 순이익 곡선 (GUI 시각화용)."""
        p = params or self.params
        n = months or p.horizon_months
        monthly_profit = max(cate, 0) * aov * p.margin_rate - p.monthly_fee
        ms = np.arange(1, n + 1)
        return pd.DataFrame({
            "month": ms,
            "cumulative_profit": (monthly_profit * ms).round(0),
            "cumulative_fee": (p.monthly_fee * ms).round(0),
        })
