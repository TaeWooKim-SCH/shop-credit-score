from __future__ import annotations

import pandas as pd

from ..config import Paths
from ..scoring import (
    GradeAssigner,
    GromongScoreCalculator,
    IndexCalculatorRegistry,
    LatestShopSelector,
    ROISimulator,
)
from ..viz import DemoCard, RadarChart, ScorePlotter


class ScoreBuilder:
    """Phase 6–7 오케스트레이션. 매장 선정 → 지수 → 스코어 → 등급 → ROI → 데모."""

    OUTPUT_FILE = "latest_shop_scores.csv"
    REPORT_COLS = [
        "platform_shop_id", "shop_name", "year_month", "grade",
        "gromong_score", "gromong_score_default",
        "idx_RRI", "idx_OPI", "idx_SRI", "idx_RSI", "idx_MRI",
        "growth_probability", "monthly_order_count", "avg_rating", "owner_reply_rate",
        "predicted_treatment_effect", "expected_roi_12m", "payback_months",
    ]

    def __init__(
        self,
        paths: Paths,
        selector: LatestShopSelector | None = None,
        index_registry: IndexCalculatorRegistry | None = None,
        score_calculator: GromongScoreCalculator | None = None,
        grader: GradeAssigner | None = None,
        plotter: ScorePlotter | None = None,
        radar: RadarChart | None = None,
        demo_card: DemoCard | None = None,
        roi_simulator: ROISimulator | None = None,
        fallback_effect: float | None = None,
    ):
        self.paths = paths
        self.selector = selector or LatestShopSelector()
        self.indices = index_registry or IndexCalculatorRegistry()
        self.score_calc = score_calculator or GromongScoreCalculator()
        self.grader = grader or GradeAssigner()
        self.plotter = plotter or ScorePlotter(paths)
        self.radar = radar or RadarChart(paths)
        self.demo_card = demo_card or DemoCard()
        self.roi_simulator = roi_simulator or ROISimulator()
        self.fallback_effect = fallback_effect

        self.latest: pd.DataFrame | None = None

    def run(
        self,
        master: pd.DataFrame,
        model_df: pd.DataFrame,
        shap_weights: dict[str, float],
        demo_id: str | None = None,
    ) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("PHASE 6 ▸ 그로몽 스코어 산출 (RRI / OPI / SRI / RSI)")
        print("=" * 60)
        latest = self.selector.select(master, model_df)
        latest = self.indices.apply(latest)
        latest = self.score_calc.compute(latest, shap_weights)
        latest = self.grader.assign(latest)

        # ROI 시뮬레이션
        if self.fallback_effect is not None:
            self.roi_simulator.params.fallback_effect = self.fallback_effect
        latest = self.roi_simulator.compute(latest)

        self.latest = latest

        self._print_summary()
        self._print_roi_summary()
        self.plotter.plot_index_distribution(latest)
        self.plotter.plot_grade_dist(latest)

        print("\n" + "=" * 60)
        print("PHASE 7 ▸ DEMO — 매장 스코어 조회")
        print("=" * 60)
        if demo_id:
            self._render_demo(demo_id)

        self._save()
        self._print_top()
        return latest

    def _render_demo(self, shop_id: str) -> None:
        row = self.latest[self.latest["platform_shop_id"] == shop_id]
        if row.empty:
            print(f"❌ 매장 '{shop_id}' 을(를) 찾을 수 없습니다.")
            return
        r = row.iloc[0]
        self.demo_card.render(r)
        path = self.radar.render(r, shop_id)
        print(f"  레이더 차트 저장: {path}")

    def _print_summary(self) -> None:
        print("\n[등급 분포]")
        print(self.latest["grade"].value_counts().sort_index())
        print("\n[그로몽 스코어 통계]")
        print(self.latest["gromong_score"].describe().round(2))

    def _print_roi_summary(self) -> None:
        if "expected_roi_12m" not in self.latest.columns:
            return
        roi = self.latest["expected_roi_12m"]
        pos = (roi > 0).sum()
        print(f"\n[ROI 시뮬레이션 — 12개월 기준]")
        print(f"  ROI > 0 매장 수: {pos:,} / {len(self.latest):,} "
              f"({pos/len(self.latest)*100:.1f}%)")
        print(f"  평균 ROI : {roi.mean():.2%}")
        print(f"  중앙 ROI : {roi.median():.2%}")
        print(f"  Top10% ROI : {roi.quantile(0.9):.2%}")
        payback = self.latest["payback_months"].dropna()
        if len(payback):
            print(f"  Payback 평균: {payback.mean():.1f}개월, "
                  f"중앙 {payback.median():.1f}개월 (NaN={len(self.latest)-len(payback)})")

    def _print_top(self, n: int = 20) -> None:
        cols = [c for c in self.REPORT_COLS if c in self.latest.columns]
        print(f"\n[그로몽 스코어 Top {n}]")
        print(
            self.latest[cols]
            .sort_values("gromong_score", ascending=False)
            .head(n)
            .to_string(index=False)
        )

    def _save(self) -> None:
        self.latest.to_csv(
            self.paths.output_dir / self.OUTPUT_FILE,
            index=False, encoding="utf-8-sig",
        )
