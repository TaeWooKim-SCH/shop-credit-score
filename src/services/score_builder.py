from __future__ import annotations

import pandas as pd

from ..config import Paths
from ..scoring import (
    GradeAssigner,
    GromongScoreCalculator,
    IndexCalculatorRegistry,
    LatestShopSelector,
)
from ..viz import DemoCard, RadarChart, ScorePlotter


class ScoreBuilder:
    """Phase 6–7 오케스트레이션. 매장 선정 → 지수 → 스코어 → 등급 → 데모."""

    OUTPUT_FILE = "latest_shop_scores.csv"
    REPORT_COLS = [
        "platform_shop_id", "shop_name", "year_month", "grade",
        "gromong_score", "gromong_score_default",
        "idx_RRI", "idx_OPI", "idx_SRI", "idx_RSI",
        "growth_probability", "monthly_order_count", "avg_rating", "owner_reply_rate",
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
    ):
        self.paths = paths
        self.selector = selector or LatestShopSelector()
        self.indices = index_registry or IndexCalculatorRegistry()
        self.score_calc = score_calculator or GromongScoreCalculator()
        self.grader = grader or GradeAssigner()
        self.plotter = plotter or ScorePlotter(paths)
        self.radar = radar or RadarChart(paths)
        self.demo_card = demo_card or DemoCard()

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
        self.latest = latest

        self._print_summary()
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
