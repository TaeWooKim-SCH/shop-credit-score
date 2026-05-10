from __future__ import annotations

import json

import pandas as pd

from ..analysis import (
    DIDEstimator,
    DIDResult,
    EventStudyAnalyzer,
    EventStudyResult,
    HomogeneityTester,
    PSMMatcher,
    PSMResult,
)
from ..config import Paths
from ..viz import CausalPlotter, EdaPlotter


class CausalAnalyst:
    """Phase 1–3 오케스트레이션. 분석 객체 + EDA/Causal 플로터 조립."""

    SUMMARY_FILE = "did_summary.json"

    def __init__(
        self,
        paths: Paths,
        homogeneity_tester: HomogeneityTester | None = None,
        did_estimator: DIDEstimator | None = None,
        event_study: EventStudyAnalyzer | None = None,
        psm_matcher: PSMMatcher | None = None,
        eda_plotter: EdaPlotter | None = None,
        causal_plotter: CausalPlotter | None = None,
    ):
        self.paths = paths
        self.tester = homogeneity_tester or HomogeneityTester()
        self.did = did_estimator or DIDEstimator()
        self.event = event_study or EventStudyAnalyzer()
        self.psm = psm_matcher or PSMMatcher()
        self.eda_plotter = eda_plotter or EdaPlotter(paths)
        self.causal_plotter = causal_plotter or CausalPlotter(paths)

        self.did_result: DIDResult | None = None
        self.event_result: EventStudyResult | None = None
        self.psm_result: PSMResult | None = None

    def run(
        self, master: pd.DataFrame, master_labeled: pd.DataFrame | None = None,
    ) -> dict:
        self._run_eda(master, master_labeled)
        self._run_did(master)
        self._run_psm(master)
        self._save_summary()
        return self._summary_dict()

    def _run_eda(self, master: pd.DataFrame,
                 master_labeled: pd.DataFrame | None) -> None:
        print("\n" + "=" * 60)
        print("PHASE 1 ▸ EDA")
        print("=" * 60)
        self.tester.test(master)
        self.eda_plotter.plot_monthly_trend(master)
        self.eda_plotter.plot_rating_dist(master)
        if master_labeled is not None:
            self._print_label_summary(master_labeled)
        self._print_group_summary(master)
        self.eda_plotter.plot_brand_sales_dist(master)
        print("\n[EDA 차트 저장 완료]")

    def _run_did(self, master: pd.DataFrame) -> None:
        print("\n" + "=" * 60)
        print("PHASE 2 ▸ TWFE DID 인과추론")
        print("=" * 60)
        did_df = self.did.prepare(master)
        self.did_result = self.did.estimate(did_df)

        print("\n[PHASE 2-b] Event Study Plot 생성")
        self.event_result = self.event.estimate(did_df)
        self.causal_plotter.plot_event_study(self.event_result)
        if self.event_result.coefs:
            print("  Event Study Plot 저장 완료")

    def _run_psm(self, master: pd.DataFrame) -> None:
        print("\n" + "=" * 60)
        print("PHASE 3 ▸ PSM (성향점수매칭)")
        print("=" * 60)
        psm_pre = self.psm.estimate_propensity(master)
        if psm_pre is None:
            print("  PSM: 충분한 데이터 없음")
            return
        self.causal_plotter.plot_propensity(psm_pre)
        result = self.psm.match(psm_pre)
        baseline_att = self.did_result.att if self.did_result else None
        self.psm_result = self.psm.estimate_did_on_matched(
            master, result, baseline_att=baseline_att,
        )

    def _print_label_summary(self, master_labeled: pd.DataFrame) -> None:
        summary = master_labeled.groupby("growth_label").agg(
            avg_sales=("monthly_sales", "mean"),
            avg_orders=("monthly_order_count", "mean"),
            avg_reviews=("monthly_review_count", "mean"),
            avg_rating=("avg_rating", "mean"),
            avg_reply_rate=("owner_reply_rate", "mean"),
            avg_neg_ratio=("negative_review_ratio", "mean"),
        ).round(3)
        print("\n[EDA 4] 성장 레이블별 평균 비교")
        print(summary.to_string())

    def _print_group_summary(self, master: pd.DataFrame) -> None:
        summary = master.groupby("treated").agg(
            avg_sales=("monthly_sales", "mean"),
            avg_orders=("monthly_order_count", "mean"),
            avg_reviews=("monthly_review_count", "mean"),
            avg_rating=("avg_rating", "mean"),
            avg_reply_rate=("owner_reply_rate", "mean"),
        ).round(3)
        print("\n[EDA 5] 실험군(1) vs 통제군(0) 비교")
        print(summary.to_string())

    def _summary_dict(self) -> dict:
        d = self.did_result
        e = self.event_result
        p = self.psm_result
        return {
            "att": d.att if d else None,
            "p": d.p_value if d else None,
            "ci_lo": d.ci_lo if d else None,
            "ci_hi": d.ci_hi if d else None,
            "r2": d.r2 if d else None,
            "parallel_trends_p": e.pre_trend_p if e else None,
            "att_psm": p.att_psm if p else None,
            "p_psm": p.p_value_psm if p else None,
            "matched_n": p.matched_n if p else 0,
        }

    def _save_summary(self) -> None:
        with open(self.paths.output_dir / self.SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(self._summary_dict(), f, ensure_ascii=False, indent=2)
