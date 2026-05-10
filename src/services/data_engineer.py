from __future__ import annotations

import pandas as pd

from ..config import Paths
from ..domain import (
    BrandFeatureBuilder,
    GrowthLabeler,
    LightweightSentimentAnalyzer,
    MetaBuilder,
    OrderAggregator,
    PanelMerger,
    RawDatasetLoader,
    ReviewAggregator,
    TreatmentPanelBuilder,
)


class DataEngineer:
    """Phase 0–4 오케스트레이션. 도메인 객체를 조립해 master 패널을 만든다."""

    OUTPUT_FILE = "master_dataset.csv"
    SENTIMENT_KW_FILE = "sentiment_keywords.csv"

    def __init__(
        self,
        paths: Paths,
        loader: RawDatasetLoader | None = None,
        order_aggregator: OrderAggregator | None = None,
        review_aggregator: ReviewAggregator | None = None,
        meta_builder: MetaBuilder | None = None,
        panel_merger: PanelMerger | None = None,
        treatment_panel: TreatmentPanelBuilder | None = None,
        growth_labeler: GrowthLabeler | None = None,
        brand_features: BrandFeatureBuilder | None = None,
        sentiment_analyzer: LightweightSentimentAnalyzer | None = None,
    ):
        self.paths = paths
        self.loader = loader or RawDatasetLoader(paths)
        self.order_aggregator = order_aggregator or OrderAggregator()
        self.review_aggregator = review_aggregator or ReviewAggregator()
        self.meta_builder = meta_builder or MetaBuilder()
        self.panel_merger = panel_merger or PanelMerger()
        self.treatment_panel = treatment_panel or TreatmentPanelBuilder()
        self.growth_labeler = growth_labeler or GrowthLabeler()
        self.brand_features = brand_features or BrandFeatureBuilder()
        self.sentiment_analyzer = sentiment_analyzer or LightweightSentimentAnalyzer()

        self.master: pd.DataFrame | None = None
        self.master_labeled: pd.DataFrame | None = None

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw = self.loader.load()
        order_monthly = self.order_aggregator.aggregate_monthly(raw.order)
        review_monthly = self.review_aggregator.aggregate_monthly(raw.review)
        sentiment_monthly = self._build_sentiment(raw.review)

        master = self.panel_merger.merge(
            order_monthly=order_monthly,
            review_monthly=review_monthly,
            shop_meta=self.meta_builder.shop_meta(raw),
            treat_meta=self.meta_builder.treat_meta(raw),
            control_meta=self.meta_builder.control_meta(raw),
            address_meta=self.meta_builder.address_meta(raw),
        )
        if not sentiment_monthly.empty:
            master = master.merge(
                sentiment_monthly, on=["platform_shop_id", "year_month"], how="left",
            )
        print(f"\n[master] shape: {master.shape}")

        master = self.treatment_panel.add(master)
        master = self.growth_labeler.add(master)
        labeled = self.growth_labeler.labeled_subset(master)
        master = self.brand_features.add(master)

        self.master = master
        self.master_labeled = labeled
        return master, labeled

    def _build_sentiment(self, review_raw: pd.DataFrame) -> pd.DataFrame:
        # ReviewAggregator는 raw에서 review_date/year_month/rating을 만들기 위해 transform을 호출하지만
        # 자체 부수효과는 없으므로 동일 로직을 한 번 더 적용 (또는 review_aggregator.transform 결과 재활용)
        prepared = self.review_aggregator.transform(review_raw, verbose=False)
        scored = self.sentiment_analyzer.add_review_signals(prepared)

        # 상위 키워드 통계 저장 (변별력 분석)
        try:
            kw_top = self.sentiment_analyzer.top_keywords(scored, top_n=50)
            kw_top.to_csv(
                self.paths.output_dir / self.SENTIMENT_KW_FILE,
                index=False, encoding="utf-8-sig",
            )
        except Exception as e:
            print(f"  키워드 변별력 분석 스킵: {e}")

        return self.sentiment_analyzer.aggregate_monthly(scored)

    def save(self) -> None:
        self.master.to_csv(
            self.paths.output_dir / self.OUTPUT_FILE,
            index=False, encoding="utf-8-sig",
        )
