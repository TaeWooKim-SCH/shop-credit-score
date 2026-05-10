"""그로몽 파이프라인 진입점.

실행:
  python -m src        (프로젝트 루트에서)

레이어 구조:
  io/         - 외부 파일 입출력 (JsonLoader, ExcelLoader, ColumnCleaner)
  domain/     - 도메인 변환 (Aggregator, Merger, Labeler, ...)
  analysis/   - 인과추론 추정기 (DID, EventStudy, PSM, Homogeneity)
  modeling/   - 학습 컴포넌트 (FeatureRegistry, GrowthPredictor, ShapAnalyzer)
  scoring/    - 지수/점수/등급 (Strategy 패턴 IndexCalculator)
  viz/        - 시각화 + 콘솔 카드
  services/   - 4개 얇은 오케스트레이터 (서브에이전트 1:1 매핑)
"""
from .config import Paths
from .services import CausalAnalyst, DataEngineer, MLModeler, ScoreBuilder
from .utils import setup_korean_font

DEMO_SHOP_ID = "ba_10015935"

OUTPUT_FILES = [
    "master_dataset.csv", "model_dataset_with_score.csv",
    "latest_shop_scores.csv", "shap_feature_importance.csv",
    "shap_weights.json", "did_summary.json",
    "eda_monthly_trend.png", "eda_rating_dist.png", "eda_brand_sales_dist.png",
    "psm_propensity_dist.png", "event_study_plot.png",
    "roc_curve.png", "shap_importance.png", "weight_comparison.png",
    "index_distribution.png", "grade_score_dist.png",
    f"radar_{DEMO_SHOP_ID}.png",
]


def main() -> None:
    setup_korean_font()
    paths = Paths()

    de = DataEngineer(paths)
    master, master_labeled = de.run()
    de.save()

    ca = CausalAnalyst(paths)
    ca.run(master, master_labeled)

    ml = MLModeler(paths)
    model_df, shap_weights, _ = ml.run(master, master_labeled)

    sb = ScoreBuilder(paths)
    sb.run(master, model_df, shap_weights, demo_id=DEMO_SHOP_ID)

    print("\n✅ 전체 파이프라인 완료")
    print("저장 파일 목록:")
    for fname in OUTPUT_FILES:
        print(f"  - output/{fname}")


if __name__ == "__main__":
    main()
