from __future__ import annotations

import json

import pandas as pd
from xgboost import XGBClassifier

from ..config import Paths
from ..modeling import (
    FeatureRegistry,
    GrowthPredictor,
    ModelComparator,
    MulticlassGrowthPredictor,
    ProbabilityCalibrator,
    ShapAnalysis,
    ShapAnalyzer,
    ShopClusterer,
    ThresholdOptimizer,
)
from ..viz import ModelPlotter


class MLModeler:
    """Phase 4–5 오케스트레이션. 피처 정의 → 학습 → SHAP → 가중치 산출.

    선택적으로 알고리즘 비교 / 임계값 최적화 / 확률 보정도 수행해 평가지표를
    산출물로 저장한다.
    """

    MODEL_DF_FILE = "model_dataset_with_score.csv"
    SHAP_IMPORTANCE_FILE = "shap_feature_importance.csv"
    SHAP_WEIGHTS_FILE = "shap_weights.json"
    MODEL_COMPARISON_FILE = "model_comparison.csv"
    THRESHOLD_FILE = "threshold_analysis.json"
    CALIBRATION_FILE = "calibration_report.json"
    MULTICLASS_FILE = "multiclass_evaluation.json"
    CLUSTER_SUMMARY_FILE = "cluster_summary.csv"
    TRANSFER_COLS_FROM_MASTER = [
        "sales_vs_brand_avg", "sales_rank_in_brand", "review_per_order",
        "sales_log", "orders_log", "reviews_log", "order_cv",
    ]
    CALIBRATION_HOLDOUT_RATIO = 0.5  # X_test의 절반을 calibration용으로 사용

    def __init__(
        self,
        paths: Paths,
        registry: FeatureRegistry | None = None,
        predictor: GrowthPredictor | None = None,
        shap_analyzer: ShapAnalyzer | None = None,
        plotter: ModelPlotter | None = None,
        multiclass_predictor: MulticlassGrowthPredictor | None = None,
        clusterer: ShopClusterer | None = None,
        run_comparison: bool = True,
        run_calibration: bool = True,
        run_multiclass: bool = True,
        run_clustering: bool = True,
    ):
        self.paths = paths
        self.registry = registry or FeatureRegistry()
        self.predictor = predictor or GrowthPredictor()
        self.shap_analyzer = shap_analyzer or ShapAnalyzer()
        self.plotter = plotter or ModelPlotter(paths)
        self.multiclass_predictor = multiclass_predictor or MulticlassGrowthPredictor()
        self.clusterer = clusterer or ShopClusterer(n_clusters=5)
        self.run_comparison = run_comparison
        self.run_calibration = run_calibration
        self.run_multiclass = run_multiclass
        self.run_clustering = run_clustering

        self.model_df: pd.DataFrame | None = None
        self.shap_analysis: ShapAnalysis | None = None
        self.threshold_analysis = None
        self.calibration_report = None
        self.comparison_df: pd.DataFrame | None = None
        self.multiclass_eval = None
        self.cluster_result = None
        self._calibrator: ProbabilityCalibrator | None = None

    def run(
        self, master: pd.DataFrame, master_labeled: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, float], XGBClassifier]:
        print("\n" + "=" * 60)
        print("PHASE 5 ▸ XGBoost 학습 & 평가")
        print("=" * 60)

        self.registry.fit(master)
        master_encoded = self.registry.encode_categories(master)
        labeled = self._propagate_features(master_encoded, master_labeled)

        feature_cols = [c for c in self.registry.feature_cols if c in labeled.columns]
        print(f"\n총 피처 수: {len(self.registry.feature_cols)}")
        print("피처 목록:", self.registry.feature_cols)

        self.model_df = self._build_model_df(labeled, feature_cols)
        print(f"학습 데이터: {self.model_df[feature_cols].shape}  |  "
              f"양성 비율: {self.model_df['growth_label'].mean():.1%}")

        split = self.predictor.split(self.model_df, feature_cols)
        print(f"train: {split.X_train.shape}  |  test: {split.X_test.shape}")

        self.predictor.train(split.X_train, split.y_train)
        eval_result = self.predictor.evaluate(split.X_test, split.y_test)
        self._print_evaluation(eval_result)
        self.plotter.plot_roc(eval_result)

        if self.run_comparison:
            self._run_model_comparison(split)

        print("\n[SHAP 가중치 최적화]")
        self.shap_analysis = self.shap_analyzer.analyze(
            self.predictor.model, split.X_test, self.registry.groups,
        )
        self._print_weights(self.shap_analysis)
        self.plotter.plot_shap_importance(self.shap_analysis)
        self.plotter.plot_weight_comparison(self.shap_analysis)

        self._run_threshold_analysis(split.X_test, split.y_test)

        if self.run_calibration:
            self._run_calibration(split)

        if self.run_multiclass:
            self._run_multiclass(feature_cols)

        if self.run_clustering:
            self._run_clustering()

        # 최종 확률: calibrated 우선, 없으면 raw
        if self.calibration_report is not None and self._calibrator is not None:
            self.model_df["growth_probability"] = self._calibrator.predict_proba(
                self.model_df[feature_cols]
            )
            self.model_df["growth_probability_raw"] = self.predictor.predict_proba(
                self.model_df[feature_cols]
            )
        else:
            self.model_df["growth_probability"] = self.predictor.predict_proba(
                self.model_df[feature_cols]
            )

        self._save()
        return self.model_df, self.shap_analysis.group_weights_normalized, self.predictor.model

    def _propagate_features(
        self, master_encoded: pd.DataFrame, labeled: pd.DataFrame,
    ) -> pd.DataFrame:
        out = labeled.copy()
        for col in self.registry.cat_cols:
            out[col] = master_encoded[col].reindex(out.index)
        for c in self.TRANSFER_COLS_FROM_MASTER:
            if c in master_encoded.columns:
                out[c] = master_encoded[c].reindex(out.index)
        return out

    def _build_model_df(
        self, labeled: pd.DataFrame, feature_cols: list[str],
    ) -> pd.DataFrame:
        df = labeled.dropna(subset=["year_month_dt"]).copy()
        df = df[df["pre_treat_avg_sales"] > 0].copy()
        df[feature_cols] = df[feature_cols].fillna(0)
        return df

    def _print_evaluation(self, ev) -> None:
        print("\n[분류 리포트]")
        print(ev.report)
        flag = "과적합 의심 ⚠️ — 피처 점검 필요" if ev.overfit_warning else "정상 범위 ✅"
        print(f"ROC-AUC : {ev.auc:.4f}  ({flag})")
        print("Confusion Matrix:")
        print(ev.cm)

    def _print_weights(self, analysis: ShapAnalysis) -> None:
        print("  SHAP 기반 최적 가중치:")
        for k, v in analysis.group_weights_normalized.items():
            print(f"    {k}: {v:.3f}  ({v * 100:.1f}%)")

    def _run_model_comparison(self, split) -> None:
        print("\n" + "=" * 60)
        print("PHASE 5-b ▸ 알고리즘 비교 (5-fold TimeSeriesSplit)")
        print("=" * 60)
        scale_pos = (split.y_train == 0).sum() / max((split.y_train == 1).sum(), 1)
        cmp = ModelComparator(scale_pos_weight=float(scale_pos))
        # train + test를 합쳐 시계열 CV (시간 순서는 model_df에 따라 이미 정렬됨)
        X_full = pd.concat([split.X_train, split.X_test])
        y_full = pd.concat([split.y_train, split.y_test])
        results = cmp.compare(X_full, y_full)
        self.comparison_df = ModelComparator.to_dataframe(results)
        if results:
            best = results[0]
            print(f"  → 최고 AUC: {best.name} ({best.auc_mean:.4f})")
            print(f"     XGBoost 선택 이유: SHAP 안정성 + GPU 가속 + 베이스라인 일관성")

    def _run_threshold_analysis(self, X_test, y_test) -> None:
        print("\n[임계값 분석] PR-curve 기반 F1-optimal / Youden / Cost-based")
        proba = self.predictor.predict_proba(X_test)
        opt = ThresholdOptimizer()
        self.threshold_analysis = opt.analyze(y_test, proba)
        print(f"  F1-optimal threshold     : {self.threshold_analysis.f1_optimal:.4f}")
        print(f"  Youden's J threshold     : {self.threshold_analysis.youden_optimal:.4f}")
        print(f"  Cost-optimal (FP=5,FN=1) : {self.threshold_analysis.cost_optimal:.4f}")
        print(f"  F1 @ default(0.5) : {self.threshold_analysis.f1_at_default:.4f}")
        print(f"  F1 @ optimal      : {self.threshold_analysis.f1_at_f1_optimal:.4f}  "
              f"(Δ {self.threshold_analysis.f1_at_f1_optimal - self.threshold_analysis.f1_at_default:+.4f})")

    def _run_multiclass(self, feature_cols: list[str]) -> None:
        print("\n[3-class 분류] 성장(1) / 유지(0) / 하락(-1) — XGBoost multi:softprob")
        df = self.model_df.dropna(subset=["growth_label_3c"]).copy()
        if df["growth_label_3c"].nunique() < 3:
            print("  3-class 학습 불가 — 클래스 부족")
            return
        X_tr, y_tr, X_te, y_te = self.multiclass_predictor.split(df, feature_cols)
        self.multiclass_predictor.train(X_tr, y_tr)
        self.multiclass_eval = self.multiclass_predictor.evaluate(X_te, y_te)
        print(self.multiclass_eval.report)
        print(f"  Macro F1   : {self.multiclass_eval.macro_f1:.4f}")
        print(f"  Weighted F1: {self.multiclass_eval.weighted_f1:.4f}")
        print("  Confusion Matrix (rows=true, cols=pred, order=[-1,0,1]):")
        print(self.multiclass_eval.cm)

        # 매장별 마지막 시점에 대해 3-class 확률 부여
        proba = self.multiclass_predictor.predict_proba(self.model_df[feature_cols])
        for col in proba.columns:
            self.model_df[col] = proba[col]

    def _run_clustering(self) -> None:
        print("\n[매장 군집화] KMeans(k=5) + Isolation Forest 이상치 탐지")
        # 매장별 통계 (latest row 기준)
        latest_per_shop = (
            self.model_df.sort_values(["platform_shop_id", "year_month_dt"])
            .groupby("platform_shop_id").tail(1)
        )
        clustered, result = self.clusterer.fit_predict(latest_per_shop)
        self.cluster_result = result
        print(f"  Silhouette score: {result.silhouette:.4f}")
        print(f"  이상치 매장: {result.outlier_count}/{len(latest_per_shop)} "
              f"({result.outlier_count/len(latest_per_shop)*100:.1f}%)")
        print("\n  [클러스터별 평균]")
        print(result.cluster_summary.to_string())

        # cluster_id, is_outlier를 model_df의 매장 단위로 매핑
        cluster_map = clustered.set_index("platform_shop_id")[["cluster_id", "is_outlier"]]
        self.model_df["cluster_id"] = self.model_df["platform_shop_id"].map(
            cluster_map["cluster_id"]
        )
        self.model_df["is_outlier"] = self.model_df["platform_shop_id"].map(
            cluster_map["is_outlier"]
        )

    def _run_calibration(self, split) -> None:
        print("\n[확률 보정] Isotonic Regression — calibration holdout 분리")
        n_test = len(split.X_test)
        n_cal = int(n_test * self.CALIBRATION_HOLDOUT_RATIO)
        X_cal, y_cal = split.X_test.iloc[:n_cal], split.y_test.iloc[:n_cal]
        X_eval, y_eval = split.X_test.iloc[n_cal:], split.y_test.iloc[n_cal:]
        if len(X_eval) < 50:
            print("  calibration 표본 부족 — 건너뜀")
            return
        self._calibrator = ProbabilityCalibrator(method="isotonic").fit(
            self.predictor.model, split.X_train, split.y_train, X_cal, y_cal,
        )
        report = self._calibrator.report(self.predictor.model, X_eval, y_eval)
        self.calibration_report = report
        print(f"  Brier  : {report.brier_before:.4f} → {report.brier_after:.4f} "
              f"(Δ {report.brier_after - report.brier_before:+.4f})")
        print(f"  LogLoss: {report.log_loss_before:.4f} → {report.log_loss_after:.4f} "
              f"(Δ {report.log_loss_after - report.log_loss_before:+.4f})")

    def _save(self) -> None:
        self.model_df.to_csv(
            self.paths.output_dir / self.MODEL_DF_FILE,
            index=False, encoding="utf-8-sig",
        )
        self.shap_analysis.importance_df.to_csv(
            self.paths.output_dir / self.SHAP_IMPORTANCE_FILE,
            index=False, encoding="utf-8-sig",
        )
        with open(self.paths.output_dir / self.SHAP_WEIGHTS_FILE, "w",
                  encoding="utf-8") as f:
            json.dump(self.shap_analysis.group_weights_normalized,
                      f, ensure_ascii=False, indent=2)

        if self.comparison_df is not None:
            self.comparison_df.to_csv(
                self.paths.output_dir / self.MODEL_COMPARISON_FILE,
                index=False, encoding="utf-8-sig",
            )
        if self.threshold_analysis is not None:
            payload = {
                "f1_optimal": self.threshold_analysis.f1_optimal,
                "youden_optimal": self.threshold_analysis.youden_optimal,
                "cost_optimal": self.threshold_analysis.cost_optimal,
                "f1_at_default": self.threshold_analysis.f1_at_default,
                "f1_at_f1_optimal": self.threshold_analysis.f1_at_f1_optimal,
            }
            with open(self.paths.output_dir / self.THRESHOLD_FILE, "w",
                      encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        if self.calibration_report is not None:
            payload = {
                "method": self.calibration_report.method,
                "brier_before": self.calibration_report.brier_before,
                "brier_after": self.calibration_report.brier_after,
                "log_loss_before": self.calibration_report.log_loss_before,
                "log_loss_after": self.calibration_report.log_loss_after,
            }
            with open(self.paths.output_dir / self.CALIBRATION_FILE, "w",
                      encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        if self.multiclass_eval is not None:
            payload = {
                "classes": self.multiclass_eval.classes,
                "macro_f1": self.multiclass_eval.macro_f1,
                "weighted_f1": self.multiclass_eval.weighted_f1,
                "confusion_matrix": self.multiclass_eval.cm.tolist(),
                "report": self.multiclass_eval.report,
            }
            with open(self.paths.output_dir / self.MULTICLASS_FILE, "w",
                      encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        if self.cluster_result is not None:
            self.cluster_result.cluster_summary.to_csv(
                self.paths.output_dir / self.CLUSTER_SUMMARY_FILE,
                encoding="utf-8-sig",
            )
