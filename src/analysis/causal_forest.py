"""HTE (Heterogeneous Treatment Effect) 추정 — 매장별 개별 ATT.

전체 ATT(+113건/월)를 매장 특성에 따라 분해해 매장마다 다른 효과(CATE) 산출.
"평균 수렴" 우려 해소 + ROI 시뮬레이션의 입력값 제공.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class HTEResult:
    cate_per_shop: pd.DataFrame  # platform_shop_id, cate, ate_check
    mean_cate: float
    std_cate: float
    feature_cols: list[str]


class CausalForestEstimator:
    """Causal Forest DML로 매장별 CATE(조건부 평균 처치효과) 추정.

    pre-treatment 매장 특성 X 에 따라 처치효과가 어떻게 다른지 학습.
    DID ATT 와 평균은 일치하면서, 매장별 분산을 보존.

    출력 CATE는 ``전처리 효과 = 처치 후 평균 - 처치 전 평균`` 단위(건/월).
    """

    DEFAULT_FEATURES = [
        "avg_orders_pre",
        "avg_sales_pre",
        "avg_rating_pre",
        "avg_reply_rate_pre",
        "avg_neg_ratio_pre",
    ]
    N_ESTIMATORS = 200
    MAX_DEPTH = 5

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: CausalForestDML | None = None

    def _build_panel(self, master: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """매장당 1행 — pre-treatment 평균 X, treated T, post 평균 Y."""
        pre = master[master["post_treatment"] == 0]
        post = master[master["post_treatment"] == 1]

        # pre-treatment 매장 특성
        pre_agg = (
            pre.groupby("platform_shop_id", as_index=False)
            .agg(
                treated=("treated", "first"),
                avg_orders_pre=("monthly_order_count", "mean"),
                avg_sales_pre=("monthly_sales", "mean"),
                avg_rating_pre=("avg_rating", "mean"),
                avg_reply_rate_pre=("any_reply_rate", "mean")
                    if "any_reply_rate" in pre.columns
                    else ("owner_reply_rate", "mean"),
                avg_neg_ratio_pre=("negative_review_ratio", "mean"),
            )
        )

        # post-treatment outcome (평균 주문수)
        post_agg = (
            post.groupby("platform_shop_id", as_index=False)
            .agg(avg_orders_post=("monthly_order_count", "mean"))
        )

        panel = pre_agg.merge(post_agg, on="platform_shop_id", how="left")
        panel["avg_orders_post"] = panel["avg_orders_post"].fillna(
            panel["avg_orders_pre"]
        )
        # outcome = 변화량 (단위: 건/월)
        panel["delta_orders"] = panel["avg_orders_post"] - panel["avg_orders_pre"]
        panel = panel.dropna(subset=self.DEFAULT_FEATURES + ["treated"])

        return panel, [c for c in self.DEFAULT_FEATURES if c in panel.columns]

    def fit_predict(
        self, master: pd.DataFrame, verbose: bool = True
    ) -> HTEResult:
        panel, feature_cols = self._build_panel(master)
        if panel["treated"].nunique() < 2 or len(panel) < 50:
            if verbose:
                print("  [HTE] 데이터 부족 — Causal Forest 학습 불가")
            return HTEResult(pd.DataFrame(), 0.0, 0.0, feature_cols)

        X = panel[feature_cols].values
        T = panel["treated"].values.astype(int)
        Y = panel["delta_orders"].values

        self.model = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3),
            model_t=GradientBoostingRegressor(n_estimators=100, max_depth=3),
            discrete_treatment=True,
            n_estimators=self.N_ESTIMATORS,
            max_depth=self.MAX_DEPTH,
            random_state=self.random_state,
        )
        self.model.fit(Y, T, X=X)

        cate = self.model.effect(X)  # shape (n,) 각 매장별 처치효과
        ate_check = float(np.mean(cate))

        result_df = pd.DataFrame({
            "platform_shop_id": panel["platform_shop_id"].values,
            "predicted_treatment_effect": cate,
            "treated_actual": T,
        })

        result = HTEResult(
            cate_per_shop=result_df,
            mean_cate=ate_check,
            std_cate=float(np.std(cate)),
            feature_cols=feature_cols,
        )
        if verbose:
            print(f"  [HTE Causal Forest] 학습 {len(panel)}개 매장")
            print(f"    평균 CATE: {result.mean_cate:.3f} 건/월 (DID ATT와 비교)")
            print(f"    CATE std : {result.std_cate:.3f}")
            print(f"    CATE 분포: min={cate.min():.2f}, p25={np.percentile(cate,25):.2f}, "
                  f"p50={np.percentile(cate,50):.2f}, p75={np.percentile(cate,75):.2f}, "
                  f"max={cate.max():.2f}")
        return result

    def predict_for_shops(self, panel: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """학습된 모델로 매장 특성 X에 대해 CATE 예측."""
        if self.model is None:
            raise RuntimeError("fit_predict() 먼저 호출")
        X = panel[feature_cols].values
        return self.model.effect(X)
