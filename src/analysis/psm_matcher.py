from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..config import PSM_CALIPER_MULT


@dataclass
class PSMResult:
    propensity_df: pd.DataFrame | None
    matched_ids: set[str]
    matched_n: int
    caliper: float
    att_psm: float | None = None
    p_value_psm: float | None = None


class PSMMatcher:
    """1:1 NN propensity score matching + matched-sample TWFE DID."""

    PSM_FEATURES = [
        "avg_orders_pre", "avg_sales_pre", "avg_rating_pre",
        "avg_reply_rate_pre", "avg_neg_ratio_pre",
    ]
    DID_FORMULA = "monthly_order_count ~ treated_x_post + C(shop_fe) + C(month_fe)"
    MIN_SHOPS = 20

    def estimate_propensity(self, master: pd.DataFrame) -> pd.DataFrame | None:
        psm_pre = (
            master[master["post_treatment"] == 0]
            .groupby("platform_shop_id", as_index=False)
            .agg(
                treated=("treated", "first"),
                avg_orders_pre=("monthly_order_count", "mean"),
                avg_sales_pre=("monthly_sales", "mean"),
                avg_rating_pre=("avg_rating", "mean"),
                avg_reply_rate_pre=("owner_reply_rate", "mean"),
                avg_neg_ratio_pre=("negative_review_ratio", "mean"),
            )
            .dropna()
        )

        if psm_pre["treated"].nunique() != 2 or len(psm_pre) <= self.MIN_SHOPS:
            return None

        X = psm_pre[self.PSM_FEATURES].fillna(0)
        Xs = StandardScaler().fit_transform(X)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(Xs, psm_pre["treated"])
        psm_pre["ps"] = lr.predict_proba(Xs)[:, 1]
        return psm_pre

    def match(self, psm_pre: pd.DataFrame, verbose: bool = True) -> PSMResult:
        treat_idx = psm_pre[psm_pre["treated"] == 1].index
        ctrl_idx = psm_pre[psm_pre["treated"] == 0].index
        treat_ps = psm_pre.loc[treat_idx, "ps"].values.reshape(-1, 1)
        ctrl_ps = psm_pre.loc[ctrl_idx, "ps"].values.reshape(-1, 1)

        nn = NearestNeighbors(n_neighbors=1).fit(ctrl_ps)
        dists, idxs = nn.kneighbors(treat_ps)

        caliper = float(PSM_CALIPER_MULT * psm_pre["ps"].std())
        valid = dists.flatten() <= caliper
        matched_n = int(valid.sum())

        matched_ctrl_ids = psm_pre.loc[ctrl_idx[idxs.flatten()[valid]], "platform_shop_id"].tolist()
        matched_treat_ids = psm_pre.loc[treat_idx[valid], "platform_shop_id"].tolist()
        matched_ids = set(matched_ctrl_ids + matched_treat_ids)

        if verbose:
            print(f"  Caliper={caliper:.4f}  매칭 성공: {matched_n}쌍 / {len(treat_idx)}개 실험군")

        return PSMResult(propensity_df=psm_pre, matched_ids=matched_ids,
                         matched_n=matched_n, caliper=caliper)

    def estimate_did_on_matched(
        self, master: pd.DataFrame, result: PSMResult,
        baseline_att: float | None = None, verbose: bool = True,
    ) -> PSMResult:
        if not result.matched_ids:
            return result
        matched = master[master["platform_shop_id"].isin(result.matched_ids)].copy()
        matched["shop_fe"] = LabelEncoder().fit_transform(matched["platform_shop_id"].astype(str))
        matched["month_fe"] = LabelEncoder().fit_transform(matched["year_month"].astype(str))

        try:
            res = smf.ols(self.DID_FORMULA, data=matched).fit(
                cov_type="cluster",
                cov_kwds={"groups": matched["platform_shop_id"]},
            )
            result.att_psm = res.params.get("treated_x_post", None)
            result.p_value_psm = res.pvalues.get("treated_x_post", None)
            if verbose and result.att_psm is not None:
                print(f"  PSM-DID ATT : {result.att_psm:.3f}건/월   p={result.p_value_psm:.4f}")
                if baseline_att is not None:
                    consistent = abs(result.att_psm - baseline_att) < abs(baseline_att) * 0.3
                    print(f"  → {'Full-sample DID와 일관 ✅' if consistent else '추정치 차이 확인 필요 ⚠️'}")
        except Exception as e:
            if verbose:
                print(f"  PSM-DID 오류: {e}")
        return result
