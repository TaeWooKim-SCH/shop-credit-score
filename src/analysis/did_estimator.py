from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder


@dataclass
class DIDResult:
    att: float | None
    p_value: float | None
    ci_lo: float | None
    ci_hi: float | None
    r2: float | None


class DIDEstimator:
    """TWFE DID. Cluster-robust SE (cluster = 매장)."""

    FORMULA = "monthly_order_count ~ treated_x_post + C(shop_fe) + C(month_fe)"
    ALPHA = 0.05

    def prepare(self, master: pd.DataFrame) -> pd.DataFrame:
        df = master[master["pre_treat_avg_orders"] > 0].copy()
        df["shop_fe"] = LabelEncoder().fit_transform(df["platform_shop_id"].astype(str))
        df["month_fe"] = LabelEncoder().fit_transform(df["year_month"].astype(str))
        return df

    def estimate(self, did_df: pd.DataFrame, verbose: bool = True) -> DIDResult:
        try:
            res = smf.ols(self.FORMULA, data=did_df).fit(
                cov_type="cluster",
                cov_kwds={"groups": did_df["platform_shop_id"]},
            )
            att = res.params.get("treated_x_post", None)
            pval = res.pvalues.get("treated_x_post", None)
            ci = res.conf_int()
            ci_lo = ci.loc["treated_x_post", 0] if "treated_x_post" in ci.index else None
            ci_hi = ci.loc["treated_x_post", 1] if "treated_x_post" in ci.index else None
            result = DIDResult(att, pval, ci_lo, ci_hi, res.rsquared)
            if verbose:
                self._print(result)
            return result
        except Exception as e:
            print(f"  DID 추정 오류: {e}")
            return DIDResult(None, None, None, None, None)

    def _print(self, r: DIDResult) -> None:
        if r.att is not None:
            print(f"\n  DID ATT : {r.att:.3f}건/월")
        if r.ci_lo is not None:
            print(f"  95% CI  : [{r.ci_lo:.3f}, {r.ci_hi:.3f}]")
        if r.p_value is not None:
            sig = "유의한 처치 효과 ✅" if r.p_value < self.ALPHA else "유의하지 않음 ⚠️"
            print(f"  p-value : {r.p_value:.4f}  →  {sig}")
        if r.r2 is not None:
            print(f"  R²      : {r.r2:.4f}")
