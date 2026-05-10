from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from ..config import EVENT_WINDOW


@dataclass
class EventStudyResult:
    coefs: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    pre_trend_p: float | None = None

    @property
    def times(self) -> list[int]:
        return sorted(self.coefs)


class EventStudyAnalyzer:
    """t = -K..+K 더미 회귀로 동적 처치 효과 추정 + Parallel Trends 검증."""

    FORMULA = "monthly_order_count ~ D_t + C(shop_fe) + C(month_fe)"
    BASELINE = -1
    INVALID_T = -999
    MIN_ROWS = 100
    ALPHA = 0.05

    def __init__(self, window: tuple[int, int] = EVENT_WINDOW):
        self.window = window

    def estimate(self, did_df: pd.DataFrame, verbose: bool = True) -> EventStudyResult:
        lo, hi = self.window
        edf = did_df[
            (did_df["months_since_treatment"] >= lo)
            & (did_df["months_since_treatment"] <= hi)
            & (did_df["months_since_treatment"] != self.INVALID_T)
        ].copy()

        result = EventStudyResult()
        if len(edf) <= self.MIN_ROWS:
            if verbose:
                print("  Event Study: 충분한 데이터 없음")
            return result

        for t in sorted(edf["months_since_treatment"].unique()):
            if t == self.BASELINE:
                continue
            tmp = edf.copy()
            tmp["D_t"] = (tmp["months_since_treatment"] == t).astype(int) * tmp["treated"]
            try:
                res = smf.ols(self.FORMULA, data=tmp).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": tmp["platform_shop_id"]},
                )
                if "D_t" in res.params:
                    result.coefs[int(t)] = (
                        float(res.params["D_t"]),
                        float(res.conf_int().loc["D_t", 0]),
                        float(res.conf_int().loc["D_t", 1]),
                    )
            except Exception:
                continue

        result.pre_trend_p = self._pre_trend_test(result, verbose=verbose)
        return result

    def _pre_trend_test(self, result: EventStudyResult, verbose: bool) -> float | None:
        pre_coefs = [result.coefs[t][0] for t in result.times if t < 0]
        if len(pre_coefs) < 2:
            return None
        t_stat, p = stats.ttest_1samp(pre_coefs, 0)
        if verbose:
            verdict = "충족 ✅" if p > self.ALPHA else "위배 가능 ⚠️ — PSM-DID 결과 우선 참고"
            print(f"  Pre-trend joint test: t={t_stat:.3f}, p={p:.4f}")
            print(f"  Parallel Trends 가정: {verdict}")
        return float(p)
