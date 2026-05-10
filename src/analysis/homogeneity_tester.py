from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import stats


@dataclass
class HomogeneityResult:
    variable: str
    label: str
    p_value: float
    homogeneous: bool


class HomogeneityTester:
    """처치 전 표본에서 실험군/통제군 간 분포 동질성 검정 (Mann-Whitney U)."""

    DEFAULT_VARS = [
        ("monthly_order_count", "주문건수"),
        ("avg_rating", "평균별점"),
        ("monthly_sales", "월매출"),
        ("owner_reply_rate", "응답률"),
    ]
    ALPHA = 0.05

    def __init__(self, variables: list[tuple[str, str]] | None = None):
        self.variables = variables or self.DEFAULT_VARS

    def test(self, master: pd.DataFrame, verbose: bool = True) -> list[HomogeneityResult]:
        pre = master[master["post_treatment"] == 0]
        if verbose:
            print("\n[EDA 1] 처치 전 동질성 검정 (Mann-Whitney U)")

        results: list[HomogeneityResult] = []
        for var, label in self.variables:
            exp_ = pre[pre["treated"] == 1][var].dropna()
            ctrl_ = pre[pre["treated"] == 0][var].dropna()
            if len(exp_) == 0 or len(ctrl_) == 0:
                continue
            _, p = stats.mannwhitneyu(exp_, ctrl_, alternative="two-sided")
            r = HomogeneityResult(var, label, float(p), p > self.ALPHA)
            results.append(r)
            if verbose:
                flag = "동질 ✅" if r.homogeneous else "차이 있음 ⚠️ (PSM 권장)"
                print(f"  {label:10s} | p={p:.4f}  {flag}")
        return results
