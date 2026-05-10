from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ..analysis import EventStudyResult
from ..config import Paths


class CausalPlotter:
    """인과추론 시각화: 성향점수 분포 / Event Study Plot."""

    def __init__(self, paths: Paths):
        self.paths = paths

    def plot_propensity(self, propensity_df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        for grp, lbl, clr in [(1, "실험군", "tomato"), (0, "통제군", "steelblue")]:
            d = propensity_df[propensity_df["treated"] == grp]["ps"]
            ax.hist(d, bins=30, alpha=0.6, label=f"{lbl} (n={len(d)})", color=clr)
        ax.set_xlabel("성향 점수")
        ax.set_title("PSM — 성향점수 분포 (매칭 전)")
        ax.legend()
        self._save(fig, "psm_propensity_dist.png")

    def plot_event_study(self, result: EventStudyResult) -> None:
        if not result.coefs:
            return
        times = result.times
        coefs = [result.coefs[t][0] for t in times]
        ci_lo = [result.coefs[t][1] for t in times]
        ci_hi = [result.coefs[t][2] for t in times]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, coefs, "o-", color="tomato", linewidth=2, label="처치 효과 추정치")
        ax.fill_between(times, ci_lo, ci_hi, alpha=0.2, color="tomato", label="95% CI")
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.axvline(0, color="navy", linewidth=1.5, linestyle=":", label="처치 시점")
        ax.set_xlabel("처치 기준 상대 시간 (월)")
        ax.set_ylabel("처치 효과 (주문 건수)")
        ax.set_title(
            "Event Study Plot — Parallel Trends 검증\n처치 전(t<0) 계수 ≈ 0 이어야 DID 유효"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        self._save(fig, "event_study_plot.png")

    def _save(self, fig, fname: str) -> None:
        fig.tight_layout()
        fig.savefig(self.paths.output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
