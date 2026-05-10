from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ..config import GRADE_COLORS, GRADE_ORDER, Paths


class ScorePlotter:
    """스코어 시각화: 4개 지수 분포 / 등급별 박스플롯."""

    INDEX_TITLES = [
        ("idx_RRI", "RRI: 리뷰 응답 개선 여지"),
        ("idx_OPI", "OPI: 주문 성과 지수"),
        ("idx_SRI", "SRI: 감성 개선 여지"),
        ("idx_RSI", "RSI: 운영 안정성 지수"),
    ]

    def __init__(self, paths: Paths):
        self.paths = paths

    def plot_index_distribution(self, latest: pd.DataFrame) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (col, title) in zip(axes.flatten(), self.INDEX_TITLES):
            ax.hist(latest[col].dropna(), bins=30,
                    color="steelblue", edgecolor="white", alpha=0.8)
            ax.set_title(title)
            ax.set_xlabel("지수 값 (0~1)")
            ax.set_ylabel("매장 수")
        fig.suptitle("4개 지수 분포", fontsize=14, y=1.01)
        self._save(fig, "index_distribution.png")

    def plot_grade_dist(self, latest: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        for g in GRADE_ORDER:
            d = latest[latest["grade"] == g]["gromong_score"]
            if len(d) > 0:
                ax.boxplot(d, positions=[GRADE_ORDER.index(g)], widths=0.5,
                           patch_artist=True,
                           boxprops=dict(facecolor=GRADE_COLORS[g], alpha=0.7))
        ax.set_xticks(range(len(GRADE_ORDER)))
        ax.set_xticklabels(GRADE_ORDER)
        ax.set_xlabel("등급")
        ax.set_ylabel("그로몽 스코어")
        ax.set_title("등급별 그로몽 스코어 분포")
        ax.grid(axis="y", alpha=0.3)
        self._save(fig, "grade_score_dist.png")

    def _save(self, fig, fname: str) -> None:
        fig.tight_layout()
        fig.savefig(self.paths.output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
