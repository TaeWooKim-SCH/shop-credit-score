from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Paths


class RadarChart:
    """매장별 4개 지수 레이더 차트."""

    CATEGORIES = ["RRI\n응답개선", "OPI\n주문성과", "SRI\n감성개선", "RSI\n운영안정"]
    INDEX_COLS = ["idx_RRI", "idx_OPI", "idx_SRI", "idx_RSI"]

    def __init__(self, paths: Paths):
        self.paths = paths

    def render(self, row: pd.Series, shop_id: str) -> Path:
        values = [row[c] for c in self.INDEX_COLS]
        values += values[:1]
        N = len(self.CATEGORIES)
        angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

        # 12시 방향이 첫 항목(RRI)이 되도록 회전 + 시계방향
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.plot(angles, values, "o-", linewidth=2, color="tomato")
        ax.fill(angles, values, alpha=0.25, color="tomato")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.CATEGORIES, size=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8)
        ax.tick_params(axis="x", pad=15)  # 라벨이 원과 겹치지 않게

        # 제목은 suptitle로 빼서 폴라 라벨과 충돌 방지
        fig.suptitle(
            f"그로몽 지수 레이더 — {shop_id}\n"
            f"스코어 {row['gromong_score']:.1f}점  ({row['grade']}등급)",
            size=12, y=0.98,
        )
        # 위/아래 여백 명시
        fig.subplots_adjust(top=0.85, bottom=0.10, left=0.10, right=0.90)

        out_path = self.paths.output_dir / f"radar_{shop_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.4)
        plt.close(fig)
        return out_path
