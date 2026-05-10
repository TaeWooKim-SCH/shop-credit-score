from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RawDataset:
    """6개의 원본 DataFrame을 담는 컨테이너."""

    shop: pd.DataFrame
    treat: pd.DataFrame
    order: pd.DataFrame
    review: pd.DataFrame
    control: pd.DataFrame
    address: pd.DataFrame

    def shapes(self) -> dict[str, tuple[int, int]]:
        return {
            "shop": self.shop.shape, "treat": self.treat.shape,
            "order": self.order.shape, "review": self.review.shape,
            "control": self.control.shape, "address": self.address.shape,
        }
