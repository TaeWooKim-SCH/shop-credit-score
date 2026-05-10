from __future__ import annotations

from pathlib import Path

import pandas as pd


class ExcelLoader:
    """엑셀 단일 시트 로더."""

    def load(self, file_path: Path | str) -> pd.DataFrame:
        return pd.read_excel(file_path)
