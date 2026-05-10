from __future__ import annotations

import pandas as pd


class ColumnCleaner:
    """컬럼명 표준화 + 선택적 리네임."""

    REPLACEMENTS = [(" ", "_"), (" ", "_"), ("/", "_"), (".", "_")]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [self._normalize(c) for c in df.columns]
        return df

    def rename_if_exists(self, df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
        return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

    def _normalize(self, name: str) -> str:
        out = str(name).strip()
        for old, new in self.REPLACEMENTS:
            out = out.replace(old, new)
        return out
