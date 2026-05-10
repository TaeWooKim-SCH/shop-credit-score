from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class JsonLoader:
    """JSON 형식(레코드 / 라인 / dict) 자동 감지 로더."""

    def load(self, file_path: Path | str) -> pd.DataFrame:
        try:
            return pd.read_json(file_path)
        except ValueError:
            try:
                return pd.read_json(file_path, lines=True)
            except ValueError:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return pd.DataFrame(data)
