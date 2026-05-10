import json
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

try:
    import koreanize_matplotlib  # noqa: F401  (side-effect: registers font)
except (ImportError, ModuleNotFoundError):
    koreanize_matplotlib = None


_MAC_FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/AppleGothic.ttf",
]
_LINUX_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
]


def setup_korean_font() -> None:
    families = {f.name for f in fm.fontManager.ttflist}
    target = None
    if "NanumGothic" in families:
        target = "NanumGothic"
    else:
        for path in _LINUX_FONT_CANDIDATES + _MAC_FONT_CANDIDATES:
            if os.path.exists(path):
                fm.fontManager.addfont(path)
        families = {f.name for f in fm.fontManager.ttflist}
        for cand in ("NanumGothic", "AppleSDGothicNeo", "AppleGothic"):
            if cand in families:
                target = cand
                break
    if target:
        plt.rcParams.update({"font.family": target, "axes.unicode_minus": False})
    else:
        plt.rcParams.update({"axes.unicode_minus": False})


def load_json_auto(file_path) -> pd.DataFrame:
    try:
        return pd.read_json(file_path)
    except ValueError:
        try:
            return pd.read_json(file_path, lines=True)
        except ValueError:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip()
        .replace(" ", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "_")
        for c in df.columns
    ]
    return df


def rename_if_exists(df: pd.DataFrame, rename_dict: dict) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
