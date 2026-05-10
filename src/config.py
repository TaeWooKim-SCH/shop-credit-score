from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    base_dir: Path = Path("data")
    output_dir: Path = Path("output")

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)

    @property
    def shop_file(self) -> Path:
        return self.base_dir / "매장_식별_데이터.json"

    @property
    def treat_file(self) -> Path:
        return self.base_dir / "처치_시점_정의_데이터.json"

    @property
    def order_file(self) -> Path:
        return self.base_dir / "order_주요_성과_변수.json"

    @property
    def review_file(self) -> Path:
        return self.base_dir / "reviews_주요_성과_변수.json"

    @property
    def control_file(self) -> Path:
        return self.base_dir / "20260310_통제변수.json"

    @property
    def address_file(self) -> Path:
        return self.base_dir / "final_shop_address.xlsx"

    def all_input_files(self):
        return [
            self.shop_file, self.treat_file, self.order_file,
            self.review_file, self.control_file, self.address_file,
        ]


DEFAULT_WEIGHTS = {
    "RRI (응답개선)": 0.30,
    "OPI (주문성과)": 0.30,
    "SRI (감성개선)": 0.25,
    "RSI (운영안정)": 0.15,
}

INDEX_KEY_MAP = {
    "idx_RRI": "RRI (응답개선)",
    "idx_OPI": "OPI (주문성과)",
    "idx_SRI": "SRI (감성개선)",
    "idx_RSI": "RSI (운영안정)",
}

GRADE_LABELS = ["D", "C", "B", "A", "S"]
GRADE_ORDER = ["D", "C", "B", "A", "S"]

# 등급 산출 방식 — "quantile" (분포 기반 자동) 또는 "fixed" (GRADE_BINS 사용)
GRADE_BIN_METHOD = "quantile"

# quantile 모드: 목표 비율 (D 하위 → S 상위 순, 합=1.0)
# 예) D=8%, C=42%, B=30%, A=15%, S=5% → 신용평가 등급 분포 유사
GRADE_TARGET_RATIOS = {"D": 0.08, "C": 0.42, "B": 0.30, "A": 0.15, "S": 0.05}

# fixed 모드용 (legacy, 첫 실행 분포 기반)
GRADE_BINS = [0, 25, 38, 50, 60, 101]
GRADE_EMOJI = {"S": "🏆", "A": "✅", "B": "🔵", "C": "🟡", "D": "🔴"}
GRADE_COLORS = {
    "D": "#e74c3c", "C": "#e67e22", "B": "#3498db",
    "A": "#27ae60", "S": "#8e44ad",
}

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
)

GROWTH_THRESHOLD = 0.10           # binary: 미래 주문 +10% 이상 → 성장(1)
GROWTH_DECLINE_THRESHOLD = -0.10  # 3-class: 미래 주문 -10% 이하 → 하락(-1)
GROWTH_HORIZON_MONTHS = 3

# Bayesian shrinkage (Empirical Bayes) — 표본 크기가 작을 때 글로벌 평균으로 회귀
# prior 강도 = 가상 리뷰 N개 만큼의 weight를 글로벌 평균에 부여
# 리뷰 1건 5점 매장이 통계적 유의성 없이 만점 받는 것을 방지
SHRINKAGE_PRIOR_STRENGTH = 10
TRAIN_QUANTILE = 0.8
MIN_MONTHS_FOR_SCORING = 6
PSM_CALIPER_MULT = 0.2
EVENT_WINDOW = (-6, 6)
