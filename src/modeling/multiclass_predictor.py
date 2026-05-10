from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from ..config import TRAIN_QUANTILE, XGB_PARAMS


@dataclass
class MulticlassEvaluationResult:
    classes: list[int]                  # [-1, 0, 1]
    report: str
    cm: np.ndarray
    macro_f1: float
    weighted_f1: float


class MulticlassGrowthPredictor:
    """3-class XGBoost (성장/유지/하락) — binary 분류기와 별도로 학습."""

    LABEL_ORDER = [-1, 0, 1]

    def __init__(self, train_quantile: float = TRAIN_QUANTILE):
        self.train_quantile = train_quantile
        self.model: XGBClassifier | None = None

    def split(self, df: pd.DataFrame, feature_cols: list[str]):
        cutoff = df["year_month_dt"].quantile(self.train_quantile)
        train = df[df["year_month_dt"] <= cutoff]
        test = df[df["year_month_dt"] > cutoff]
        return (
            train[feature_cols], train["growth_label_3c"],
            test[feature_cols], test["growth_label_3c"],
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        # XGB는 라벨이 0,1,2,...로 시작해야 함 → -1,0,1 → 0,1,2 매핑
        self._label_map = {orig: i for i, orig in enumerate(self.LABEL_ORDER)}
        self._inv_map = {i: orig for orig, i in self._label_map.items()}
        y_mapped = y_train.map(self._label_map)

        params = {**XGB_PARAMS, "objective": "multi:softprob", "num_class": 3}
        params.pop("eval_metric", None)
        self.model = XGBClassifier(**params, eval_metric="mlogloss")
        self.model.fit(X_train, y_mapped)
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("train() 먼저 호출")
        pred_idx = self.model.predict(X)
        return np.array([self._inv_map[i] for i in pred_idx])

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """각 클래스에 대한 확률 (열 순서: -1, 0, 1)."""
        if self.model is None:
            raise RuntimeError("train() 먼저 호출")
        proba = self.model.predict_proba(X)
        return pd.DataFrame(
            proba, columns=[f"prob_{self._inv_map[i]}" for i in range(3)], index=X.index,
        )

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> MulticlassEvaluationResult:
        from sklearn.metrics import f1_score
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, labels=self.LABEL_ORDER, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=self.LABEL_ORDER)
        macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        return MulticlassEvaluationResult(
            classes=self.LABEL_ORDER, report=report, cm=cm,
            macro_f1=macro, weighted_f1=weighted,
        )
