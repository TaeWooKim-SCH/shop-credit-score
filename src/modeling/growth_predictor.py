from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier

from ..config import TRAIN_QUANTILE, XGB_PARAMS


@dataclass
class TrainTestSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


@dataclass
class EvaluationResult:
    auc: float
    report: str
    cm: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    overfit_warning: bool


class GrowthPredictor:
    """XGBoost 학습 + 시간 기반 train/test 분할 + 평가."""

    OVERFIT_AUC_THRESHOLD = 0.99

    def __init__(self, train_quantile: float = TRAIN_QUANTILE):
        self.train_quantile = train_quantile
        self.model: XGBClassifier | None = None

    def split(self, df: pd.DataFrame, feature_cols: list[str]) -> TrainTestSplit:
        cutoff = df["year_month_dt"].quantile(self.train_quantile)
        train = df[df["year_month_dt"] <= cutoff]
        test = df[df["year_month_dt"] > cutoff]
        return TrainTestSplit(
            X_train=train[feature_cols], y_train=train["growth_label"],
            X_test=test[feature_cols], y_test=test["growth_label"],
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        self.model = XGBClassifier(scale_pos_weight=scale_pos, **XGB_PARAMS)
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> EvaluationResult:
        if self.model is None:
            raise RuntimeError("train() 를 먼저 호출하세요.")
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        return EvaluationResult(
            auc=auc,
            report=classification_report(y_test, y_pred),
            cm=confusion_matrix(y_test, y_pred),
            fpr=fpr, tpr=tpr,
            overfit_warning=auc >= self.OVERFIT_AUC_THRESHOLD,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("train() 를 먼저 호출하세요.")
        return self.model.predict_proba(X)[:, 1]
