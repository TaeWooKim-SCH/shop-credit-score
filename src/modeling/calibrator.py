from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class CalibrationReport:
    method: str
    brier_before: float
    brier_after: float
    log_loss_before: float
    log_loss_after: float
    calibration_curve: pd.DataFrame  # bin_mean_pred, bin_frac_positive, bin_count


class ProbabilityCalibrator:
    """확률 보정. Platt scaling (sigmoid) 또는 Isotonic regression.

    분류기는 확률처럼 보이는 값을 출력하지만 실제 확률과 차이가 있음.
    예: 모델이 "0.7" 라고 했을 때 실제로 70%의 확률로 양성이어야 신뢰 가능.
    """

    METHODS = ("isotonic", "sigmoid")
    N_BINS = 10

    def __init__(self, method: str = "isotonic"):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method
        self._calibrator: CalibratedClassifierCV | None = None

    def fit(self, base_estimator, X_train, y_train, X_cal, y_cal) -> "ProbabilityCalibrator":
        """base_estimator는 사전 학습된 모델. X_cal/y_cal은 holdout calibration 셋."""
        # sklearn 1.6+ : cv="prefit" 제거 → FrozenEstimator 패턴 사용
        self._calibrator = CalibratedClassifierCV(
            FrozenEstimator(base_estimator), method=self.method,
        )
        self._calibrator.fit(X_cal, y_cal)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._calibrator is None:
            raise RuntimeError("fit() 먼저 호출")
        return self._calibrator.predict_proba(X)[:, 1]

    def report(
        self, base_estimator, X_test: pd.DataFrame, y_test: pd.Series,
    ) -> CalibrationReport:
        proba_before = base_estimator.predict_proba(X_test)[:, 1]
        proba_after = self.predict_proba(X_test)
        clip = lambda p: np.clip(p, 1e-9, 1 - 1e-9)

        curve = self._calibration_curve(y_test, proba_after)

        return CalibrationReport(
            method=self.method,
            brier_before=float(brier_score_loss(y_test, proba_before)),
            brier_after=float(brier_score_loss(y_test, proba_after)),
            log_loss_before=float(log_loss(y_test, clip(proba_before))),
            log_loss_after=float(log_loss(y_test, clip(proba_after))),
            calibration_curve=curve,
        )

    def _calibration_curve(self, y_true: pd.Series, y_proba: np.ndarray) -> pd.DataFrame:
        bins = np.linspace(0, 1, self.N_BINS + 1)
        idx = np.digitize(y_proba, bins[1:-1])
        records = []
        for b in range(self.N_BINS):
            mask = idx == b
            n = int(mask.sum())
            if n == 0:
                continue
            records.append({
                "bin_mean_pred": float(y_proba[mask].mean()),
                "bin_frac_positive": float(y_true.values[mask].mean()),
                "bin_count": n,
            })
        return pd.DataFrame(records)
