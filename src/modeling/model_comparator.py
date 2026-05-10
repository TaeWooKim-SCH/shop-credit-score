from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


@dataclass
class ModelScore:
    name: str
    auc_mean: float
    auc_std: float
    pr_auc_mean: float
    f1_mean: float
    log_loss_mean: float
    brier_mean: float
    fit_seconds: float


class ModelComparator:
    """여러 분류기를 동일한 시계열 CV로 비교 + 베스트 추천.

    평가 기준:
      - ROC-AUC (분리 능력)
      - PR-AUC (불균형 클래스에 더 적합)
      - F1 (임계값 0.5 기준)
      - Log Loss (확률 정확성)
      - Brier Score (calibration 품질)
      - Fit time (실전 속도)
    """

    N_SPLITS = 5
    DEFAULT_F1_THRESHOLD = 0.5

    def __init__(self, scale_pos_weight: float = 1.0, random_state: int = 42):
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state

    def candidates(self) -> dict[str, object]:
        return {
            "XGBoost": XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.random_state, eval_metric="logloss",
                verbosity=0,
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=200, max_depth=-1, num_leaves=31,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.random_state, verbose=-1,
            ),
            "CatBoost": CatBoostClassifier(
                iterations=200, depth=5, learning_rate=0.05,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.random_state, verbose=False,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=300, max_depth=10,
                class_weight="balanced", n_jobs=-1,
                random_state=self.random_state,
            ),
            "LogisticRegression": LogisticRegression(
                max_iter=1000, class_weight="balanced",
                random_state=self.random_state,
            ),
        }

    def compare(
        self,
        X: pd.DataFrame, y: pd.Series,
        sort_by: str = "auc_mean", verbose: bool = True,
    ) -> list[ModelScore]:
        tscv = TimeSeriesSplit(n_splits=self.N_SPLITS)
        results: list[ModelScore] = []

        for name, model in self.candidates().items():
            aucs, pr_aucs, f1s, lls, briers, times = [], [], [], [], [], []
            for tr, te in tscv.split(X):
                X_tr, X_te = X.iloc[tr], X.iloc[te]
                y_tr, y_te = y.iloc[tr], y.iloc[te]
                t0 = time.time()
                try:
                    model.fit(X_tr, y_tr)
                    times.append(time.time() - t0)
                    proba = model.predict_proba(X_te)[:, 1]
                    pred = (proba >= self.DEFAULT_F1_THRESHOLD).astype(int)
                    aucs.append(roc_auc_score(y_te, proba))
                    pr_aucs.append(average_precision_score(y_te, proba))
                    f1s.append(f1_score(y_te, pred, zero_division=0))
                    lls.append(log_loss(y_te, np.clip(proba, 1e-9, 1 - 1e-9)))
                    briers.append(brier_score_loss(y_te, proba))
                except Exception as e:
                    if verbose:
                        print(f"  {name} fold 실패: {e}")
                    continue

            if not aucs:
                continue

            score = ModelScore(
                name=name,
                auc_mean=float(np.mean(aucs)),
                auc_std=float(np.std(aucs)),
                pr_auc_mean=float(np.mean(pr_aucs)),
                f1_mean=float(np.mean(f1s)),
                log_loss_mean=float(np.mean(lls)),
                brier_mean=float(np.mean(briers)),
                fit_seconds=float(np.mean(times)),
            )
            results.append(score)
            if verbose:
                print(f"  {name:18s} | AUC={score.auc_mean:.4f}±{score.auc_std:.4f} "
                      f"| PR-AUC={score.pr_auc_mean:.4f} | F1={score.f1_mean:.4f} "
                      f"| LogLoss={score.log_loss_mean:.4f} "
                      f"| Brier={score.brier_mean:.4f} | fit={score.fit_seconds:.2f}s")

        results.sort(key=lambda s: getattr(s, sort_by), reverse=True)
        return results

    @staticmethod
    def to_dataframe(results: list[ModelScore]) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in results])
