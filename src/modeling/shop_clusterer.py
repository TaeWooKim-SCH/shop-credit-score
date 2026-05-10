from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterResult:
    n_clusters: int
    silhouette: float
    cluster_summary: pd.DataFrame  # cluster_id별 평균 통계
    outlier_count: int


class ShopClusterer:
    """매장 군집화 (KMeans) + 이상치 탐지 (Isolation Forest).

    유사 매장 그룹 내에서 상대 평가 → 등급 공정성 ↑
    이상치는 별도 처리 또는 보수적 등급 부여 권장.
    """

    DEFAULT_FEATURES = [
        "monthly_order_count", "monthly_sales", "avg_rating",
        "negative_review_ratio", "any_reply_rate", "menu_count",
        "active_menu_ratio", "order_cv",
    ]
    OUTLIER_CONTAMINATION = 0.05

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans: KMeans | None = None
        self.isolation: IsolationForest | None = None
        self.feature_cols: list[str] = []

    def fit_predict(
        self, df: pd.DataFrame, feature_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, ClusterResult]:
        cols = feature_cols or self.DEFAULT_FEATURES
        cols = [c for c in cols if c in df.columns]
        if len(cols) < 2:
            raise ValueError(f"군집화에 사용할 피처가 부족합니다: {cols}")
        self.feature_cols = cols

        X = df[cols].fillna(0).values
        Xs = self.scaler.fit_transform(X)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10,
        )
        cluster_id = self.kmeans.fit_predict(Xs)

        self.isolation = IsolationForest(
            contamination=self.OUTLIER_CONTAMINATION,
            random_state=self.random_state, n_jobs=-1,
        )
        outlier = (self.isolation.fit_predict(Xs) == -1).astype(int)

        out = df.copy()
        out["cluster_id"] = cluster_id
        out["is_outlier"] = outlier

        sil = float(silhouette_score(Xs, cluster_id)) if len(set(cluster_id)) > 1 else 0.0
        summary = out.groupby("cluster_id")[cols].mean().round(3)
        summary["count"] = out.groupby("cluster_id").size()

        return out, ClusterResult(
            n_clusters=self.n_clusters,
            silhouette=sil,
            cluster_summary=summary,
            outlier_count=int(outlier.sum()),
        )

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.kmeans is None:
            raise RuntimeError("fit_predict() 먼저 호출")
        X = df[self.feature_cols].fillna(0).values
        Xs = self.scaler.transform(X)
        return pd.Series(self.kmeans.predict(Xs), index=df.index, name="cluster_id")
