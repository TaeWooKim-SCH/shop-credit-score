from __future__ import annotations

import pandas as pd

from ..config import SHRINKAGE_PRIOR_STRENGTH


class ReviewAggregator:
    """리뷰 raw → 매장×월 패널. 날짜/평점 컬럼 자동 감지 + 댓글 3유형 분리."""

    DATE_CANDIDATES = [
        "review_created_at", "created_at", "review_date", "date",
        "review_time", "created_time", "reg_date", "write_date",
    ]
    DATE_REGEX = r"\d{4}"
    NEGATIVE_THRESHOLD = 2

    # replies.request_status → 댓글 유형 추정 매핑
    # TODO(르몽 확인): 정확한 status 코드 의미를 르몽에 확인 후 확정. 아래는 샘플 톤 분석 기반 추정.
    #   - status 3 (64%): 정형화된 친절체 → 댓글몽 AI 자동응답으로 추정
    #   - status 2, 6   : "신메뉴 출시" 등 프로모션 → 마케팅 댓글로 추정
    #   - status 4, 10  : 매장명 직접 언급, 사장님 톤 → 사장님 직접 응답으로 추정
    AI_REPLY_STATUS = {3}
    MARKETING_REPLY_STATUS = {2, 6}
    OWNER_REPLY_STATUS = {4, 10}

    def transform(self, review_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        df = review_df.copy()
        date_col = self._detect_date_column(df)
        if verbose:
            print(f"  리뷰 날짜 컬럼: '{date_col}' 사용")

        df["review_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["year_month"] = df["review_date"].dt.to_period("M").astype(str)
        df["rating"] = self._extract_rating(df)

        df = self._add_reply_flags(df, verbose=verbose)
        df["negative_flag"] = (df["rating"] <= self.NEGATIVE_THRESHOLD).astype(int)
        return df

    def _add_reply_flags(self, df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
        comment_col = "replies_comment" if "replies_comment" in df.columns else None
        status_col = "replies_request_status" if "replies_request_status" in df.columns else None

        any_reply = (
            df[comment_col].notna() if comment_col is not None
            else pd.Series(False, index=df.index)
        )

        if status_col is not None:
            status = pd.to_numeric(df[status_col], errors="coerce")
            ai_mask = status.isin(self.AI_REPLY_STATUS) & any_reply
            mkt_mask = status.isin(self.MARKETING_REPLY_STATUS) & any_reply
            owner_mask = status.isin(self.OWNER_REPLY_STATUS) & any_reply
            df["ai_reply_flag"] = ai_mask.astype(int)
            df["marketing_reply_flag"] = mkt_mask.astype(int)
            # owner_reply_flag: 사장님 직접 응답 — status 매핑 우선, 없으면 전체 응답률
            df["owner_reply_flag"] = owner_mask.astype(int)
            # any_reply_flag: 응답 자체 여부 (3유형 합) — RRI 기존 의미 유지용
            df["any_reply_flag"] = any_reply.astype(int)

            if verbose:
                total = len(df)
                print(f"  [댓글 3유형 분리] AI={df['ai_reply_flag'].sum():,}건 "
                      f"({df['ai_reply_flag'].mean()*100:.1f}%) | "
                      f"마케팅={df['marketing_reply_flag'].sum():,}건 "
                      f"({df['marketing_reply_flag'].mean()*100:.1f}%) | "
                      f"사장님={df['owner_reply_flag'].sum():,}건 "
                      f"({df['owner_reply_flag'].mean()*100:.1f}%) "
                      f"/ 전체 응답 {df['any_reply_flag'].sum():,}건")
        else:
            # status 컬럼 없을 때: 응답 여부만 owner로 사용 (구버전 호환)
            df["owner_reply_flag"] = any_reply.astype(int)
            df["ai_reply_flag"] = 0
            df["marketing_reply_flag"] = 0
            df["any_reply_flag"] = any_reply.astype(int)

        return df

    def aggregate_monthly(self, review_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        df = self.transform(review_df, verbose=verbose)
        monthly = (
            df.groupby(["platform_shop_id", "year_month"], as_index=False)
            .agg(
                monthly_review_count=("platform_shop_id", "size"),
                avg_rating=("rating", "mean"),
                rating_std=("rating", "std"),
                owner_reply_rate=("owner_reply_flag", "mean"),
                ai_reply_rate=("ai_reply_flag", "mean"),
                marketing_reply_rate=("marketing_reply_flag", "mean"),
                any_reply_rate=("any_reply_flag", "mean"),
                negative_review_ratio=("negative_flag", "mean"),
            )
        )
        monthly["rating_std"] = monthly["rating_std"].fillna(0)
        monthly = self._add_shrunk_metrics(monthly, verbose=verbose)
        return monthly

    def _add_shrunk_metrics(self, monthly: pd.DataFrame, verbose: bool) -> pd.DataFrame:
        """Empirical Bayes shrinkage: 표본이 작은 매장의 평점/응답률/부정비율을
        글로벌 평균으로 회귀시켜 표본 크기 보정."""
        prior = SHRINKAGE_PRIOR_STRENGTH
        n = monthly["monthly_review_count"]

        # 글로벌 평균은 리뷰 weight를 반영해 가중평균
        def weighted_mean(col: str) -> float:
            mask = n > 0
            return float((monthly.loc[mask, col] * n[mask]).sum() / n[mask].sum())

        global_rating = weighted_mean("avg_rating")
        global_reply = weighted_mean("any_reply_rate")
        global_neg = weighted_mean("negative_review_ratio")

        denom = n + prior
        monthly["avg_rating_shrunk"] = (
            monthly["avg_rating"] * n + global_rating * prior
        ) / denom
        monthly["any_reply_rate_shrunk"] = (
            monthly["any_reply_rate"] * n + global_reply * prior
        ) / denom
        monthly["negative_review_ratio_shrunk"] = (
            monthly["negative_review_ratio"] * n + global_neg * prior
        ) / denom

        if verbose:
            print(f"  [Shrinkage] prior={prior} | "
                  f"global rating={global_rating:.3f} / "
                  f"reply={global_reply:.3f} / neg={global_neg:.3f}")
        return monthly

    def _detect_date_column(self, df: pd.DataFrame) -> str:
        for c in self.DATE_CANDIDATES:
            if c in df.columns:
                return c
        for c in df.columns:
            if df[c].dtype == "object":
                hit_rate = df[c].astype(str).str.match(self.DATE_REGEX, na=False).mean()
                if hit_rate > 0.5:
                    return c
        raise KeyError(f"날짜 컬럼을 찾을 수 없습니다. 컬럼: {df.columns.tolist()}")

    def _extract_rating(self, df: pd.DataFrame) -> pd.Series:
        if "review_rating" in df.columns:
            return pd.to_numeric(df["review_rating"], errors="coerce")
        if "rating" in df.columns:
            return pd.to_numeric(df["rating"], errors="coerce")
        raise KeyError("rating 컬럼 없음")
