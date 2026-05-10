from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SentimentSignal:
    monthly: pd.DataFrame   # platform_shop_id × year_month × signals
    keyword_top: pd.DataFrame  # 부정/긍정 상위 단어


class LightweightSentimentAnalyzer:
    """경량 한국어 감성 분석 — KoBERT 대안.

    설계 의도:
      - 22만 건 리뷰 텍스트에 대형 LM 추론은 GPU 없이 비현실적
      - 별점이 약한 노이즈가 있으나 ground-truth로 쓸 수 있음 (1-2점=부정, 4-5점=긍정)
      - 부정 키워드 사전 (음식/배달 도메인 특화) + 별점-텍스트 불일치 탐지

    산출:
      - sentiment_neg_kw_count : 매장×월별 부정 키워드 평균 등장 수
      - sentiment_pos_kw_count : 긍정 키워드 평균 등장 수
      - rating_text_inconsist  : 별점-텍스트 불일치 비율 (5점인데 부정 키워드 다수)
      - text_length_mean        : 평균 리뷰 길이 (관여도 지표)
    """

    # 음식/배달 도메인 부정 키워드 (확장 권장)
    NEG_KEYWORDS = [
        "별로", "최악", "실망", "맛없", "맛 없", "차갑", "식었", "느림", "늦",
        "불친절", "짜증", "환불", "재구매 안", "다시는", "비추", "비위생",
        "머리카락", "벌레", "이물질", "딱딱", "퍽퍽", "짜요", "짜다", "싱겁",
        "배달 늦", "오래 걸", "누락", "빠짐", "안 와", "안 옴", "차가워",
        "맛이 없", "양이 적", "양 적", "비싸", "비싼", "별로에", "없어요",
        "안왔", "왜 이래", "최악이", "구려", "꽝", "역겨", "토할",
    ]
    POS_KEYWORDS = [
        "맛있", "맛잇", "최고", "굿", "good", "추천", "또 시킬", "또 시켜",
        "재구매", "친절", "빠르", "신속", "푸짐", "양 많", "양 많아",
        "정성", "감동", "최애", "단골", "사랑", "행복",
        "따뜻", "신선", "고소", "바삭", "촉촉",
    ]
    NEG_RATING_MAX = 2.0
    POS_RATING_MIN = 4.0
    INCONSISTENCY_NEG_KW_THRESHOLD = 2  # 5점인데 부정어 2개 이상 → 의심

    def __init__(self):
        self._neg_pattern = re.compile("|".join(map(re.escape, self.NEG_KEYWORDS)))
        self._pos_pattern = re.compile("|".join(map(re.escape, self.POS_KEYWORDS)))

    def add_review_signals(self, review_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        df = review_df.copy()
        text = df["review.content"].fillna("").astype(str) if "review.content" in df.columns \
               else df.get("review_content", pd.Series("", index=df.index)).fillna("").astype(str)

        df["neg_kw_count"] = text.map(lambda t: len(self._neg_pattern.findall(t)))
        df["pos_kw_count"] = text.map(lambda t: len(self._pos_pattern.findall(t)))
        df["text_length"] = text.str.len()

        # 별점-텍스트 불일치: 5점인데 부정어 다수
        rating = df.get("rating", pd.Series(0, index=df.index))
        df["rating_text_inconsistent"] = (
            (rating >= self.POS_RATING_MIN)
            & (df["neg_kw_count"] >= self.INCONSISTENCY_NEG_KW_THRESHOLD)
        ).astype(int)

        if verbose:
            print(f"  [감성 시그널] 부정 키워드 평균: {df['neg_kw_count'].mean():.3f}/리뷰 | "
                  f"긍정 키워드 평균: {df['pos_kw_count'].mean():.3f}/리뷰")
            print(f"  [별점-텍스트 불일치] {df['rating_text_inconsistent'].sum():,}건 "
                  f"({df['rating_text_inconsistent'].mean()*100:.2f}%)")
        return df

    def aggregate_monthly(self, review_df: pd.DataFrame) -> pd.DataFrame:
        return (
            review_df.groupby(["platform_shop_id", "year_month"], as_index=False)
            .agg(
                sentiment_neg_kw=("neg_kw_count", "mean"),
                sentiment_pos_kw=("pos_kw_count", "mean"),
                text_length_mean=("text_length", "mean"),
                rating_text_inconsist_rate=("rating_text_inconsistent", "mean"),
            )
        )

    def top_keywords(self, review_df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
        text_col = "review.content" if "review.content" in review_df.columns else "review_content"
        rating = review_df.get("rating", pd.Series(0, index=review_df.index))
        neg_text = " ".join(
            review_df.loc[rating <= self.NEG_RATING_MAX, text_col].fillna("").astype(str).tolist()
        )
        pos_text = " ".join(
            review_df.loc[rating >= self.POS_RATING_MIN, text_col].fillna("").astype(str).tolist()
        )

        records = []
        for kw in self.NEG_KEYWORDS:
            records.append({
                "keyword": kw, "polarity": "negative",
                "neg_count": neg_text.count(kw), "pos_count": pos_text.count(kw),
            })
        for kw in self.POS_KEYWORDS:
            records.append({
                "keyword": kw, "polarity": "positive",
                "neg_count": neg_text.count(kw), "pos_count": pos_text.count(kw),
            })

        df = pd.DataFrame(records)
        # discriminative power = (pos_count - neg_count) / total
        total = df["pos_count"] + df["neg_count"] + 1
        df["discrimination"] = (df["pos_count"] - df["neg_count"]) / total
        return df.sort_values("discrimination", ascending=False).head(top_n)
