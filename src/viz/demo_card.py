from __future__ import annotations

import pandas as pd

from ..config import GRADE_EMOJI


class DemoCard:
    """매장 1개에 대한 콘솔 카드 + AI 추천 액션 출력."""

    THRESHOLDS = {
        "RRI_HIGH": 0.60, "SRI_HIGH": 0.50,
        "RSI_LOW": 0.30, "OPI_HIGH": 0.60,
    }

    def render(self, row: pd.Series) -> None:
        emoji = GRADE_EMOJI.get(row["grade"], "")
        print(self._header(row, emoji))
        for line in self.actions(row):
            print(f"║  → {line}")
        print("╚══════════════════════════════════════════════════════╝")

    def actions(self, r: pd.Series) -> list[str]:
        """AI 추천 액션을 텍스트 리스트로 반환 (콘솔/GUI 양쪽 재사용)."""
        out: list[str] = []
        if r["idx_RRI"] > self.THRESHOLDS["RRI_HIGH"]:
            out.append("리뷰 응답률이 낮습니다. 댓글몽 도입 시 즉각 효과 기대 🚀")
        if r["idx_SRI"] > self.THRESHOLDS["SRI_HIGH"]:
            out.append("부정 리뷰 관리 필요. AI 자동응답으로 개선 여지 큼 📝")
        if r["idx_RSI"] < self.THRESHOLDS["RSI_LOW"]:
            out.append("운영 불안정. 댓글몽 도입 전 근본 문제 점검 필요 ⚠️")
        if r["idx_OPI"] > self.THRESHOLDS["OPI_HIGH"]:
            out.append("기본 주문 체력 우수. 댓글몽 효과 극대화 기대 ✅")
        if r["grade"] in ["S", "A"]:
            out.append(f"[{r['grade']}등급] 댓글몽 영업 우선 타겟 매장입니다 🎯")
        elif r["grade"] == "D":
            out.append("[D등급] 근본 운영 문제 해결 후 도입 권장")
        elif r["grade"] == "C":
            out.append("[C등급] 데이터 추가 확보 또는 운영 개선 후 재평가")

        if not out:
            out.append("전반적으로 안정적인 운영 상태입니다. 꾸준한 관리 유지 권장 📊")
        return out

    def _header(self, r: pd.Series, emoji: str) -> str:
        reply_rate = r.get("any_reply_rate", r.get("owner_reply_rate", 0))
        return f"""
╔══════════════════════════════════════════════════════╗
║  매장 ID   : {r["platform_shop_id"]}
║  매장명    : {r.get("shop_name", "")}
║  그로몽 스코어 : {r["gromong_score"]:.1f}점   [{r["grade"]}등급] {emoji}
║  성장 확률     : {r["growth_probability"] * 100:.1f}%
╠══════════════════════════════════════════════════════╣
║  [4개 지수 점수]
║  RRI 리뷰 응답 개선 여지  : {r["idx_RRI"] * 100:.1f}점
║  OPI 주문 성과 지수        : {r["idx_OPI"] * 100:.1f}점
║  SRI 감성 개선 여지        : {r["idx_SRI"] * 100:.1f}점
║  RSI 운영 안정성 지수      : {r["idx_RSI"] * 100:.1f}점
╠══════════════════════════════════════════════════════╣
║  [현황 요약]
║  평균 별점       : {r["avg_rating"]:.2f} / 5.0
║  리뷰 응답률     : {reply_rate * 100:.1f}%
║  부정 리뷰 비율  : {r.get("negative_review_ratio", 0) * 100:.1f}%
║  월 평균 주문    : {r["monthly_order_count"]:.0f}건
╠══════════════════════════════════════════════════════╣
║  [AI 추천 액션]"""
