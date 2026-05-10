"""그로몽 스코어 — Streamlit 데모 GUI.

실행:
  streamlit run app.py

전제: 먼저 `python -m src` 로 파이프라인을 한 번 실행해 output/ 산출물이 있어야 함.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import DEFAULT_WEIGHTS, GRADE_COLORS, GRADE_ORDER, Paths
from src.viz.demo_card import DemoCard

st.set_page_config(
    page_title="그로몽 스코어",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 사이드바 너비 확장 + 폰트 크기/여백 조정
st.markdown("""
<style>
[data-testid="stSidebar"] {
    min-width: 360px !important;
    max-width: 480px !important;
    width: 400px !important;
}
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width: 400px !important;
}
[data-testid="stMetricValue"] { font-size: 1.4rem; }
[data-testid="stMetricLabel"] { font-size: 0.85rem; }
/* 본문 상단 여백 확보 (Streamlit toolbar/header에 탭이 가려지는 문제 방지) */
.block-container {
    padding-top: 4rem !important;
    padding-bottom: 1rem;
}
/* 탭 자체 여백 */
[data-testid="stTabs"] {
    margin-top: 0.5rem;
}
[data-baseweb="tab-list"] {
    gap: 0.5rem;
}
[data-baseweb="tab"] {
    height: 3rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
hr { margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── 이모지 제거 유틸 ─────────────────────────────────────────
_EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF"
    "\U0001F1E0-\U0001F1FF" "\U00002B00-\U00002BFF" "]+",
    flags=re.UNICODE,
)
def strip_emoji(s: str) -> str:
    return _EMOJI_RE.sub("", s).strip()


# ── 데이터 로더 ──────────────────────────────────────────────
@st.cache_data
def load_scores(output_dir: str) -> pd.DataFrame:
    p = Path(output_dir) / "latest_shop_scores.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_shap_weights(output_dir: str) -> dict[str, float]:
    p = Path(output_dir) / "shap_weights.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data
def load_did_summary(output_dir: str) -> dict:
    p = Path(output_dir) / "did_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data
def load_model_comparison(output_dir: str) -> pd.DataFrame:
    p = Path(output_dir) / "model_comparison.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_threshold(output_dir: str) -> dict:
    p = Path(output_dir) / "threshold_analysis.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data
def load_calibration(output_dir: str) -> dict:
    p = Path(output_dir) / "calibration_report.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data
def load_multiclass(output_dir: str) -> dict:
    p = Path(output_dir) / "multiclass_evaluation.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data
def load_clusters(output_dir: str) -> pd.DataFrame:
    p = Path(output_dir) / "cluster_summary.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_keywords(output_dir: str) -> pd.DataFrame:
    p = Path(output_dir) / "sentiment_keywords.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_shap_importance(output_dir: str) -> pd.DataFrame:
    p = Path(output_dir) / "shap_feature_importance.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_model_dataset(output_dir: str) -> pd.DataFrame:
    p = Path(output_dir) / "model_dataset_with_score.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    cols = ["platform_shop_id", "year_month_dt"]
    extra = [c for c in ["cluster_id", "is_outlier", "prob_-1", "prob_0", "prob_1"]
             if c in df.columns]
    df = df[cols + extra].copy()
    df["year_month_dt"] = pd.to_datetime(df["year_month_dt"], errors="coerce")
    return (
        df.sort_values(["platform_shop_id", "year_month_dt"])
        .groupby("platform_shop_id").tail(1)
    )


# ── 차트 ─────────────────────────────────────────────────────
def render_radar(row: pd.Series) -> go.Figure:
    categories = ["RRI<br>응답개선", "OPI<br>주문성과", "SRI<br>감성개선", "RSI<br>운영안정"]
    values = [row["idx_RRI"], row["idx_OPI"], row["idx_SRI"], row["idx_RSI"]]
    fig = go.Figure(
        data=go.Scatterpolar(
            r=values + values[:1],
            theta=categories + categories[:1],
            fill="toself",
            line=dict(color="#FF6B6B", width=2),
            fillcolor="rgba(255,107,107,0.25)",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                tickfont=dict(size=10),
            ),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        showlegend=False,
        # 라벨이 잘리지 않도록 위/아래 여백 충분히 확보
        margin=dict(l=60, r=60, t=50, b=50),
        height=420,
    )
    return fig


def render_grade_distribution(scores: pd.DataFrame) -> go.Figure:
    counts = scores["grade"].value_counts().reindex(GRADE_ORDER, fill_value=0)
    fig = go.Figure(
        go.Bar(
            x=counts.index, y=counts.values,
            marker_color=[GRADE_COLORS[g] for g in counts.index],
            text=counts.values, textposition="outside",
        )
    )
    fig.update_layout(
        title="등급 분포", xaxis_title=None, yaxis_title="매장 수",
        height=240, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def render_score_histogram(scores: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        scores, x="gromong_score", color="grade",
        nbins=30, color_discrete_map=GRADE_COLORS,
        category_orders={"grade": GRADE_ORDER},
    )
    fig.update_layout(
        title="그로몽 스코어 분포 (등급별)",
        xaxis_title="스코어", yaxis_title="매장 수",
        height=380, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def render_weight_comparison(shap_w: dict[str, float]) -> go.Figure:
    keys = list(DEFAULT_WEIGHTS.keys())
    short = {k: k.split(" ")[0] for k in keys}
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="SHAP", x=[short[k] for k in keys],
        y=[shap_w.get(k, 0) for k in keys],
        marker_color="#FF6B6B",
    ))
    fig.add_trace(go.Bar(
        name="기본값", x=[short[k] for k in keys],
        y=[DEFAULT_WEIGHTS[k] for k in keys],
        marker_color="#4682B4",
    ))
    fig.update_layout(
        barmode="group", height=260, title="SHAP vs 기본 가중치",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def render_cluster_scatter(scores: pd.DataFrame) -> go.Figure | None:
    if "cluster_id" not in scores.columns:
        return None
    df = scores.dropna(subset=["cluster_id"])
    if df.empty:
        return None
    df = df.copy()
    df["cluster_label"] = df["cluster_id"].astype(int).map(lambda c: f"Cluster {c}")
    fig = px.scatter(
        df, x="monthly_order_count", y="avg_rating",
        color="cluster_label", size="gromong_score",
        hover_data=["shop_name", "grade", "gromong_score"],
        labels={"monthly_order_count": "월 평균 주문", "avg_rating": "평균 별점"},
    )
    fig.update_layout(
        title="매장 군집 산점도 (주문 × 별점, 크기=스코어)",
        height=460, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def render_index_table(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "지수": ["RRI 응답개선", "OPI 주문성과", "SRI 감성개선", "RSI 운영안정"],
        "점수": [
            f"{row['idx_RRI'] * 100:.1f}",
            f"{row['idx_OPI'] * 100:.1f}",
            f"{row['idx_SRI'] * 100:.1f}",
            f"{row['idx_RSI'] * 100:.1f}",
        ],
        "0~1": [
            row["idx_RRI"], row["idx_OPI"], row["idx_SRI"], row["idx_RSI"],
        ],
    })


# ── 사이드바 ────────────────────────────────────────────────
def render_sidebar(scores: pd.DataFrame, did: dict) -> None:
    with st.sidebar:
        st.title("그로몽 스코어")
        st.caption("AI 기반 성장 유망 매장 스코어링 시스템")
        st.divider()

        st.subheader("Overview")
        a, b = st.columns(2)
        a.metric("스코어링 매장", f"{len(scores):,}")
        b.metric("평균 스코어", f"{scores['gromong_score'].mean():.1f}")
        c, d = st.columns(2)
        c.metric("최고 스코어", f"{scores['gromong_score'].max():.1f}")
        d.metric("S+A 등급", f"{int((scores['grade'].isin(['S','A'])).sum()):,}")

        st.plotly_chart(render_grade_distribution(scores), use_container_width=True)

        if did:
            st.divider()
            st.subheader("인과추론 결과")
            st.metric(
                "DID ATT (월 추가 주문)",
                f"+{did.get('att', 0):.1f}건",
                help=f"95% CI [{did.get('ci_lo', 0):.1f}, {did.get('ci_hi', 0):.1f}], "
                     f"p={did.get('p', 0):.1e}",
            )
            if did.get("att_psm") is not None:
                st.metric(
                    "PSM-DID ATT",
                    f"+{did['att_psm']:.1f}건",
                    help=f"매칭 {did.get('matched_n', 0)}쌍, p={did.get('p_psm', 0):.1e}",
                )
            pt_p = did.get("parallel_trends_p")
            if pt_p is not None:
                ok = pt_p > 0.05
                st.metric(
                    "Parallel Trends p",
                    f"{pt_p:.3f}",
                    delta="충족" if ok else "위배 가능",
                    delta_color="normal" if ok else "inverse",
                )

        st.divider()


# ── 탭 1: 매장 분석 ─────────────────────────────────────────
def render_tab_shop(scores: pd.DataFrame) -> None:
    st.subheader("매장 검색")

    f1, f2, f3 = st.columns([3, 2, 2])
    with f1:
        search = st.text_input(
            "매장 ID 또는 매장명",
            placeholder="예: ba_10015935 또는 굽네치킨",
            label_visibility="collapsed",
        )
    with f2:
        grade_filter = st.multiselect(
            "등급 필터", GRADE_ORDER, default=GRADE_ORDER,
            label_visibility="collapsed",
        )
    with f3:
        sort_option = st.selectbox(
            "정렬", ["스코어 ↓", "스코어 ↑", "성장 확률 ↓"],
            label_visibility="collapsed",
        )

    sort_map = {
        "스코어 ↓": ("gromong_score", False),
        "스코어 ↑": ("gromong_score", True),
        "성장 확률 ↓": ("growth_probability", False),
    }
    sort_col, asc = sort_map[sort_option]
    filtered = scores[scores["grade"].isin(grade_filter)]
    if search:
        mask = (
            filtered["platform_shop_id"].astype(str).str.contains(search, case=False, na=False)
            | filtered["shop_name"].astype(str).str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]
    filtered = filtered.sort_values(sort_col, ascending=asc)

    if filtered.empty:
        st.warning("조건에 맞는 매장이 없습니다.")
        return

    options = filtered.head(200).apply(
        lambda r: f"{r['platform_shop_id']}  |  {r['shop_name']}  ·  "
                  f"{r['gromong_score']:.1f}점 ({r['grade']})",
        axis=1,
    ).tolist()
    pick = st.selectbox(
        f"매장 선택 — 검색 결과 {len(filtered):,}개 중 상위 {min(200, len(filtered))}",
        options,
    )
    selected_id = pick.split("  |  ")[0]
    row = filtered[filtered["platform_shop_id"] == selected_id].iloc[0]

    st.divider()

    # ── 결과 카드 (큼직하게 4개) ────────────────────────────
    h1, h2, h3, h4 = st.columns(4)
    review_n = int(row.get("monthly_review_count", 0))
    low_sample = review_n < 5
    name_help = f"매장 ID: {row['platform_shop_id']}"
    if low_sample:
        name_help += f"  |  ⚠ 리뷰 {review_n}건 — 표본 부족, 신뢰도 낮음"

    h1.metric("매장명", row.get("shop_name", "—"), help=name_help)
    h2.metric("그로몽 스코어", f"{row['gromong_score']:.1f}")
    h3.metric("등급", str(row["grade"]))
    h4.metric("성장 확률", f"{row['growth_probability'] * 100:.1f}%")

    if low_sample:
        st.warning(
            f"이 매장은 최근 월 리뷰 {review_n}건으로 표본이 적어 평점/응답률 신뢰도가 낮습니다. "
            f"스코어는 글로벌 평균으로 회귀(Bayesian shrinkage) 적용 후 산출되었습니다."
        )

    # ── 차트 + 표 ───────────────────────────────────────────
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**4개 지수 레이더**")
        st.plotly_chart(render_radar(row), use_container_width=True)
    with right:
        st.markdown("**4개 지수 점수**")
        idx_df = render_index_table(row)
        st.dataframe(
            idx_df[["지수", "점수"]],
            hide_index=True, use_container_width=True,
            column_config={
                "점수": st.column_config.ProgressColumn(
                    "점수", min_value=0, max_value=100, format="%.1f",
                ),
            },
        )

        # 현황 요약
        st.markdown("**현황 요약**")
        reply_rate = row.get("any_reply_rate", row.get("owner_reply_rate", 0))
        cluster_text = "—"
        if "cluster_id" in row and not pd.isna(row["cluster_id"]):
            cid = int(row["cluster_id"])
            outlier = " (이상치)" if row.get("is_outlier", 0) == 1 else ""
            cluster_text = f"Cluster {cid}{outlier}"
        st.dataframe(
            pd.DataFrame({
                "항목": ["평균 별점", "리뷰 응답률", "부정 리뷰 비율",
                       "월 평균 주문", "기본 가중치 점수", "클러스터"],
                "값": [
                    f"{row['avg_rating']:.2f} / 5.0",
                    f"{reply_rate * 100:.1f}%",
                    f"{row.get('negative_review_ratio', 0) * 100:.1f}%",
                    f"{row['monthly_order_count']:.0f} 건",
                    f"{row['gromong_score_default']:.1f} (Δ {row['gromong_score'] - row['gromong_score_default']:+.1f})",
                    cluster_text,
                ],
            }),
            hide_index=True, use_container_width=True,
        )

    # ── 3-class 확률 ────────────────────────────────────────
    if "prob_1" in row and not pd.isna(row.get("prob_1", float("nan"))):
        st.markdown("**3-class 분류 확률**")
        m1, m2, m3 = st.columns(3)
        m1.metric("성장 (+10% 이상)", f"{row['prob_1'] * 100:.1f}%")
        m2.metric("유지", f"{row['prob_0'] * 100:.1f}%")
        m3.metric("하락 (-10% 이하)", f"{row['prob_-1'] * 100:.1f}%")

    # ── AI 추천 액션 ────────────────────────────────────────
    st.markdown("**AI 추천 액션**")
    raw_actions = DemoCard().actions(row)
    actions = [strip_emoji(a) for a in raw_actions]
    for a in actions:
        st.info(a)


# ── 탭 2: 포트폴리오 ────────────────────────────────────────
def render_tab_portfolio(scores: pd.DataFrame, cluster_summary: pd.DataFrame) -> None:
    st.subheader("전체 포트폴리오 분석")

    a, b = st.columns([2, 1])
    with a:
        st.plotly_chart(render_score_histogram(scores), use_container_width=True)
    with b:
        st.markdown("**등급 별 평균**")
        agg = scores.groupby("grade").agg(
            매장수=("platform_shop_id", "size"),
            평균스코어=("gromong_score", "mean"),
            평균주문=("monthly_order_count", "mean"),
            평균별점=("avg_rating", "mean"),
        ).round(2).reindex(GRADE_ORDER)
        st.dataframe(agg, use_container_width=True)

    scatter = render_cluster_scatter(scores)
    if scatter is not None:
        st.plotly_chart(scatter, use_container_width=True)

    if not cluster_summary.empty:
        st.markdown("**클러스터별 평균 (KMeans k=5)**")
        st.dataframe(
            cluster_summary.rename(columns={
                "cluster_id": "ID", "count": "매장수",
                "monthly_order_count": "월주문",
                "monthly_sales": "월매출",
                "avg_rating": "별점",
                "negative_review_ratio": "부정비율",
                "any_reply_rate": "응답률",
                "menu_count": "메뉴수",
                "active_menu_ratio": "활성메뉴",
                "order_cv": "주문변동",
            }),
            hide_index=True, use_container_width=True,
        )

    st.markdown("**Top 30 매장**")
    cols = [
        "platform_shop_id", "shop_name", "grade", "gromong_score",
        "growth_probability", "monthly_order_count", "avg_rating",
    ]
    available = [c for c in cols if c in scores.columns]
    top = scores.sort_values("gromong_score", ascending=False).head(30)
    st.dataframe(
        top[available].rename(columns={
            "platform_shop_id": "매장ID", "shop_name": "매장명", "grade": "등급",
            "gromong_score": "스코어", "growth_probability": "성장확률",
            "monthly_order_count": "월주문", "avg_rating": "별점",
        }),
        use_container_width=True, hide_index=True,
    )


# ── 탭 3: 모델 성능 ─────────────────────────────────────────
def render_tab_model(
    shap_w: dict, shap_imp: pd.DataFrame, model_cmp: pd.DataFrame,
    threshold: dict, calibration: dict, multiclass: dict, keywords: pd.DataFrame,
) -> None:
    st.subheader("모델 성능 및 해석")

    # 알고리즘 비교
    if not model_cmp.empty:
        st.markdown("**알고리즘 비교 (5-fold TimeSeriesSplit CV)**")
        df = model_cmp.copy()
        for c in ["auc_mean", "pr_auc_mean", "f1_mean", "log_loss_mean", "brier_mean"]:
            df[c] = df[c].round(4)
        df = df.rename(columns={
            "name": "모델", "auc_mean": "AUC", "auc_std": "AUC_std",
            "pr_auc_mean": "PR-AUC", "f1_mean": "F1",
            "log_loss_mean": "LogLoss", "brier_mean": "Brier",
            "fit_seconds": "Fit(s)",
        })
        st.dataframe(
            df[["모델", "AUC", "PR-AUC", "F1", "LogLoss", "Brier", "Fit(s)"]],
            hide_index=True, use_container_width=True,
        )

    a, b = st.columns(2)
    with a:
        if threshold:
            st.markdown("**임계값 분석**")
            t1, t2 = st.columns(2)
            t1.metric("F1-optimal", f"{threshold['f1_optimal']:.3f}")
            t2.metric("Cost-optimal (FP=5,FN=1)", f"{threshold['cost_optimal']:.3f}")
            f1_def = threshold["f1_at_default"]
            f1_opt = threshold["f1_at_f1_optimal"]
            st.metric(
                "F1 (default 0.5 → optimal)",
                f"{f1_def:.3f} → {f1_opt:.3f}",
                delta=f"{f1_opt - f1_def:+.3f}",
            )

    with b:
        if calibration:
            st.markdown("**확률 보정 (Isotonic)**")
            bb, ba = calibration["brier_before"], calibration["brier_after"]
            lb, la = calibration["log_loss_before"], calibration["log_loss_after"]
            st.metric("Brier", f"{bb:.4f} → {ba:.4f}",
                      delta=f"{ba - bb:+.4f}", delta_color="inverse")
            st.metric("LogLoss", f"{lb:.4f} → {la:.4f}",
                      delta=f"{la - lb:+.4f}", delta_color="inverse")
            st.caption("작을수록 좋음 (확률이 더 신뢰 가능)")

    if multiclass:
        st.markdown("**3-class 분류 성능 (성장/유지/하락)**")
        m1, m2 = st.columns(2)
        m1.metric("Macro F1", f"{multiclass['macro_f1']:.4f}")
        m2.metric("Weighted F1", f"{multiclass['weighted_f1']:.4f}")
        with st.expander("Classification Report"):
            st.code(multiclass.get("report", ""), language="text")

    st.divider()
    st.markdown("**SHAP 가중치 (4지수)**")
    if shap_w:
        st.plotly_chart(render_weight_comparison(shap_w), use_container_width=True)

    if not shap_imp.empty:
        st.markdown("**SHAP 피처 중요도 Top 15**")
        top = shap_imp.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top["mean_abs_shap"], y=top["feature"], orientation="h",
            marker_color="steelblue",
        ))
        fig.update_layout(
            height=500, margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Mean |SHAP|",
        )
        st.plotly_chart(fig, use_container_width=True)

    if not keywords.empty:
        st.markdown("**리뷰 키워드 변별력 Top 20**")
        top_kw = keywords.head(20)
        st.dataframe(
            top_kw[["keyword", "polarity", "neg_count", "pos_count", "discrimination"]]
            .rename(columns={
                "keyword": "키워드", "polarity": "극성",
                "neg_count": "부정등장", "pos_count": "긍정등장",
                "discrimination": "변별력",
            }),
            hide_index=True, use_container_width=True,
        )


# ── 탭 4: 인과추론 ─────────────────────────────────────────
def render_tab_causal(did: dict, output_dir: Path) -> None:
    st.subheader("인과추론 — 댓글몽 도입 효과 검증")

    if not did:
        st.info("인과추론 결과가 없습니다.")
        return

    a, b, c = st.columns(3)
    a.metric(
        "TWFE DID ATT",
        f"+{did.get('att', 0):.2f} 건/월",
        help=f"p={did.get('p', 0):.2e}",
    )
    a.caption(f"95% CI [{did.get('ci_lo', 0):.2f}, {did.get('ci_hi', 0):.2f}]")

    if did.get("att_psm") is not None:
        b.metric(
            "PSM-DID ATT",
            f"+{did['att_psm']:.2f} 건/월",
            help=f"p={did.get('p_psm', 0):.2e}",
        )
        b.caption(f"매칭 {did.get('matched_n', 0)}쌍")

    pt = did.get("parallel_trends_p")
    if pt is not None:
        c.metric(
            "Parallel Trends p",
            f"{pt:.4f}",
            delta="충족" if pt > 0.05 else "위배",
            delta_color="normal" if pt > 0.05 else "inverse",
        )
        c.caption("p > 0.05 시 DID 가정 충족")

    st.divider()

    img1, img2 = st.columns(2)
    es_path = output_dir / "event_study_plot.png"
    if es_path.exists():
        img1.markdown("**Event Study Plot**")
        img1.image(str(es_path), use_container_width=True)
        img1.caption("처치 전(t<0) 계수 ≈ 0 → Parallel Trends 가정 검증")

    psm_path = output_dir / "psm_propensity_dist.png"
    if psm_path.exists():
        img2.markdown("**PSM 성향점수 분포**")
        img2.image(str(psm_path), use_container_width=True)
        img2.caption("매칭 전 실험군/통제군 ps 분포 — 겹침이 클수록 매칭 가능")

    trend_path = output_dir / "eda_monthly_trend.png"
    if trend_path.exists():
        st.markdown("**월별 평균 주문 추세 (실험군 vs 통제군)**")
        st.image(str(trend_path), use_container_width=True)


# ── main ─────────────────────────────────────────────────────
def main() -> None:
    paths = Paths()
    scores = load_scores(str(paths.output_dir))

    if scores.empty:
        st.error(
            "스코어 데이터가 없습니다. 먼저 터미널에서 다음 명령을 실행하세요:\n\n"
            "```\npython -m src\n```"
        )
        st.stop()

    shap_w = load_shap_weights(str(paths.output_dir))
    shap_imp = load_shap_importance(str(paths.output_dir))
    did_summary = load_did_summary(str(paths.output_dir))
    model_cmp = load_model_comparison(str(paths.output_dir))
    threshold = load_threshold(str(paths.output_dir))
    calibration = load_calibration(str(paths.output_dir))
    multiclass = load_multiclass(str(paths.output_dir))
    cluster_summary = load_clusters(str(paths.output_dir))
    keywords = load_keywords(str(paths.output_dir))
    model_extra = load_model_dataset(str(paths.output_dir))

    if not model_extra.empty:
        scores = scores.merge(
            model_extra.drop(columns=["year_month_dt"]),
            on="platform_shop_id", how="left",
        )

    render_sidebar(scores, did_summary)

    tab1, tab2, tab3, tab4 = st.tabs([
        "매장 분석", "포트폴리오", "모델 성능", "인과추론",
    ])
    with tab1:
        render_tab_shop(scores)
    with tab2:
        render_tab_portfolio(scores, cluster_summary)
    with tab3:
        render_tab_model(
            shap_w, shap_imp, model_cmp, threshold,
            calibration, multiclass, keywords,
        )
    with tab4:
        render_tab_causal(did_summary, paths.output_dir)


if __name__ == "__main__":
    main()
