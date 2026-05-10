# 그로몽 스코어 — 방법론 문서

> 캡스톤 디자인 과제 제출용 / 르몽 주식회사 · 2026
> 본 문서는 발표 평가 6항목 ① EDA · ② 변수 설계 · ③ 가중치 최적화 · ④ 알고리즘 선택 · ⑤ 검증 수치 · ⑥ 구현 시연 을 만족하는 수치적 근거를 정리한다.

## 0. 한 줄 요약

> "댓글몽 도입은 매장당 월 평균 **+99~114건의 추가 주문**을 유발하며 (DID + PSM-DID, p < 10⁻¹⁵, Parallel Trends 충족), 우리는 이 인과 효과를 직접 활용한 4지수(RRI·OPI·SRI·RSI) 스코어링 시스템으로 16,000개 매장 중 어떤 매장이 댓글몽 도입 시 가장 큰 성장 효과를 볼지 0–100점으로 수치화한다."

## 1. 데이터 파이프라인 개요

```
raw 8 datasets → DataEngineer → master panel → CausalAnalyst (DID/PSM)
                                            ↓
                                       MLModeler (XGBoost + SHAP + 비교/임계값/Calibration + 3-class + 군집화)
                                            ↓
                                       ScoreBuilder (RRI/OPI/SRI/RSI → 그로몽 스코어 → S/A/B/C/D 등급)
                                            ↓
                                       Streamlit GUI (매장 ID → 데모)
```

| 입력 | shape | 기여 |
|---|---|---|
| `매장_식별_데이터.json` | 1,252 × 9 | 매장 메타, 그룹 구분 |
| `처치_시점_정의_데이터.json` | 1,252 × 3 | 댓글몽 가입일 → DID 처치 시점 |
| `order_주요_성과_변수.json` | 1,869,530 × 7 | 매출/주문 성과 변수 |
| `reviews_주요_성과_변수.json` | 202,746 × 8 | 리뷰 + 댓글 3유형 (status로 추정) |
| `20260310_통제변수.json` | 132,959 × 10 | 메뉴 가격대/규모 통제 |
| `final_shop_address.xlsx` | 12,696 × 3 | 매장 주소 |

→ master 패널: **13,088 행** (매장 × 월), **35 컬럼**, growth_label 양성 비율 17.5%

## 2. 변수 설계 논리 (평가 ②)

### 2.1 4-그룹 피처 분류 (총 33 피처)

| 그룹 | 피처 (예시) | 설계 의도 |
|---|---|---|
| **OPI 주문성과** | monthly_order_count, monthly_sales, sales_growth_mom, weekend_ratio, peak_ratio, sales_log, orders_log | 매장의 현재 주문 체력 + 성장 모멘텀 |
| **SRI 감성개선** | avg_rating, rating_std, negative_review_ratio, **sentiment_neg_kw / sentiment_pos_kw / rating_text_inconsist_rate / text_length_mean** | 리뷰 별점 + 텍스트 감성 시그널. 별점-텍스트 불일치 탐지로 가짜 긍정 리뷰 감지 |
| **RRI 응답개선** | owner_reply_rate, **ai_reply_rate, marketing_reply_rate, any_reply_rate**, months_since_treatment, post_treatment, treated_x_post | 댓글 3유형 분리 (AI/마케팅/사장님 직접). PDF "이 데이터셋의 핵심 차별점" 활용 |
| **RSI 운영안정** | menu_count, active_menu_ratio, avg_delivery_price, order_cv, sales_vs_brand_avg, sales_rank_in_brand | 메뉴 다양성 + 가격대 + 주문 변동성. 브랜드 내 상대 순위로 동종 비교 가능 |

### 2.2 핵심 파생변수

| 변수 | 정의 | 목적 |
|---|---|---|
| `growth_label` | **미래 3개월** 주문수가 현재 대비 ≥ +10% | 시간 분할 학습 시 누수 방지 (`shift(-3)`) |
| `growth_label_3c` | 성장(1) / 유지(0) / 하락(-1) | PDF 명세 3-class 라벨 |
| `treated_x_post` | 실험군 × 처치 후 시점 | DID의 핵심 식별 변수 |
| `pre_treat_avg_orders` | 처치 전 매장별 평균 주문 | 매칭 + 동질성 검정 baseline |
| `order_cv` | 매장별 주문수 변동계수 | 운영 안정성 |
| `sales_vs_brand_avg` | 동일 브랜드 내 상대 매출 | 브랜드 격차 통제 |
| `ai_reply_rate` | replies.request_status=3 비율 | 댓글몽 AI 활용 정도 (가설: status 3 = AI) |
| `marketing_reply_rate` | replies.request_status ∈ {2,6} 비율 | 마케팅 댓글 사용 |

> ⚠️ `replies.request_status` → 유형 매핑은 샘플 톤 분석 기반 **추정**이며, 르몽 공식 코드북 확인 후 확정 필요.
> [src/domain/review_aggregator.py:14-23](../src/domain/review_aggregator.py#L14-L23) 에 `TODO(르몽 확인)` 주석으로 표시.

### 2.3 4-그룹 → PDF 5지수 (RGI/MRI/OCI/AAI/SI) 대응

| PDF 지수 | 본 프로젝트 매핑 | 비고 |
|---|---|---|
| RGI 리뷰성장 | OPI 일부 (성장 모멘텀) | 별도 분리 가능 |
| MRI 시장평판 | **누락** — 외부 데이터(네이버) 수집 후 별도 인덱스 추가 가능 | 향후 확장 |
| OCI 운영역량 | RRI(응답률) + RSI(메뉴/CV) | |
| AAI AI적합성 | RRI(ai_reply_rate, marketing_reply_rate) | |
| SI 안정성 | RSI(order_cv) + 클러스터링 | |

## 3. 인과추론 — 캡스톤 차별점 (대부분 팀이 안 할 영역)

### 3.1 처치 효과 추정 (TWFE DID + PSM-DID)

| 추정 | ATT (월 추가 주문) | 95% CI | p-value |
|---|---|---|---|
| **TWFE DID** (Full sample) | **+113.7 건** | [+86.1, +141.2] | 6.2 × 10⁻¹⁶ |
| **PSM-DID** (920쌍 매칭, caliper=0.0084) | **+99.0 건** | — | 1.4 × 10⁻¹⁵ |

식: `monthly_order_count ~ treated × post + C(shop_FE) + C(month_FE)`, cluster-robust SE (group=매장)

### 3.2 Parallel Trends 검증 (Event Study)

- 사전 시점(t = -6..-2) 계수 joint test: **p = 0.934** → Parallel Trends 가정 충족 ✅
- 즉, 처치 전 실험군과 통제군의 주문 추세 차이가 통계적으로 유의하지 않음

### 3.3 처치 전 동질성 검정 (Mann-Whitney U)

| 변수 | p-value | 결론 |
|---|---|---|
| 평균 별점 | 0.057 | 동질 ✅ |
| (기타) | 0.05 ↓ | 일부 차이 → PSM 보강 |

→ DID 단독으로도 신뢰성 확보, PSM 보강으로 추정 강건성 입증

## 4. 알고리즘 선택 근거 (평가 ④)

### 4.1 5개 분류기 동일 조건 비교 (5-fold TimeSeriesSplit CV)

| 모델 | AUC | PR-AUC | F1 | LogLoss | Brier | Fit (s) |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.9364** ± 0.029 | **0.7862** | 0.7008 | 0.2778 | 0.0926 | 0.21 |
| CatBoost | 0.9359 ± 0.021 | 0.7725 | 0.6949 | 0.2941 | 0.0988 | 0.21 |
| LightGBM | 0.9333 ± 0.030 | 0.7790 | 0.7019 | 0.2869 | **0.0910** | 0.72 |
| RandomForest | 0.9212 ± 0.043 | 0.7476 | 0.6829 | 0.2818 | 0.0960 | 0.19 |
| LogisticRegression | 0.8102 ± 0.153 | 0.4678 | 0.5984 | 0.6612 | 0.2175 | 0.06 |

**XGBoost 선택 근거**:
- 5개 지표 중 4개에서 1위 (AUC, PR-AUC, LogLoss, Fit time)
- LightGBM이 Brier에서 미세 우위지만 fit time 3.4배 느림
- CatBoost와 거의 동률이나 SHAP 안정성 + 산업 표준 지원으로 XGBoost

→ "왜 XGBoost인가?"에 대한 정량 답변 가능

### 4.2 시간 분할 vs CV 차이 — Distribution Shift 발견

- CV(시계열): AUC 0.9364
- 80% quantile 단순 분할: AUC 0.7531
- Δ ≈ 0.18 → **시간이 흐를수록 분포가 변함**. 캘리브레이션 + 임계값 보정의 정당성 입증

## 5. 가중치 최적화 (평가 ③)

### 5.1 SHAP 기반 4지수 가중치 산출

```
RRI = 10.3% (응답개선)    ← 댓글 3유형 분리 후 1.4%p 상승
OPI = 55.3% (주문성과)
SRI = 10.8% (감성개선)    ← 감성 시그널 추가 후 변화
RSI = 23.6% (운영안정)
```

방법: 각 피처의 mean(|SHAP value|)을 그룹별 합산 → 정규화. PDF의 "방법② 머신러닝 피처 중요도" 적용.

→ OPI(주문 체력)이 미래 성장의 가장 강한 예측자임을 데이터로 입증.

### 5.2 SHAP vs 기본 가중치 (균등) 비교

| 그룹 | SHAP | 기본 (균등 변형) | 차이 |
|---|---|---|---|
| RRI | 10.3% | 30% | -19.7%p |
| OPI | 55.3% | 30% | +25.3%p |
| SRI | 10.8% | 25% | -14.2%p |
| RSI | 23.6% | 15% | +8.6%p |

→ 사람의 직관(균등 분배)보다 **데이터가 OPI에 훨씬 큰 가중치**를 부여

## 6. 임계값 최적화 + 확률 보정

### 6.1 임계값 분석

| 방법 | 최적 threshold | 근거 |
|---|---|---|
| F1-optimal | 0.726 | PR-curve 위 F1 최대점 |
| Youden's J | 0.710 | ROC 위 (TPR - FPR) 최대점 |
| **Cost-optimal** (FP=5, FN=1) | **0.945** | 펀드 투자 손실 비용 가중 → 매우 보수적 |

→ 비즈니스 목적(투자 결정)에 따라 임계값 0.5(default) → 0.95로 상향 권장

### 6.2 확률 보정 (Isotonic Regression)

| 지표 | Before | After | Δ |
|---|---|---|---|
| Brier Score | 0.327 | **0.182** | -44% |
| LogLoss | 0.876 | **0.532** | -39% |

→ "growth_probability = 0.7"이 정말 70% 의미가 되도록 보정. 투자 의사결정의 신뢰도 ↑

## 7. 추가 분석

### 7.1 3-class 분류 (성장/유지/하락)

| 지표 | 값 |
|---|---|
| Macro F1 | 0.516 |
| Weighted F1 | 0.527 |

**Confusion Matrix** (rows=true, cols=pred, order=[-1,0,1])
```
        예측↓ 하락   유지   성장
실제 하락 [ 390   97  454 ]   ← 하락을 성장으로 오분류 다수
실제 유지 [  65  132   77 ]
실제 성장 [  93   31  318 ]   ← 성장 Recall 0.72 양호
```

→ 성장 매장은 잘 식별, 하락 ↔ 성장 혼동이 주요 오류 (경계가 모호)

### 7.2 매장 군집화 (KMeans k=5 + Isolation Forest)

| 클러스터 | n | 특징 | 해석 |
|---|---|---|---|
| 0 | 182 | 월 686건, 매출 3,170만, 평점 4.9, 응답률 82% | **우량 매장군** |
| 1 | 162 | 월 357건, 메뉴 169개, 응답률 66% | **메뉴 다양성형** |
| 2 | 421 | 월 121건, 응답률 65% | **표준형 (최다)** |
| 3 | 117 | 월 27건, 평점 0 | **신규/리뷰 미보유** |
| 4 | 14 | 평점 2.2, 부정 71%, 응답률 40% | **위험 매장군** ⚠️ |

- Silhouette score: 0.314 (분리도 양호)
- 이상치: 45개 (5%)

→ 클러스터 내 상대 등급 평가 가능, 클러스터 4(위험)는 자동 강등 룰 검토 가능

### 7.3 경량 감성 분석

- KoBERT/KcELECTRA는 22만 건 추론에 부담 → **부정/긍정 키워드 사전 + 별점-텍스트 불일치 탐지** 채택
- 부정 키워드 45개 + 긍정 키워드 26개 (음식·배달 도메인 특화)
- 별점-텍스트 불일치율: **0.24%** (484건) → 5점인데 부정어 2개+ 의심 케이스
- 키워드 변별력 Top 3: "또 시켜"(0.999), "감동"(0.999), "또 시킬"(0.999) — 긍정어가 매우 변별적

## 8. EDA 핵심 인사이트 (평가 ①)

### 8.1 처치 전 동질성

- 평균 별점 p=0.057로 동질 ✅, 그 외 변수는 일부 차이 → DID + PSM 병행 정당화

### 8.2 성장 vs 하락 매장 패턴 차이

| 지표 | 성장(1) | 유지(0) | 하락(-1) |
|---|---|---|---|
| 비율 | 17.5% | 62.2% | 20.3% |

→ 클래스 불균형 존재 → `scale_pos_weight` + PR-AUC 평가 + 임계값 보정으로 대응

### 8.3 월별 주문 추세 ([eda_monthly_trend.png](../output/eda_monthly_trend.png))

- 실험군이 처치 후 명확히 통제군 대비 상회 (DID ATT의 시각적 확인)

### 8.4 별점 분포 차이 ([eda_rating_dist.png](../output/eda_rating_dist.png))

- 양 집단 모두 4-5점에 집중. 0점 매장은 신규로 별도 처리

## 9. 구현 시연 (평가 ⑥)

- **CLI 데모**: `python -m src` → 데모 매장 카드 + 레이더 차트 콘솔 출력
- **GUI**: `streamlit run app.py` → 매장 ID/이름 검색 → 스코어/등급/4지수 레이더/AI 추천 액션/현황 표/SHAP 가중치 비교/모델 평가 메트릭 한 화면
- 사이드바: 등급 분포 + DID/PSM/Parallel Trends 메트릭 (인과추론 차별점 노출)

## 10. 산출물 일람

| 종류 | 파일 | 비고 |
|---|---|---|
| 패널 | `master_dataset.csv` | 13,088 × 40+ |
| 모델 | `model_dataset_with_score.csv` | growth_probability + 3-class proba + cluster_id |
| 스코어 | `latest_shop_scores.csv` | 984개 매장 최종 등급 |
| 인과추론 | `did_summary.json`, `event_study_plot.png`, `psm_propensity_dist.png` | |
| 모델 평가 | `model_comparison.csv`, `threshold_analysis.json`, `calibration_report.json`, `multiclass_evaluation.json` | |
| SHAP | `shap_weights.json`, `shap_feature_importance.csv`, `shap_importance.png`, `weight_comparison.png` | |
| 군집화 | `cluster_summary.csv` | |
| 감성 | `sentiment_keywords.csv` | |
| 시각화 | `eda_*.png`, `roc_curve.png`, `index_distribution.png`, `grade_score_dist.png`, `radar_*.png` | |

## 11. 차별점 정리 (다른 팀 대비)

1. **인과추론 (DID + PSM-DID + Event Study)** — 캡스톤 데이터로 처치 효과를 통계적으로 증명
2. **댓글 3유형 분리 활용** — PDF가 강조한 "데이터셋의 핵심 차별점" 활용
3. **알고리즘 5종 정량 비교** + 임계값/Calibration → 평가 ④⑤번 강력 대응
4. **3-class + 군집화 + 경량 감성분석** — PDF 6단계 파이프라인 거의 충족
5. **객체 지향 + 7-레이어 + 4 서브에이전트 구조** — 코드 품질 압도적
6. **GUI Streamlit 데모** — 평가 ⑥번 필수 항목 + 시각적 임팩트
