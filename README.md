# 그로몽 스코어 — AI 기반 성장 유망 매장 스코어링

> 르몽 캡스톤 디자인 과제 · 2026 / **F&B 마이크로펀드 × AI 알고리즘 연구개발**

식당판 신용점수처럼 매장의 성장 가능성을 **0–100점**으로 수치화한다. 댓글몽 도입 효과를 인과추론으로 증명하고, ML로 어떤 매장이 도입 시 가장 큰 성장을 볼지 예측한다.

---

## 핵심 결과

| 결과 | 수치 |
|---|---|
| **DID 처치 효과** | ATT = +113.7건/월, 95% CI [86.1, 141.2], p < 10⁻¹⁵ |
| **Parallel Trends** | p = 0.934 → 충족 ✅ |
| **PSM-DID 일관성** | ATT = +99.0건/월, 920쌍 매칭 → Full-sample과 일관 ✅ |
| **XGBoost AUC (5-fold CV)** | **0.9364** (5개 모델 중 1위) |
| **PR-AUC** | 0.7862 |
| **Brier (Calibration)** | 0.327 → **0.182** (-44%) |
| **3-class Macro F1** | 0.516 |
| **스코어링 매장** | 984개 / S=7, A=140, B=327, C=431, D=79 |

---

## 빠른 시작

```bash
# 0) 클론
git clone https://github.com/<your-username>/shop-credit-score.git
cd shop-credit-score

# 르몽 raw 데이터 8종을 data/ 폴더에 배치 (별도 제공)
mkdir -p data
# data/매장_식별_데이터.json, data/처치_시점_정의_데이터.json, ...

# 1) 가상환경 + 의존성
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# macOS XGBoost용
brew install libomp

# 2) 전체 파이프라인 (4–8분 소요)
python -m src

# 3) GUI 데모
streamlit run app.py

# 4) 외부 데이터 스크래핑 데모 (mock)
python -m scripts.scrape_external --limit 50
```

---

## 프로젝트 구조

```
lemong/
├── data/                  # 8 raw 입력 데이터셋
├── output/                # 17+ 산출물 (CSV/PNG/JSON)
├── docs/
│   └── METHODOLOGY.md     # 발표용 종합 방법론 문서
├── app.py                 # Streamlit GUI 진입점
├── scripts/
│   └── scrape_external.py # 외부 데이터 수집 데모
├── .claude/agents/        # 4개 서브에이전트 (data/causal/ml/score)
└── src/
    ├── __main__.py        # python -m src 진입점
    ├── config.py          # 모든 상수/하이퍼파라미터
    ├── utils.py
    ├── io/                # JsonLoader, ExcelLoader, ColumnCleaner
    ├── domain/            # 도메인 변환 (집계/병합/라벨/감성)
    ├── analysis/          # 인과추론 (DID/PSM/EventStudy/Homogeneity)
    ├── modeling/          # 학습 (XGBoost/SHAP/Comparator/Calibrator/Multiclass/Clusterer)
    ├── scoring/           # 지수/스코어/등급 (Strategy 패턴)
    ├── viz/               # 시각화 + 콘솔 카드
    ├── services/          # 4개 얇은 오케스트레이터
    └── external/          # 외부 데이터 (네이버 mock)
```

---

## 다른 팀 대비 차별점

1. **인과추론 (DID + PSM-DID + Event Study)** — 처치 효과를 통계적으로 증명. 대부분 팀은 ML만 함
2. **댓글 3유형 분리 활용** — `replies.request_status` 추정 매핑으로 PDF의 "데이터셋의 핵심 차별점" 활용
3. **5종 알고리즘 정량 비교** — XGBoost vs LightGBM vs CatBoost vs RF vs LogReg, 5-fold CV
4. **임계값 최적화 + 확률 보정** — F1/Youden/Cost-based + Isotonic Calibration (Brier -44%)
5. **3-class 분류 + 매장 군집화 + 경량 감성분석** — PDF 6단계 파이프라인 거의 완전 충족
6. **객체 지향 7-레이어 + 4 서브에이전트** — 코드 품질 압도적
7. **Streamlit GUI** — 평가 필수 항목 + 시각적 임팩트

자세한 방법론은 [docs/METHODOLOGY.md](docs/METHODOLOGY.md) 참조.

---

## 4개 서브에이전트 (Claude Code 자동 라우팅)

| 에이전트 | 트리거 | 책임 |
|---|---|---|
| [data-engineer](.claude/agents/data-engineer.md) | "전처리", "master 갱신" | raw → master 패널 |
| [causal-analyst](.claude/agents/causal-analyst.md) | "DID", "PSM", "Event Study" | 인과추론 + EDA |
| [ml-modeler](.claude/agents/ml-modeler.md) | "XGBoost", "SHAP", "지수 가중치" | 학습 + 가중치 산출 |
| [score-builder](.claude/agents/score-builder.md) | "스코어링", "등급", "RRI/OPI/SRI/RSI" | 4지수 + 그로몽 스코어 |

---

## 주요 산출물

| 파일 | 내용 |
|---|---|
| `output/master_dataset.csv` | 13,088 × 40+ 매장×월 패널 |
| `output/latest_shop_scores.csv` | 984개 매장 최종 스코어 + 등급 |
| `output/did_summary.json` | ATT, Parallel Trends, PSM 결과 |
| `output/model_comparison.csv` | 5개 알고리즘 5-fold CV 비교 |
| `output/threshold_analysis.json` | F1/Youden/Cost-optimal 임계값 |
| `output/calibration_report.json` | Brier/LogLoss before/after |
| `output/multiclass_evaluation.json` | 3-class 분류 결과 |
| `output/cluster_summary.csv` | 5개 클러스터 평균 통계 |
| `output/sentiment_keywords.csv` | 부정/긍정 키워드 변별력 |
| `output/shap_weights.json` | 4지수 SHAP 가중치 |
| `output/*.png` | EDA + Event Study + ROC + SHAP + 등급/지수 분포 + 레이더 |

---

## 알려진 한계 / TODO

- `replies.request_status` → 댓글 유형 매핑은 **추정**. 르몽 공식 코드북 확인 후 확정 필요 ([src/domain/review_aggregator.py:14](src/domain/review_aggregator.py#L14))
- 유저 운영 행동 데이터(⑥) / 탈퇴 유저(⑦) / 월별 가입자(⑧)는 미보유 — 르몽 추가 제공 시 통합 가능
- `external/naver_place.py` 는 현재 mock 모드. 실 호출은 client_id/secret 발급 후 `_real()` 구현 필요
- KoBERT 같은 대형 LM은 22만 건 추론 부담으로 미사용. 키워드 사전 + 별점-텍스트 불일치로 대체

---

🍋 **"데이터로 증명하고, 알고리즘으로 투자한다"**
