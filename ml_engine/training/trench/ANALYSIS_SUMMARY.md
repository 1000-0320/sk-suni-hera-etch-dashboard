# Trench Etch 공정 분석 요약

팀원 질문(데이터 규모 / 파라미터별 모델 선택 / 품질 점수 공식 / Defect 분석) 답변용 문서.
모든 수치는 `ml_engine/trench_models/metadata.json`(학습 결과 메타데이터)과
`ml_engine/trench_core.py`(배포 코드) 원문에서 그대로 가져왔습니다 — 코드 확인하고 싶으면
같이 보면 됩니다.

---

## 1. 어떤 데이터를, 얼마나 분석했나

원본 파일: `trench recipe full dataset.xlsx`, `trench recipe master.csv`
(레포엔 `data/trench_dataset.xlsx`로 포함, recipe master csv는 원본 별도 보관 — 내용은 xlsx에 이미 포함되어 있어 중복 업로드 안 함)

| 항목 | 값 |
|---|---|
| Recipe Version 개수 | 16개 (`Base`, `Rev1`~`Rev15`) |
| Wafer 개수 (Wafer_Summary) | 236장 |
| Site 측정값 개수 (Site_Level_Raw) | 5,900행 (= 236장 × Wafer당 25개 Site, 정확히 균등) |
| Wafer당 Site 좌표 구성 | Center 1개, Mid 4개, Edge 10개, Extreme Edge 10개 = 25개 (`site_template`) |
| Equipment/Chamber 관측 조합 | 4개 — `AMAT-CENTRIS/CH-C`(63장), `AMAT-CENTRIS/CH-D`(64장), `LAM-2300FLEX/CH-A`(54장), `LAM-2300FLEX/CH-B`(55장) |
| Recipe × Equipment × Chamber 조합별 Wafer 수 | `metadata.json`의 `tool_support.recipe_pair_wafer_counts`에 16개 Recipe × 4개 Tool 조합 전부 기록됨 (예: `Rev15`+`AMAT-CENTRIS`+`CH-C` = 7장) |

**중요한 제약**: 관측된 Recipe는 16개뿐이고, Rev들이 서로 독립적인 실험이 아니라 Base에서 시작해 파라미터를
한두 개씩 순차적으로 바꿔나간 이력(`Changed_Params_This_Rev` 컬럼)이라 **파라미터 간 상관관계가 높음** —
그래서 "이 파라미터 하나가 원인이다"라고 인과적으로 말할 수 없고, 어디까지나 모델이 잡아낸 연관성(association)임.
이 점은 `metadata.json`의 `limitations`에도 명시돼 있음.

---

## 2. 모델 학습 — 파라미터, 항목별 모델 선택, 선택 로직

### 입력 파라미터 (20개, Stage 4개)

| Stage | 파라미터 |
|---|---|
| S1 | Time(s), CF4(sccm), CHF3(sccm), RF Bias(W), Pressure(mT) |
| S2 | Time(s), O2(sccm), N2(sccm), RF Bias(W), Pressure(mT) |
| S3 | Time(s), CF4(sccm), C4F8(sccm), RF Bias(W), Pressure(mT) |
| S4 | Time(s), HBr(sccm), Cl2(sccm), RF Bias(W), Pressure(mT) |

실제 모델 입력 Feature는 여기에 Site 위치 정보(`Radius_frac`, `Angle_sin`, `Angle_cos`)와
범주형 `Equipment_Model`, `Chamber_ID`, `Zone`을 더해 **총 25개 Feature**(`metadata.json`의 `feature_columns`).
즉 모델은 "이 Recipe로, 이 장비/챔버에서, Wafer의 이 위치를 찍으면 값이 어떻게 나올까"를 Site 단위로 예측함.

### 항목별로 학습한 것 (모두 RandomForest / XGBoost / MLP 세 가지 후보로 학습)

| 항목 | 종류 | 배포 선택 모델 |
|---|---|---|
| Top/Mid/Bottom CD, Depth (nm) | 회귀 | Top CD→**XGBoost**, Mid/Bottom CD·Depth→**RandomForest** |
| Top/Mid/Bottom CD, Depth Spec Pass 여부 | 이진분류 | Top CD→**XGBoost**, Mid CD→**RandomForest**, Bottom CD·Depth→**MLP** |
| Particle_Defect (Site별 Defect 발생 여부) | 이진분류 | **MLP** |
| Defect_Count, Defect_Severity_Score | 회귀 | 둘 다 **MLP** |

(정확한 매핑은 `metadata.json`의 `selected_models`/`quality_v2.selected_models` 참고)

### 모델을 "어떻게" 골랐나 (선택 로직)

핵심은 **일반 교차검증이 아니라 "Recipe 통째로 빼고 검증"(whole-recipe-holdout)** 방식을 씀:

- `validate_unseen_recipes.py`에서 `GroupKFold(n_splits=4)`를 **Recipe_Version 단위**로 묶어서 실행
  (`ml_engine/training/trench/validate_unseen_recipes.py:31-32`). 즉 어떤 fold에서 검증할 때
  그 Recipe의 Wafer/Site는 학습에 아예 안 들어가게 만들어서, "새로운 Recipe(=시뮬레이터에서
  사용자가 아직 안 써본 조합)에 대해서도 잘 예측하는가"를 테스트함.
- 이거 말고 **Wafer 단위**로만 나누는 `GroupKFold`(같은 Recipe의 다른 Wafer는 학습에 남아있는 방식,
  `wafer_group_cv_selected_models`)도 별도로 돌렸는데, **배포에는 안 씀**. 이유:
  Wafer 단위 검증은 "이미 아는 Recipe의 새 Wafer" 정도만 맞히면 되니까 더 쉬워서 점수가 잘 나오는데,
  시뮬레이터 목적(새 Recipe 조합 예측)엔 안 맞음. 실제로 Wafer 단위 기준으론 MLP가 CD/Depth
  회귀에서 좋게 나왔지만, Recipe-holdout 기준으로 보면 성능이 무너져서 배포에서 제외함
  (`metadata.json`의 `deployment_selection_note`: *"MLP is retained for comparison but excluded
  from simulator deployment because recipe-holdout performance collapsed."*)
- 항목별 최종 선택 기준:
  - CD/Depth 회귀, Defect_Count/Severity 회귀 → **Recipe-holdout CV MAE(평균절대오차)가 가장 낮은 모델**
  - Spec Pass 이진분류 → **Recipe-holdout Balanced Accuracy가 가장 높은 모델**(동점이면 ROC-AUC로 재비교)
  - Particle_Defect 이진분류 → **Recipe-holdout Average Precision이 가장 높은 모델**(동점이면 Balanced Accuracy)

즉 "정확도만 높은 모델"이 아니라 "안 써본 Recipe 조합에서도 안 무너지는 모델"을 기준으로 골랐다는 게 포인트.

---

## 3. 품질 점수 — 공식, 지표별 가중치, 정규화 방식

**주의: 이 프로젝트엔 품질 점수가 두 가지 있고, 서로 다른 목적으로 씀.** 팀원한테 설명할 때 헷갈리기 쉬운
부분이라 명확히 구분해서 적음.

### (A) 대시보드에 표시되는 "목표 대비 종합 점수" — 사용자가 입력한 목표 기준 (`ml_engine/scoring.py`)

Output B(목표 대비 평가), Output C/D(최적 Recipe 추천/비교), Process Dashboard의 Rev별 스코어링
표에 실제로 보이는 점수. 사용자가 화면에서 입력한 "목표 스펙"에 얼마나 가까운지 채점함.

```
종합 점수 = Pass Rate 60% + Uniformity 25% + 목표 근접도 15%
```

- **Pass Rate (60%)**: 모델이 예측한 "Spec Pass 여부(0/1)" 4개 항목(Top/Mid/Bottom CD, Depth)을
  25개 Site 전체에 대해 평균낸 비율(`Overall Spec Pass Rate`, 0~100%). 그대로 0~100 클립.
- **Uniformity (25%)**: `CD Uniformity`, `Depth Uniformity`(각각 Site 간 표준편차/평균×100, 즉
  변동계수%) 두 값을 사용자가 지정한 "허용 최대치"(`max_cd_uniformity`, `max_depth_uniformity`)
  기준으로 선형 정규화 — 값이 0이면 100점, 허용 최대치 이상이면 0점. 두 점수를 평균.
- **목표 근접도 (15%)**: Top/Mid/Bottom CD, Depth 4개 항목 각각에 대해, 예측값이 사용자가 입력한
  목표값과 오차 0이면 100점, 오차가 목표값의 ±5%(기본 tolerance) 이상 벌어지면 0점으로 선형 정규화.
  4개 항목 점수를 평균.
- 세 서브점수를 0.60 / 0.25 / 0.15로 가중합해서 최종 `total` 점수(0~100) 산출.
- (오늘 세션에서 이 공식의 NaN 처리 버그 하나 고쳤음 — 결측치가 만점으로 둔갑하던 문제, 커밋 `fc75b8b`)

### (B) 학습 파이프라인 내부 "상대 품질 지수 v3" — Base→Rev15 기준 상대평가 (`ml_engine/trench_core.py`)

사용자의 목표값과 무관하게, **관측된 Recipe 중 가장 안 좋았던 `Base`를 0점, 가장 좋았던 `Rev15`(=`optimal_recipe`)를
100점**으로 놓고 상대적으로 어디쯤인지 매기는 점수. AI 파라미터 추천(Output E, OFAT 스캔) 엔진 내부에서
"이 파라미터를 바꾸면 Rev15 방향으로 가는가"를 판단하는 데 씀 (사용자 목표 점수(A)가 1순위 기준이고, 이 v3
점수는 2순위 tie-breaker로 사용됨 — `trench_core.py`의 `ranking_key()` 참고).

```
품질 지수 v3 = 85% × Wafer 전체 Core + 15% × 최저 Zone Core
Core = 60% × Spec Pass 확률 평균 + 20% × Defect Count 점수 + 20% × Defect Severity 점수
```

- **Spec Pass 확률 평균 (60%)**: (A)의 하드 pass/fail(0·1)이 아니라, 분류모델이 낸 **확률값
  (`predict_proba`) 자체**를 25개 Site × 4개 CD/Depth 항목에 대해 평균. 확률이라 더 민감하게 반응함.
- **Defect Count 점수 (20%) / Defect Severity 점수 (20%)**: 아래 4번 항목에서 설명하는 Base↔Rev15
  기준 정규화 점수를 그대로 가져와 씀.
- **최저 Zone Core (15%)**: Center/Mid/Edge/Extreme Edge 4개 Zone 중 Core 점수가 가장 낮은 Zone의
  점수를 별도로 반영 — Wafer 전체 평균은 괜찮아도 특정 Zone(보통 Edge 계열)에서만 나쁘게 나오는
  경우를 놓치지 않기 위한 가드레일.
- `metadata.json`의 `quality_v2.weights`에 60/20/20/85/15 숫자가 그대로 박혀있음.
- **참고용 레거시 v1**: `quality_index_v1 = 70% × 경험적(empirical) Spec Pass율 + 30% × Particle-free
  Site 비율` — 초기 버전이라 지금은 화면 표시·순위 결정에 안 쓰고 내부 참고값으로만 남아있음.

**중요**: (A)와 (B)는 가중치 구성(60/25/15 vs 60/20/20+85/15)도, 기준점(사용자 목표 vs Base/Rev15
상대값)도 다른 별개 공식임. 팀원한테 설명할 때 "화면에 보이는 점수"는 (A), "AI 추천이 내부적으로
방향을 판단하는 데 쓰는 점수"는 (B)라고 구분해서 얘기하면 됨.

---

## 4. Defect(결함) 분석 및 수식화

Defect는 세 가지 모델로 나눠서 예측함 (전부 Site 단위 예측 후 Wafer 단위로 집계):

| 예측 대상 | 방식 | 배포 모델 | 후처리 |
|---|---|---|---|
| `Particle_Defect` (Site에 Defect 있음/없음, `Defect_Count > 0`로 정의) | 이진분류 확률 | MLP | threshold 0.5 |
| `Defect_Count` (Site당 예상 Defect 개수) | 회귀 | MLP | 0 미만 클립 |
| `Defect_Severity_Score` (Site당 심각도 점수) | 회귀 | MLP | 0~10 클립 |

### Wafer 단위로 집계하는 방법

- **예상 Defect 발생 Site 수** = 25개 Site의 `Particle_Defect` 확률을 그냥 합산 (`expected_defect_sites`)
- **Wafer에 하나라도 Defect가 있을 확률** = Site별 확률을 서로 독립이라고 가정하고
  `1 − Π(1 − p_site)`로 계산 (`predicted_any_particle_probability_pct_independence_approx`).
  대시보드의 "Particle 발생 여부"(True/False)는 이 값이 50% 이상이면 True로 표시함.
- **총 Defect Count** = `Defect_Count` 회귀 예측값을 25개 Site에 대해 합산.
- **평균 심각도** = `Defect_Severity_Score`를 25개 Site에 대해 평균.

### Defect를 점수로 바꾸는 정규화 방식 (3번 항목 (B)에서 쓰는 그 점수)

Defect Count/Severity는 "낮을수록 좋음" 지표라서, `Base`(가장 안 좋았던 관측 Recipe)와 `Rev15`(가장
좋았던 Recipe)의 **실측 평균값**을 기준점으로 선형 정규화함 (`_lower_is_better_score` 함수):

```
점수 = 100 × (Base 실측값 − 예측값) / (Base 실측값 − Rev15 실측값)   (0~100 클립)
```

즉 예측값이 `Base` 수준이면 0점, `Rev15` 수준이면 100점, 그 사이는 선형 보간. 이 기준값은
Wafer 전체 기준과 Zone별 기준(Center/Mid/Edge/Extreme Edge 각각 따로) 두 세트가 있고, 전부
`metadata.json`의 `quality_v2.defect_score_reference`에 숫자로 박혀있음. 예를 들어:

| 기준 | Base(0점 기준) 총 Defect Count | Rev15(100점 기준) 총 Defect Count |
|---|---|---|
| Wafer 전체 | 12.67개 | 1.31개 |
| Edge Zone | 5.2개 | 0.375개 |
| Extreme Edge Zone | 5.0개 | 0.625개 |

**명시적 한계**: 이 기준값은 회사에서 공식적으로 정한 Defect 허용 기준(spec)이 아니라, **관측된
데이터 중 최악(Base)·최선(Rev15) 사례의 평균값**을 상대적 잣대로 쓴 것임 — `metadata.json`
`limitations`에 "official defect-count and severity acceptance limits were not supplied"라고
명시되어 있음. 공식 Defect 허용 기준이 따로 있다면 그걸로 교체해야 함.

### Spec Pass 판정과는 다른 축이라는 점

CD/Depth의 "Spec 만족 여부"는 위 Defect 점수와 별개로, `empirical_spec_limits`(Site 실측치에서
역산한 상한/하한, 예: `Bottom_CD_nm` 224.43~247.8nm)를 직접 벗어났는지로 따로 판정함. Defect와
CD/Depth Spec 미달은 원인이 다를 수 있어서 두 축을 합쳐서 하나의 숫자로 뭉개지 않고, 위 3번
항목처럼 "Spec Pass 60% + Defect Count 20% + Defect Severity 20%"로 **따로 채점한 뒤 가중합**하는
구조로 만들어져 있음.

---

## 참고: 대시보드에서 보이는 "Defect 건수"와 여기서 말하는 모델 예측은 다른 것

Process Dashboard의 AI 분석 문장(`generate_dashboard_analysis`, Rule-Base)에서 "Defect 3건 초과
Wafer 몇 장" 식으로 나오는 숫자는 **모델 예측이 아니라 Site_Level_Raw의 실측 `Total_Defect_Count`
컬럼을 그대로 집계**한 것 (기준치 `PARTICLE_DEFECT_THRESHOLD = 3`, `data_utils.py`). 이 문서에서
설명한 `Particle_Defect`/`Defect_Count`/`Defect_Severity_Score` 모델 예측값과 혼동하지 않도록 주의.
