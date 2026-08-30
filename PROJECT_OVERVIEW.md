# Etch AI Decision Support System — 프로젝트 전체 정리

이 문서 하나로 "우리가 뭘 만들었고, 왜 이렇게 만들었고, 데이터/모델 학습은 어떻게 했고,
지금까지 어떤 과정을 거쳤는지" 전부 확인할 수 있게 정리한 문서입니다. 코치님·팀원 공유용.

빠르게 훑어볼 거면 [README.md](README.md), [MENTOR_UPDATE.md](MENTOR_UPDATE.md)를 먼저 보고,
"왜/어떻게"까지 궁금하면 이 문서를 보면 됩니다. trench 공정만 더 깊게 파고든 자료는
[`ml_engine/training/trench/ANALYSIS_SUMMARY.md`](ml_engine/training/trench/ANALYSIS_SUMMARY.md)에 따로 있습니다.

---

## 1. 배경과 목표

반도체 Etch 공정 평가를 지금까지는 엔지니어 개인 경험/판단에 의존해서 편차가 컸습니다.
이걸 AI 시뮬레이터로 시스템화해서, Recipe를 입력하면

1. 예상 Wafer 품질을 예측하고
2. 목표 스펙 대비 얼마나 가까운지 점수로 매기고
3. 어떤 파라미터를 어느 방향으로 조정해야 하는지까지 추천

해주는 게 최종 목표입니다. 목표는 "평가 반복 횟수(N회 재작업)를 줄이는 것".

**대상 공정 4개**: `isolation`(1차 공정, 팀에서 먼저 분석 시작), `trench`(2차 공정, 멘토님이
"새 공정도 만들어보라"고 추가로 준 데이터로 나중에 착수), `gate`(3차 공정, Gate PolySi Etch),
`metal`(4차 공정, Metal Al Etch) — gate/metal은 멘토님이 이후 추가로 준 데이터로 isolation/trench와
동일 파이프라인 재사용해서 신규 학습.

> **참고**: 4개 공정 전부 데이터 분석·모델 학습이 끝나 있고(§4, §5-3 표에 4개 공정 다 나와 있음), gate·metal도
> isolation·trench와 완전히 같은 방식(whole-recipe-holdout CV 기반 자동 모델 선택)으로 학습됨.
> 다만 §6~9(품질 점수 공식, Defect 수식)의 서술형 설명·Base/Rev15 기준값 표는 isolation·trench 예시로만
> 작성돼 있음 — 공식 자체는 4개 공정 공통(`ml_engine/scoring.py` 하나를 공유)이라 문제는 없고, gate/metal의
> 실제 기준값이 궁금하면 `ml_engine/gate_models/metadata.json`, `ml_engine/metal_models/metadata.json` 참고.

## 2. 팀 구성

| 이름 | 역할 |
|---|---|
| 천승현 (user) | 머신러닝팀 — 이 저장소(`etch-AI-dashboard`) 통합 작업 담당 |
| 곽영진 | 머신러닝팀 — isolation 공정 예측 엔진(`etch_simulator`) 원저자 |
| 정윤서, 손유찬, 박준영 | 대시보드팀 — 손유찬님이 만든 대시보드가 최종 채택됨 |
| 이태윤 멘토님 | 직무, 프로젝트 관련 주제 선정 및 불꽃 피드백 |
| 김동진 코치님 | ai활용, 머신러닝, 대시보드 관련하여 코칭 |

---

## 3. 여기까지 오게 된 과정 (타임라인)

1. **데이터 분석 단계** — isolation·trench 둘 다 데이터 정합성 검증 → 종합 품질 점수(`quality_score.py`)
   산출 → 파라미터 영향 EDA → ML 가능성 시험(LORO 교차검증, RandomForest R² 0.94~0.95)까지 끝냄.
   결과 문서: `v2_분석_결과_설명.docx`, `두번째공정/trench_분석_결과_설명.docx` 등(팀 로컬/iCloud 보관,
   이 git 저장소에는 코드/모델만 올라와 있고 이 설명 문서들은 포함되어 있지 않음 — 필요하면 별도 공유 필요).
   **주의**: 이 R² 0.94~0.95는 이 초기 타당성 검증 단계에서 나온 참고 수치이며, 실제 배포된 모델
   (§5-3)의 공식 벤치마크가 아님 — 배포 모델 선정은 §5-2의 whole-recipe-holdout CV 기준으로 별도 진행됨.
2. **대시보드 방향 전환** — 대시보드팀 3명이 각자 만든 버전 중 멘토님이 **손유찬님의
   `etch-AI-dashboard`**를 최종 선택. 기존에 별도로 만들던 레포는 폐기하고 이걸로 통합 작업 시작.
3. **곽영진님의 기존 예측 엔진 발견 → 재사용** — `github.com/sunic1616-lgtm/etch_simulator`에
   isolation 공정용 예측 엔진(항목별 모델 학습·선택, OFAT 파라미터 추천까지)이 이미 구현돼 있어서
   그대로 채택. 다만 학습된 `.joblib` 파일 자체는 그 레포에 없어서 로컬에서 직접 재학습시킴.
4. **trench 신규 구축** — 영진님 스크립트를 그대로 복사해서 파라미터 개수(12→20)와 Depth 단위(Å→nm)만
   trench 데이터에 맞게 바꿔 동일 파이프라인으로 새로 학습.
5. **대시보드 통합** — `etch-AI-dashboard` 레포의 `feature/ai-simulation-seunghyun` 브랜치에
   isolation/trench ML 엔진 연결, 공정 선택기 추가, 배포용 기본 데이터 포함.
6. **1차 멘토 UX 피드백 반영** — Rev 선택 시 파라미터 전체 표시, Top/Mid/Bottom CD·Depth Wafer Map
   한번에 비교, AI 분석 섹션을 Process Dashboard 탭 맨 위로 이동.
7. **학습 range 밖 입력값 가드 추가** — 트리 기반 모델이 학습 범위 밖 입력에서 예측이 평평해지는
   (saturate) 현상을 발견해서, 범위 밖이면 예측 전에 막고 위반 항목을 표로 보여주는 기능 추가.
8. **2차 멘토 카톡 피드백 6개 반영** — Chamber 선택 제거(장비별 대표 Chamber 자동 사용), 목표 품질
   라벨에 참고값 표시, 시뮬레이터를 목표모드/Recipe입력모드 2개로 분리, Rev별 스코어링 순위표 추가,
   Wafer 단면 Profile 차트 추가.
9. **버그 수정 + 기능 전수검토** — 아래 9번 항목에 정리.
10. **최종 점검** — `main` 브랜치가 `feature/ai-simulation-seunghyun`보다 뒤처져 있던 것을 발견해
    fast-forward로 동기화. 당시 `main`/`feature`/원격 전부 동일 커밋.
11. **gate/metal 공정 추가 + 팀원 브랜치 통합** — 3·4번째 공정(gate/metal) 신규 학습·대시보드 연결,
    팀원들이 각자 작업한 브랜치(로그인 화면, 이력 조회 탭, Dashboard 재구성 등)를 순서대로 병합.
    이 과정에서 `main`이 다시 여러 커밋 뒤처졌고, 이번 문서 정리와 함께 다시 fast-forward로
    재동기화함. 현재 `main`이 이 저장소의 최신 상태.

---

## 4. 데이터 — 어떤 데이터를 얼마나 분석했나

| 항목 | isolation | trench | gate | metal |
|---|---|---|---|---|
| 원본 파일 | `isolation recipe full dataset_2.xlsx` | `trench recipe full dataset.xlsx` | `gate recipe full dataset.xlsx` | `metal recipe full dataset.xlsx` |
| Recipe Version 개수 | 16개 (`Base`, `Rev1`~`Rev15`) | 16개 (`Base`, `Rev1`~`Rev15`) | 16개 (`Base`, `Rev1`~`Rev15`) | 16개 (`Base`, `Rev1`~`Rev15`) |
| Wafer 개수 | 244장 | 236장 | 231장 | 236장 |
| Site 측정값 개수 | 6,100행 (244×25) | 5,900행 (236×25) | 5,775행 (231×25) | 5,900행 (236×25) |
| Wafer당 Site 구성 | Center 1 + Mid 4 + Edge 10 + Extreme Edge 10 = 25개 | 동일 | 동일 | 동일 |
| Stage 개수 | 2개 (S1, S2) | 4개 (S1~S4) | 1개(접두사 없음) | 1개(접두사 없음) |
| Recipe 파라미터 개수 | 12개 | 20개 | 5개 | 5개 |
| Depth 단위 | Å (대시보드에는 ×0.1로 nm 환산해서 표시) | nm | nm | nm |
| Equipment/Chamber 관측 조합 | `AMAT-CENTRIS`+`CH-C`(47) / `CH-D`(64), `LAM-2300FLEX`+`CH-A`(80) / `CH-B`(53) | `AMAT-CENTRIS`+`CH-C`(63) / `CH-D`(64), `LAM-2300FLEX`+`CH-A`(54) / `CH-B`(55) | `AMAT-CENTRIS`+`CH-C`(58) / `CH-D`(47), `LAM-2300FLEX`+`CH-A`(61) / `CH-B`(65) | `AMAT-CENTRIS`+`CH-C`(71) / `CH-D`(53), `LAM-2300FLEX`+`CH-A`(58) / `CH-B`(54) |

**공통 제약**: 네 공정 다 관측된 Recipe가 16개뿐이고, `Base`에서 시작해 파라미터를 한두 개씩 순차적으로
바꿔나간 이력이라 파라미터 간 상관관계가 높음 — 그래서 "이 파라미터가 원인"이라고 인과적으로 말할 수
없고, 모델이 잡아낸 연관성(association)으로만 해석해야 함. (`metadata.json`의 `limitations`에 명시)

원본 데이터는 `data/{isolation,trench,gate,metal}_dataset.xlsx`로 이 저장소에 포함되어 있어서
누구든 업로드 없이 바로 대시보드를 씀. `recipe master.csv`류 원본 파일은 isolation/trench는 내용이
이미 위 xlsx에 포함돼 있어서 별도로 안 올렸고, gate/metal은 `ml_engine/training/{gate,metal}/`에
별도로 포함되어 있음.

---

## 5. 모델 학습 — 어떻게 시켰나

### 5-1. 파이프라인 (isolation·trench·gate·metal 4개 공정 공통, 스크립트 4개 순서대로 실행)

| 스크립트 | 하는 일 |
|---|---|
| `train_models.py` | CD/Depth 회귀 + Particle 발생여부 분류를 RandomForest/XGBoost/MLP 세 방식으로 각각 학습, `.joblib`로 저장 |
| `validate_unseen_recipes.py` | **Recipe 단위로 통째로 빼는** 교차검증(`GroupKFold(n_splits=4)`, `Recipe_Version` 그룹)으로 "안 써본 Recipe 조합"에 대한 일반화 성능 측정 |
| `select_deployment_models.py` | 위 결과에서 CD/Depth 회귀·Particle 분류 항목별 최고 성능 모델을 배포용으로 선정, `metadata.json` 갱신 |
| `train_quality_v2.py` | Spec Pass 여부(4개) + Defect Count/Severity를 추가 학습, Zone별 품질 점수 계산에 쓰는 기준값도 `metadata.json`에 추가 |

(경로: `ml_engine/training/{isolation,trench,gate,metal}/` — 넷 다 파라미터 개수/Stage 구조만
다르고 스크립트 로직은 완전히 동일한 복사본. gate/metal은 `created_utc: 2026-08-10`에 학습 완료,
`ml_engine/{gate,metal}_models/metadata.json`에서 확인 가능)

### 5-2. 모델을 "어떻게" 골랐나 — 핵심은 whole-recipe-holdout

일반적인 랜덤 교차검증이 아니라, **검증 fold에서 특정 Recipe의 Wafer/Site를 통째로 빼는** 방식을 씀.
목적은 "이미 아는 Recipe의 새 Wafer"가 아니라 "사용자가 시뮬레이터에서 아직 안 써본 새로운 Recipe
조합"에 대한 예측 신뢰도를 검증하는 것. Wafer 단위로만 나누는(같은 Recipe의 다른 Wafer는 학습에
남는) 더 쉬운 검증 방식도 별도로 돌렸지만(`wafer_group_cv_selected_models`), **배포 선택 기준으로는
안 씀** — 실제로 이 기준으로는 MLP가 CD/Depth 회귀에서 좋게 나왔지만, Recipe-holdout 기준으로 보면
성능이 무너져서 배포에서 제외됨.

선택 기준(항목 종류별):
- CD/Depth 회귀, Defect Count/Severity 회귀 → Recipe-holdout CV **MAE(평균절대오차) 최저**
- Spec Pass 이진분류 → Recipe-holdout **Balanced Accuracy 최고**(동점이면 ROC-AUC)
- Particle_Defect 이진분류 → Recipe-holdout **Average Precision 최고**(동점이면 Balanced Accuracy)

이 선택 로직은 사람이 수동으로 고르는 게 아니라 스크립트가 CV 점수를 정렬해서 자동으로 뽑음
(`select_deployment_models.py`, `train_quality_v2.py` 코드로 확인 가능 — 하드코딩된 편향 없음).

**실제 CV 수치(대표값)**: 4개 공정 전부 로컬에 원시 결과가 있고(위 §13 참고), Top_CD 회귀 R²만
뽑아보면 MLP가 왜 대부분 배포에서 제외되는지 바로 드러남 — MLP는 4개 공정 전부 R²가 크게 음수(=
평균값 찍기보다 못함):

| 공정 | RandomForest | XGBoost | MLP |
|---|---|---|---|
| isolation | 0.74 | 0.37 | −30.4 |
| trench | 0.22 | 0.61 | −171.0 |
| gate | −0.11 | −0.49 | −42.6 |
| metal | 0.37 | 0.27 | −28.5 |

Particle_Defect 분류(Balanced Accuracy)도 같은 패턴 — MLP는 4개 공정 전부 정확히 0.500(찍기 수준),
RandomForest/XGBoost는 0.52~0.65. **정직한 추가 발견**: gate는 RandomForest/XGBoost조차 CD 회귀
R²가 약하거나 음수(§11 한계 참고) — MAE(0.5~0.9nm)는 작아 보이지만 CD 값 자체의 변동폭이 작아서
그런 것이지 예측이 특별히 정교한 건 아님.

### 5-3. 항목별 최종 선택 모델

| 예측 항목 | isolation | trench | gate | metal |
|---|---|---|---|---|
| Top CD | RandomForest | XGBoost | RandomForest | RandomForest |
| Mid CD | RandomForest | RandomForest | RandomForest | RandomForest |
| Bottom CD | RandomForest | RandomForest | RandomForest | RandomForest |
| Depth | XGBoost | RandomForest | XGBoost | XGBoost |
| Top/Mid/Bottom/Depth Spec Pass (4개) | XGBoost / RandomForest / RandomForest / XGBoost | XGBoost / RandomForest / MLP / MLP | RandomForest / XGBoost / MLP / XGBoost | XGBoost / XGBoost / XGBoost / MLP |
| Particle_Defect | XGBoost | MLP | XGBoost | XGBoost |
| Defect_Count | RandomForest | MLP | RandomForest | MLP |
| Defect_Severity_Score | RandomForest | MLP | RandomForest | MLP |

(위 표는 `ml_engine/{process}_models/metadata.json`의 `selected_models`를 그대로 옮긴 것 —
네 공정 다 whole-recipe-holdout CV 자동 선택 결과이며 gate/metal도 사람이 고른 게 아님)

**MLP 관련 참고**: MLP는 곽영진님의 원본 `etch_simulator` 레포부터 있던 3파전 비교 후보 중 하나임
(trench에서 새로 추가한 게 아니라 그대로 복사해온 방식). 이 3개(RandomForest/XGBoost/MLP)를 왜
후보로 골랐는지 자체는 원본 레포에도, 이 문서에도 별도로 문서화된 적이 없음 — 코드에도
`families = ["RandomForest", "XGBoost", "MLP"]`가 주석 없이 그대로 있을 뿐(`train_models.py`).
표형 데이터+독립 Recipe 수가 적은 상황에 트리 계열이 보통 유리하고 MLP를 비교군으로 함께
두는 건 일반적인 ML 실무 관행 정도로 이해하면 됨 — 데이터로 증명된 선택은 아님.

**"왜 항목별로 MLP가 뽑히는 게 다른가"는 실제 데이터로 확인함(2026-08 재검증)**: 처음엔 "trench는
파라미터가 더 많아서(20개, 12개 대비)"라고 설명했었는데, 이건 틀렸음 — 실제 Feature 개수(§5-4)는
isolation 18개 / trench 26개 / gate 11개 / metal 11개로, isolation이 gate·metal보다 Feature가
많은데도 MLP 0회고, gate·metal은 Feature 개수가 완전히 같은데 MLP가 각각 1회·3회로 다름 —
"파라미터 개수"로는 설명이 안 됨. `data/{process}_dataset.xlsx`(Site_Level_Raw)와
`metadata.json`(selected_models)을 4공정×11항목(총 44개) 전부 대조해서 다시 찾아본 진짜 패턴은
**공정이 아니라 항목(target) 종류에 달려 있음**:
- Top/Mid/Bottom CD·Depth 회귀는 4공정 16번 전부 RandomForest 아니면 XGBoost — MLP는 단 한 번도 안 뽑힘
- MLP는 딱 두 부류에서만 뽑힘: ①`Defect_Count`/`Defect_Severity_Score`(고왜도 count형 회귀 — MLP가
  뽑힌 행의 skew 중앙값 2.94 vs 나머지 0.90, 변동계수 중앙값 2.78 vs 0.04로 뚜렷하게 노이즈가 큼)
  ②분류 항목 중 클래스가 비교적 균형잡힌 것(`Bottom_CD_Spec_Pass`/`Depth_Spec_Pass`/`Particle_Defect`,
  소수클래스 비율 중앙값 0.40) — `Top_CD_Spec_Pass`/`Mid_CD_Spec_Pass`처럼 클래스가 많이 치우친
  항목(소수클래스 비율 중앙값 0.17)은 MLP가 한 번도 안 뽑힘
- trench·metal이 MLP를 상대적으로 많이 쓰는 건 이 "MLP가 유리한 항목 종류"가 그 공정에서 우연히
  더 많이 MLP 쪽으로 넘어갔기 때문이지, 파라미터 개수 때문이 아님
- **정직한 한계**: MLP가 뽑힌 건 44개 중 9개뿐이고 서로 다른 항목 종류는 5가지뿐이라, 엄밀한 통계적
  유의성을 주장할 수 있는 표본은 아님 — 패턴은 실제로 보이지만 확정적 인과 증명은 아님

### 5-4. 모델 입력(Feature)

| | isolation | trench |
|---|---|---|
| Recipe 파라미터 | 12개 (S1/S2 × Time/RF Bias/Pressure/Gas 2종) | 20개 (S1~S4 × Time/RF Bias/Pressure/Gas 2종) |
| + 공통 Feature | `Radius_frac`, `Angle_sin`, `Angle_cos`(Site 위치) + `Equipment_Model`, `Chamber_ID`, `Zone`(범주형) | 동일 |
| 총 Feature 수 | 18개 | 25개 |

모델은 "이 Recipe로, 이 장비/챔버에서, Wafer의 이 위치를 찍으면 값이 어떻게 나올까"를 Site 단위로
예측하고, 대시보드에서는 이 25개 Site 예측을 평균/표준편차로 묶어서 Wafer 단위 결과로 보여줌.

---

## 6. 품질 점수 — 두 가지가 있음 (헷갈리기 쉬운 부분, 명확히 구분)

### (A) 대시보드 화면에 실제로 보이는 점수 — "목표 대비 종합 점수" (`ml_engine/scoring.py`)

Output B(목표 대비 평가), Output C/D(최적 Recipe 추천·비교), Rev별 스코어링 순위표에 보이는 점수.
**사용자가 화면에서 입력한 목표 스펙**에 얼마나 가까운지 채점.

```
종합 점수 = Pass Rate 60% + Uniformity 25% + 목표 근접도 15%
```

- Pass Rate: 4개 Spec Pass 항목(Top/Mid/Bottom CD, Depth)을 25개 Site 전체 평균한 pass 비율
- Uniformity: CD/Depth Uniformity(Site 간 변동계수%)를 사용자가 정한 허용 최대치 기준 선형 정규화(0점~100점) 후 평균
- 목표 근접도: Top/Mid/Bottom CD·Depth 4개 항목이 목표값과 오차 0=100점, ±5%(기본값) 이상=0점으로 선형 정규화 후 평균
- isolation·trench 둘 다 완전히 같은 공식(`ml_engine/scoring.py` 하나를 공유), 파라미터 개수만 다를 뿐 채점 방식은 동일
- 60/25/15 가중치는 학습으로 최적화된 값이 아니라 `ml_engine/scoring.py`에 그대로 박혀 있는 설계 상수임(과대적합 방지를 위한 실험이나 가중치 튜닝은 하지 않음)

### (B) 학습 파이프라인 내부 "상대 품질 지수 v3" — Base→Rev15 기준 상대평가 (`{process}_core.py`)

사용자 목표와 무관하게, **관측된 Recipe 중 가장 안 좋았던 `Base`를 0점, 가장 좋았던 `Rev15`(=`optimal_recipe`)를
100점**으로 놓고 상대 위치를 매김. AI 파라미터 추천(Output E) 엔진이 "이 파라미터를 바꾸면 Rev15
방향으로 가는가"를 판단하는 내부 기준(사용자 목표 점수 (A)가 1순위, 이 점수는 2순위 tie-breaker).

```
품질 지수 v3 = 85% × Wafer 전체 Core + 15% × 최저 Zone Core
Core = 60% × Spec Pass 확률평균 + 20% × Defect Count 점수 + 20% × Defect Severity 점수
```

isolation·trench 둘 다 가중치(60/20/20, 85/15)는 동일하고, Base/Rev15 기준값(다음 섹션 표)만 공정별로 다름.
이 가중치들도 (A)와 마찬가지로 학습된 값이 아니라 하드코딩된 설계 상수임 — `train_quality_v2.py`가
`metadata.json`에 그대로 써넣고, `{process}_core.py`가 그 숫자를 그대로 읽어서 쓸 뿐 fit/최적화하는
로직은 어디에도 없음.

**파라미터 추천(Output E)의 실제 우선순위**: 후보 파라미터 변경안을 비교할 때 아래 6개 지표를
순서대로 비교하는 tuple 정렬(`{process}_core.py`의 `ranking_key()`)을 씀 — 1번이 같으면 2번,
2번도 같으면 3번... 순으로 비교:
1. 사용자 목표 대비 종합 점수(A) 상승분
2. 상대 품질 지수 v3(B) 상승분
3. 최저 Zone Core 상승분
4. 평균 Spec Pass 확률 상승분
5. Defect Count 감소분
6. Defect Severity 감소분

**정리**: 화면에 보이는 건 (A), AI 추천이 내부적으로 방향 판단할 때 참고하는 건 (B). 둘은 가중치도
기준점(사용자 목표 vs Base/Rev15 상대값)도 다른 별개 공식임.

---

## 7. Defect(결함) 분석 및 수식화

세 모델로 나눠 예측(전부 Site 단위 → Wafer 단위 집계):

| 예측 대상 | 방식 | 후처리 |
|---|---|---|
| `Particle_Defect` (Site에 Defect 있음/없음) | 이진분류 확률 | threshold 0.5 |
| `Defect_Count` (Site당 예상 개수) | 회귀 | 0 미만 클립 |
| `Defect_Severity_Score` (Site당 심각도) | 회귀 | 0~10 클립 |

집계: 예상 Defect Site 수=확률 합산, Wafer에 하나라도 있을 확률=`1-Π(1-p_site)`(독립 가정),
총 Defect Count=25개 Site 합산, 평균 심각도=25개 Site 평균.

Defect Count/Severity는 "낮을수록 좋음" 지표라 `Base`(0점)·`Rev15`(100점) 실측 평균 기준 선형 정규화:

```
점수 = 100 × (Base 실측값 − 예측값) / (Base 실측값 − Rev15 실측값)   (0~100 클립)
```

| 기준(Wafer 전체) | isolation Base→Rev15 | trench Base→Rev15 |
|---|---|---|
| 총 Defect Count | 11.47개 → 1.15개 | 12.67개 → 1.31개 |
| 평균 Defect Severity | 3.66 → 0.07 | 2.92 → 0.07 |

Zone별(Center/Mid/Edge/Extreme Edge) 기준값도 따로 있음 — 전체 평균은 괜찮아 보여도 특정 Zone에서만
나쁜 경우를 놓치지 않기 위한 가드레일. **이 기준값은 공식 Defect 허용 기준(spec)이 아니라 관측된
데이터 중 최악·최선 사례를 상대적 잣대로 쓴 것**(공식 기준이 따로 있으면 교체 필요).

CD/Depth "Spec 만족 여부"는 이 Defect 점수와 별개로 `empirical_spec_limits`(Site 실측치에서 역산한
상/하한)를 벗어났는지로 따로 판정하고, 최종 점수에서 "Spec Pass 60% + Defect Count 20% + Severity 20%"로
합산됨 — 두 축을 하나로 뭉개지 않고 따로 채점 후 가중합.

**대시보드 AI 분석 문장의 "Defect 3건 초과 Wafer 몇 장"은 위 모델 예측이 아니라 실측
`Total_Defect_Count` 컬럼을 그대로 집계한 것**(기준치 `PARTICLE_DEFECT_THRESHOLD = 3`). 혼동 주의.

---

## 8. 대시보드 기능 — 지금까지 반영된 것들

| 기능 | 계기 |
|---|---|
| 공정 선택(isolation/trench/gate/metal), Recipe→예측→목표평가→추천 4단계 | 초기 통합 + gate/metal 추가 |
| Rev 선택 시 파라미터 전체 한번에 표시 | 1차 멘토 피드백 |
| Top/Mid/Bottom CD·Depth Wafer Map 한번에 비교 | 1차 멘토 피드백 |
| AI 분석 섹션 Process Dashboard 탭 맨 위 배치 | 1차 멘토 피드백 |
| 파라미터별 조정 제안(OFAT) | 영진님 엔진 기능 그대로 계승 |
| 학습 range 밖 입력값 가드 (+"그래도 실행" 우회, 신뢰도 낮음 경고) | 트리 모델 saturate 현상 발견 후 자체 추가 |
| Chamber 선택 제거(장비별 대표 Chamber 자동 사용) | 2차 멘토 카톡 피드백 |
| 목표 품질 라벨에 "참고값→목표" 형식 표시 | 2차 멘토 카톡 피드백 |
| 시뮬레이터 목표모드/Recipe입력모드 2분리 | 2차 멘토 카톡 피드백 |
| Rev별 스코어링 순위표+차트(실측 기반) | 2차 멘토 카톡 피드백 |
| Wafer 단면 Profile 차트(Edge→Center→Edge) | 2차 멘토 카톡 피드백 |
| Parameter 변경 이력 조회 (신규 탭) | 팀원(정윤서님) 신규 기능 |
| 신규 공정 품질 예측 (신규 탭) — Layer 순서 입력 → 가장 비슷한 기존 공정 실측 평균 참고값 제공 | 팀원 정의서 기반, 실제 AI 예측 아님을 화면에 명시 |

---

## 9. 발견하고 고친 버그

| 버그 | 원인 | 커밋 |
|---|---|---|
| 목표 품질 바꿔도 파라미터 추천 표가 그대로 | OFAT 후보 랭킹이 목표값과 무관한 내장 quality_index로만 매겨짐 | `05ff726` |
| 목표값이 0이면 미달 이슈 문구가 안 뜸 | severity 계산이 `target_value`가 0이면 강제로 0 처리 | `ec78502` |
| 결측치(NaN)가 만점(100점)으로 둔갑 | `min(100, nan)`이 파이썬에서 100을 반환하는 특성 때문 | `fc75b8b` |

모두 python으로 직접 예외값 테스트해서 재현·검증 후 수정, 회귀 테스트로 기존 정상 케이스 안 깨졌는지 확인함.

---

## 10. 배포

- **URL**: https://hera-etch-seunghyun.streamlit.app/ (Streamlit Community Cloud, 개인 계정 Private 배포)
- **기준 브랜치**: `main` (코드 push하면 자동 반영).
- Vercel 이전은 검토했으나 상시서버+WebSocket 구조라 서버리스인 Vercel과 안 맞아 Streamlit Cloud 유지로 코치님과 확정.

---

## 11. 한계 (모델 metadata에 명시된 그대로)

- 관측 Recipe가 16개뿐이고 서로 독립적인 실험이 아니라 순차적 변경 이력이라, 파라미터 인과관계는 알 수 없음(연관성만)
- Process/Layer 컬럼이 원본 데이터에 없어서 모델 Feature로 못 씀
- Recipe 단위 holdout으로 "안 써본 Recipe" 일반화는 검증했지만, 같은 Wafer 내 데이터 유출은 막았을 뿐 그 이상의 보장은 아님
- Spec 상/하한은 엔지니어링 공식 기준이 아니라 관측치에서 역산한 경험적 값
- Defect 정규화 기준값(Base/Rev15)도 공식 허용 기준이 아니라 관측된 최악/최선 사례
- gate 공정의 CD 회귀는 whole-recipe-holdout R²가 약하거나 음수(Top_CD RandomForest −0.11, Mid_CD
  −0.05, Bottom_CD 0.20, Depth만 0.02~0.09)임 — 채택 모델도 "완전히 새로운 Recipe"에서는 평균값
  찍기와 큰 차이가 없다는 뜻. 화면 MAE(0.5~0.9nm)가 작은 건 CD 값 자체 변동폭이 작아서지 예측이
  정교해서가 아님. isolation/trench/metal은 R²가 대체로 양수(0.2~0.8)로 더 안정적
- 파라미터별 조정 제안(OFAT)의 후보값은 관측된 16개 Recipe(`Base`~`Rev15`)에 실제로 있었던 값으로만
  제한됨(`_observed_parameter_values`) — 보간·외삽 없음. 그래서 목표 품질이 지금까지 시도된 적 없는
  값이면, 애초에 거기 도달하는 파라미터 조합 자체가 후보에 없을 수 있음
- 목표 근접도 서브점수(`_proximity_score`)는 목표 오차가 ±5%(기본값)를 넘으면 0으로 고정됨. 종합
  점수에서 근접도 비중은 15%뿐이라(Pass Rate 60%+Uniformity 25%), 목표가 많이 멀면 추천 순위가
  사실상 Pass Rate·Uniformity 위주로 정해지고 목표 방향을 충분히 반영하지 못할 수 있음 — 이런 경우
  대시보드 화면에 경고 배너로 안내됨

---

## 12. 코드/문서 지도

| 경로 | 내용 |
|---|---|
| `app.py` | 화면(UI) 전체 |
| `model.py` | 화면 입력 → AI 모델 호출 → 점수/추천 계산 |
| `ml_engine/scoring.py` | 대시보드 목표 대비 종합 점수 공식(공정 공통) |
| `ml_engine/{isolation,trench,gate,metal}_core.py` | 공정별 예측 코어(학습된 모델 로드·실행, 품질 지수 v3, Defect 정규화) — 4개 다 존재 |
| `ml_engine/{isolation,trench,gate,metal}_models/` | 학습 완료 모델(`.joblib`) + `metadata.json` — 4개 다 존재 |
| `ml_engine/training/{isolation,trench,gate,metal}/` | 모델 학습 원본 스크립트(재현 가능) — 4개 다 존재 |
| `ml_engine/training/trench/ANALYSIS_SUMMARY.md` | trench 전용 상세 분석(이 문서의 5~7번 섹션을 더 깊게) |
| `data/*.xlsx` | 배포용 기본 데이터 |
| `README.md` | 팀원용 빠른 소개 + 로컬 실행법 |
| `MENTOR_UPDATE.md` | 코치님 공유용 기능 요약 |

---

## 13. 아직 안 한 것 / 다음 단계 후보

- gate/metal은 이미 수동으로 온보딩 완료(4공정 모두 운영 중). 하지만 **온보딩 자체의 자동화**(신규 공정 추가 시
  학습 스크립트·대시보드 연결 지점을 수동으로 안 건드려도 되게 만드는 것)는 여전히 안 돼있음 — 코치님과 상의 후
  팀 자체 숙제로 보류 중
- ~~CV 비교 원본 수치가 로컬에도 보존 안 됨~~ (정정: 실제로는 4개 공정 전부 로컬에 원시 CV 결과가
  남아있음 — `.gitignore` 대상이라 git엔 없을 뿐. `etch_simulator\results\`(isolation),
  `trench_engine\results\`(trench), `ml_engine/training/{gate,metal}/results/`(gate/metal)의
  `recipe_holdout_cv.csv`. 대표 수치는 §5-2에 옮겨 적어둠)
