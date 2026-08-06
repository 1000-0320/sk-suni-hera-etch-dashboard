# 모델 학습 코드

`ml_engine/isolation_models/`, `ml_engine/trench_models/`에 있는 `.joblib` 모델과
`metadata.json`은 아래 스크립트들로 로컬에서 학습시킨 결과물입니다.

## 실행 순서

각 폴더(`isolation/`, `trench/`) 안에서 순서대로 실행합니다.

```powershell
python train_models.py --xlsx "<원본 xlsx 경로>" --recipe-csv "<recipe master csv 경로>"
python validate_unseen_recipes.py --xlsx "<원본 xlsx 경로>" --recipe-csv "<recipe master csv 경로>"
python select_deployment_models.py
python train_quality_v2.py --xlsx "<원본 xlsx 경로>" --recipe-csv "<recipe master csv 경로>"
```

`--xlsx`, `--recipe-csv`는 각자 로컬에 있는 원본 데이터 파일 경로로 바꿔서 실행해야 합니다
(레포에 포함된 `data/isolation_dataset.xlsx`, `data/trench_dataset.xlsx`가 xlsx 쪽에 해당하며,
recipe master csv는 별도 파일이라 이 레포에는 포함되어 있지 않습니다).

## 각 스크립트가 하는 일

- `train_models.py`: Top/Mid/Bottom CD, Depth, Particle 발생 여부를 RandomForest/XGBoost/MLP
  세 가지 방식으로 각각 학습시키고, `models/` 폴더에 `.joblib`로 저장합니다.
- `validate_unseen_recipes.py`: 학습에 안 쓴 Recipe로 교차검증(Recipe 단위로 통째로 빼고 검증)해서
  결과를 `results/recipe_holdout_cv.csv`에 남깁니다.
- `select_deployment_models.py`: 위 교차검증 결과를 보고, 항목마다 제일 성능이 좋은 모델을
  실제 배포용으로 선택해 `metadata.json`을 갱신합니다.
- `train_quality_v2.py`: Spec Pass 여부(4개)와 Defect Count/Severity를 추가로 학습시키고,
  Zone 단위 품질 점수 계산에 필요한 기준값을 `metadata.json`에 더합니다.

`simulator_core.py`는 `ml_engine/isolation_core.py` / `ml_engine/trench_core.py`와 같은 내용이며,
학습 스크립트가 참조하는 상수(파라미터 목록 등)를 그대로 쓰기 위해 이 폴더에도 같이 두었습니다.

## trench 상세 분석 문서

데이터 규모(Wafer/Site 수), 항목별 모델 선택 로직, 품질 점수 공식(대시보드용 vs 학습 파이프라인용
두 가지), Defect 정규화 수식까지 자세히 정리한 문서 → [`trench/ANALYSIS_SUMMARY.md`](trench/ANALYSIS_SUMMARY.md)

## isolation과 trench의 차이

trench 쪽 스크립트는 영진님이 isolation용으로 먼저 만들어두신 방식을 그대로 가져와서,
파라미터 개수(12개 → 20개, Stage 2개 → 4개)와 Depth 컬럼 단위(Å → nm)만 trench 데이터에
맞게 바꾼 것입니다. 학습 방식·검증 방식·품질 점수 계산 방식은 동일합니다.
