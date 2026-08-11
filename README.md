# Etch AI Decision Support System

반도체 Etch 공정(**isolation**, **trench**, **gate**, **metal**) 레시피를 입력하면, 학습된 AI 모델이
① 예상 웨이퍼 품질 예측 → ② 목표 스펙 대비 점수화 → ③ 파라미터 조정 추천까지 해주는 대시보드.

- 배포 링크(현재 실행 중): https://etch-ai-dashboard-2appgwew4wexbhrs9jqxcs.streamlit.app/
- 브랜치: `main` (모든 작업 브랜치가 fast-forward로 동일 커밋에 동기화되어 있음)
- **이 저장소에 문서가 여러 개라 헷갈리면 → [`DOCS_GUIDE.md`](DOCS_GUIDE.md)(문서 안내)부터 보세요**
- **전체 과정(배경/데이터/모델학습/품질점수/버그이력 등) 한번에 보려면 → [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md)**
- **발표 준비 자료(8/20 최종발표용)** → 기능 설명서 [`docs/FEATURE_GUIDE.md`](docs/FEATURE_GUIDE.md) ·
  PPT 콘텐츠 [`docs/PPT_CONTENT.md`](docs/PPT_CONTENT.md) · Q&A 대비 [`docs/QNA_PREP.md`](docs/QNA_PREP.md)

---

## 이 문서를 처음 보는 팀원을 위한 안내

GitHub 안 써봤어도 괜찮음. 위 배포 링크 누르면 코드 몰라도 바로 대시보드 씀.
"코드가 어떻게 생겼나 보고 싶다" 할 때만 아래 표 보고 파일 클릭하면 됨
(GitHub 저장소 페이지에서 폴더/파일 이름 클릭 → 내용 바로 보임, 설치 필요 없음).

저장소 주소: `github.com/sunic1616-lgtm/etch-AI-dashboard` → `main` 브랜치 선택

---

## 무엇을 했나 (한 줄 요약)

기존 대시보드는 화면만 있고 예측은 과거 데이터 조회 + 단순 보정 방식이었음.
여기에 실제로 학습시킨 RandomForest/XGBoost 모델을 붙여서 진짜 예측이 나오게 만듦.

- **isolation 공정**: 팀원(영진님)이 만든 예측 엔진을 그대로 재사용
- **trench 공정**: 같은 방법론으로 신규 구축 (파라미터 12개→20개, Stage 2개→4개, Depth 단위 Å→nm만 다름)
- **gate 공정**(Gate PolySi Etch), **metal 공정**(Metal Al Etch): 멘토님이 이후 추가로 준 3·4번째 공정 데이터로,
  동일 파이프라인 재사용해서 신규 학습 (둘 다 1-Stage 구조, 파라미터 5개, 가스 파라미터명만 다름)
- 네 공정 모두 항목별로 개별 모델 학습 → 성능 좋은 모델 자동 선택 → 관측 안 된 장비조합은 예측 차단
- 파라미터 하나씩 바꿔보면서 점수가 어떻게 변하는지 계산해 "이 파라미터를 이렇게 조정하면 +점수" 추천까지 반영

## 멘토 피드백(카카오 리뷰) 반영 사항

- 시뮬레이터를 **🎯 목표 품질 모드**(목표 스펙 → 파라미터 조정 추천)와 **📋 Recipe 직접 입력 모드**
  (Recipe 값 → 예상 품질 점수)로 분리. 기존엔 한 화면에 섞여 있어 헷갈렸음
- Chamber 선택 항목 제거 — "모든 Chamber가 동일하다"는 가정하에 장비별 대표 Chamber를 자동 사용
- 목표 품질 입력 라벨에 "기준 Recipe 값 → 목표"처럼 기준값을 같이 표시 (맨 숫자만 있던 것 개선)
- Process Dashboard에 **Rev별 스코어링 순위** 추가 — 실측 데이터 기반으로 지금까지 어떤 Recipe가
  가장 좋았는지 표+막대그래프로 바로 확인 가능
- Wafer Map에 **단면 Profile 차트**(Edge → Center → Edge) 추가 — 기존 2D 폴라 맵만으론 보기 어려웠던
  구간별 추이를 지표별로 보완

---

## 새로 추가한 기능: 학습 range 밖 입력값 가드

**문제:** RandomForest/XGBoost 같은 트리 기반 모델은 학습 때 본 값 범위를 벗어나면 예측이 "평평해짐"(saturate).
예를 들어 RF Bias를 학습 범위(180~220W) 훨씬 밖인 500W로 넣어도, 모델은 그냥 경계값 근처 예측을 그대로 뱉음.
일부러 극단값 넣어서 "나쁜 결과 나오나" 테스트했는데 여전히 멀쩡한 점수가 나오는 것처럼 보이는 이유가 이거임 —
모델이 잘못된 게 아니라, 애초에 학습 안 해본 구간이라 값을 신뢰할 수 없는 것.

**해결:** 예측 버튼 누르는 시점에, 무거운 모델 연산 들어가기 전에 입력값이 학습 range 안인지 먼저 검사함.

- 범위 밖이면 → 예측 자체를 실행하지 않고 어떤 파라미터가 얼마나 벗어났는지 바로 경고 (연산 낭비 안 함, 즉시 응답)
- 그래도 그 값으로 결과를 보고 싶으면 → **"그래도 이 값으로 실행"** 버튼으로 우회 가능. 이땐 실제로 예측은 돌아가되, 결과 화면에 "신뢰도 낮음" 경고를 같이 띄움
- 장비/챔버 조합 불일치(애초에 존재하지 않는 조합)는 이 우회가 안 됨 — 그건 값의 문제가 아니라 조합 자체가 없는 경우라 원천 차단 유지

isolation·trench·gate·metal 4개 공정 모두 동일하게 적용됨. 관련 코드: `ml_engine/{isolation,trench,gate,metal}_core.py`의
`validate_parameter_ranges()`, `model.py`의 `predict()`, `app.py`의 예측 버튼 처리 부분.

---

## 파일이 어디 있고 무슨 역할인지

| 경로 | 역할 |
|---|---|
| `app.py` | 화면(UI) 전체. 공정 선택, 입력, 결과 표시 |
| `model.py` | `predict()` 등 — 화면 입력을 받아 AI 모델 호출하고 점수·추천 계산 |
| `data_utils.py` | 데이터 불러오기, 단위 변환, 공정별 설정값 |
| `ml_engine/{isolation,trench,gate,metal}_core.py` | 공정별 예측 코어 — 학습된 모델을 실제로 불러와 실행하는 부분 |
| `ml_engine/{isolation,trench,gate,metal}_models/` | **학습 완료된 모델 파일**(`.joblib`) + 어떤 모델을 쓸지 적힌 `metadata.json` |
| `ml_engine/training/{isolation,trench,gate,metal}/` | **모델을 학습시킨 코드 원본** (재현 가능) — 상세 설명은 [`ml_engine/training/README.md`](ml_engine/training/README.md) |
| `data/*.xlsx` | 배포된 앱이 별도 업로드 없이 바로 동작하도록 넣어둔 기본 데이터 |
| `requirements.txt` | 실행에 필요한 파이썬 패키지 목록 |

> "AI 모델 어떻게 만들었는지 코드로 보여달라" 하면 → `ml_engine/training/` 폴더 + 그 안 README 보여주면 됨.
> "대시보드에 어떻게 연결됐는지 보여달라" 하면 → `model.py` → `ml_engine/{isolation,trench,gate,metal}_core.py` 순서로 보여주면 됨.

---

## 로컬에서 직접 실행하는 법

```bash
git clone https://github.com/sunic1616-lgtm/etch-AI-dashboard.git
cd etch-AI-dashboard
git checkout main
pip install -r requirements.txt
streamlit run app.py
```

`localhost:8501`에서 열림.

---

## 배포 방식 (Streamlit Cloud)

현재 [Streamlit Community Cloud](https://streamlit.io/cloud)로 배포되어 있음.

### 로그인 모드와 Secrets 설정

기본값은 공모전 제출용 `demo` 모드다. 실제 계정 검증이 필요해지면 Streamlit Community Cloud의
**App settings → Secrets**에 [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) 형식으로
값을 등록하고 `auth_mode = "production"`으로 변경한다. 실제 아이디와 비밀번호는 Git에 커밋하지 않는다.

```toml
[app]
auth_mode = "production"

[auth]
admin_user = "실제_아이디"
admin_password = "실제_비밀번호"
```

**Vercel로 옮기는 건 기술적 구조가 안 맞음** (상시 서버+WebSocket 방식 vs Vercel의 서버리스 방식) —
코치님께 확인 결과, 지금처럼 Streamlit Cloud 유지하는 걸로 확정.

---

## 아직 안 한 것

- `recipe master.csv` 원본 파일은 다른 Excel 파일(`data/*.xlsx`)에 내용이 이미 포함돼 있어서 별도로 안 올림
