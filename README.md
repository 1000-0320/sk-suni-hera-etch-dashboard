# Etch AI Decision Support System

반도체 Etch 공정(**isolation**, **trench**) 레시피를 입력하면, 학습된 AI 모델이
① 예상 웨이퍼 품질 예측 → ② 목표 스펙 대비 점수화 → ③ 파라미터 조정 추천까지 해주는 대시보드.

- 배포 링크(현재 실행 중): https://etch-ai-dashboard-ya5v2zh6mugcc2g4noc5km.streamlit.app/
- 브랜치: `feature/ai-simulation-seunghyun` (원본 `main`은 건드리지 않음)

---

## 이 문서를 처음 보는 팀원을 위한 안내

GitHub 안 써봤어도 괜찮음. 위 배포 링크 누르면 코드 몰라도 바로 대시보드 씀.
"코드가 어떻게 생겼나 보고 싶다" 할 때만 아래 표 보고 파일 클릭하면 됨
(GitHub 저장소 페이지에서 폴더/파일 이름 클릭 → 내용 바로 보임, 설치 필요 없음).

저장소 주소: `github.com/sunic1616-lgtm/etch-AI-dashboard` → `feature/ai-simulation-seunghyun` 브랜치 선택

---

## 무엇을 했나 (한 줄 요약)

기존 대시보드는 화면만 있고 예측은 과거 데이터 조회 + 단순 보정 방식이었음.
여기에 실제로 학습시킨 RandomForest/XGBoost 모델을 붙여서 진짜 예측이 나오게 만듦.

- **isolation 공정**: 팀원(영진님)이 만든 예측 엔진을 그대로 재사용
- **trench 공정**: 같은 방법론으로 신규 구축 (파라미터 12개→20개, Stage 2개→4개, Depth 단위 Å→nm만 다름)
- 두 공정 모두 항목별로 개별 모델 학습 → 성능 좋은 모델 자동 선택 → 관측 안 된 장비조합은 예측 차단
- 파라미터 하나씩 바꿔보면서 점수가 어떻게 변하는지 계산해 "이 파라미터를 이렇게 조정하면 +점수" 추천까지 반영

---

## 파일이 어디 있고 무슨 역할인지

| 경로 | 역할 |
|---|---|
| `app.py` | 화면(UI) 전체. 공정 선택, 입력, 결과 표시 |
| `model.py` | `predict()` 등 — 화면 입력을 받아 AI 모델 호출하고 점수·추천 계산 |
| `data_utils.py` | 데이터 불러오기, 단위 변환, 공정별 설정값 |
| `ml_engine/isolation_core.py`, `ml_engine/trench_core.py` | 공정별 예측 코어 — 학습된 모델을 실제로 불러와 실행하는 부분 |
| `ml_engine/isolation_models/`, `ml_engine/trench_models/` | **학습 완료된 모델 파일**(`.joblib`) + 어떤 모델을 쓸지 적힌 `metadata.json` |
| `ml_engine/training/isolation/`, `ml_engine/training/trench/` | **모델을 학습시킨 코드 원본** (재현 가능) — 상세 설명은 [`ml_engine/training/README.md`](ml_engine/training/README.md) |
| `data/*.xlsx` | 배포된 앱이 별도 업로드 없이 바로 동작하도록 넣어둔 기본 데이터 |
| `requirements.txt` | 실행에 필요한 파이썬 패키지 목록 |

> "AI 모델 어떻게 만들었는지 코드로 보여달라" 하면 → `ml_engine/training/` 폴더 + 그 안 README 보여주면 됨.
> "대시보드에 어떻게 연결됐는지 보여달라" 하면 → `model.py` → `ml_engine/isolation_core.py`/`trench_core.py` 순서로 보여주면 됨.

---

## 로컬에서 직접 실행하는 법

```bash
git clone https://github.com/sunic1616-lgtm/etch-AI-dashboard.git
cd etch-AI-dashboard
git checkout feature/ai-simulation-seunghyun
pip install -r requirements.txt
streamlit run app.py
```

`localhost:8501`에서 열림.

---

## 배포 방식 (Streamlit Cloud) — 코치님께 확인 필요한 사항

현재 [Streamlit Community Cloud](https://streamlit.io/cloud)로 배포되어 있음.

**Vercel로 옮기는 건 지금 바로는 어려움** — 기술적 구조가 안 맞기 때문:

- Streamlit은 화면이 켜져 있는 동안 서버가 계속 상태를 들고 있어야 하는 방식(상시 서버 + WebSocket).
- Vercel은 요청 올 때만 잠깐 켜졌다 꺼지는 서버리스 함수 방식이라, 이 둘이 구조적으로 안 맞음.
- Vercel에서 하려면 지금 Streamlit 화면을 통째로 버리고 React/Next.js로 새로 만들어야 함 (규모가 큰 재작업, 하루 안에 불가).

→ 내일 회의에서 코치님께: "Vercel은 Streamlit 구조와 안 맞아 정식 지원이 안 되는데, 그래도 옮기길 원하시는지 / 지금처럼 Streamlit Cloud 유지해도 되는지" 확인 필요.

---

## 아직 안 한 것

- `recipe master.csv` 원본 파일은 용량/필요성 문제로 레포에 아직 안 넣음 (학습 스크립트 재현 시 각자 경로 지정 필요)
- `main` 브랜치로 합치는 PR은 아직 안 올림 (지금은 `feature/ai-simulation-seunghyun` 브랜치 상태)
