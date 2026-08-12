# 코치님 공유용 — 대시보드 업데이트 요약

배포 링크: https://etch-ai-dashboard-2appgwew4wexbhrs9jqxcs.streamlit.app/?auth=1

---

## 뭘 만들었나 (한 줄 요약)

Etch 공정(**isolation**, **trench**, **gate**, **metal**) Recipe를 입력하면 실제 학습된 AI 모델(RandomForest/XGBoost)이
예상 품질을 예측하고, 목표 대비 점수화 → 최적 Recipe/파라미터 조정까지 추천해주는 대시보드입니다.

## 주요 기능

1. **시뮬레이터 두 모드로 분리 (멘토 피드백 반영)**
   - 🎯 목표 품질 모드 — 원하는 목표 스펙 입력 → 기준 Recipe 대비 "이 파라미터를 이렇게 바꾸면 좋아짐" 추천
   - 📋 Recipe 직접 입력 모드 — Recipe 값을 그대로 넣고 예상 품질 점수 확인
   (기존엔 한 화면에 섞여 있던 걸 목적별로 분리)
2. **품질 예측** — Stage별 Time/RF Bias/Pressure/Gas Flow 입력 → Top/Mid/Bottom CD, Depth, Uniformity,
   Pass Rate, Defect까지 실제 학습된 모델로 예측
3. **목표 대비 평가** — 설정한 목표 스펙과 비교해 종합 품질 점수 산출, 미달 항목을 우선순위별로 정리
4. **최적 Recipe 추천** — 알려진 Recipe들 중 점수가 가장 높은 것을 자동 추천 + 비교표
   (Chamber는 "모든 Chamber가 동일하다"는 가정으로 대표값 자동 선택 — 멘토 피드백으로 선택 항목에서 제거)
5. **파라미터별 조정 제안 (AI 추천)** — 기존 Recipe가 아니라 지금 입력한 값을 출발점으로, 파라미터를 하나씩
   바꿔보며 품질이 좋아지는 방향을 찾아 제안 (One-Factor-at-a-Time)
6. **학습 range 밖 입력값 가드** — AI 모델이 한 번도 안 본 극단값을 넣으면, 무거운 연산 전에 미리
   막고 어떤 파라미터가 얼마나 벗어났는지 표로 보여줌. 그래도 그 값으로 결과를 보고 싶으면 "그래도 이 값으로
   실행" 버튼으로 우회 가능 (이땐 결과에 "신뢰도 낮음" 경고가 같이 뜸). 극단값 넣고 테스트할 때 불필요한
   대기 시간을 줄이기 위한 기능입니다.
7. **Rev별 스코어링 순위 (신규)** — Process Dashboard에서, 모델 예측이 아니라 실제 측정된 Wafer 결과를
   Recipe(Rev)별로 집계해 종합 품질 점수 표+막대그래프로 보여줌. 지금까지 어떤 Recipe가 제일 잘 나왔는지
   한눈에 확인 가능
8. **Wafer 단면 Profile 차트 (신규)** — 기존 2D 폴라 Wafer Map만으론 파악하기 어려웠던 부분을 보완:
   Point 번호를 Edge → Center → Edge 순서로 배치한 단면 그래프를 지표별로 추가 제공
9. **3·4번째 공정 추가 (Gate PolySi Etch, Metal Al Etch)** — isolation/trench와 동일한 파이프라인으로
   신규 학습, 공정 선택기에서 4개 공정 전부 전환 가능
10. **Parameter 변경 이력 조회 (신규 탭)** — Rev별로 어떤 파라미터가 얼마나 바뀌었는지 이력을 표로 확인
11. **신규 공정 품질 예측 (신규 탭)** — 아직 학습 데이터가 없는 새 Layer 조합을 Layer 순서대로 입력하면,
    가장 비슷한 기존 공정의 실측 평균을 참고값으로 보여줌 (실제 AI 예측 아님, 참고용 근사치로 명시 표기)

## 배포 방식

[Streamlit Community Cloud](https://streamlit.io/cloud)로 배포 중입니다 (코치님과 상의 후 유지하기로 확정).
GitHub 브랜치에 코드를 올리면 자동으로 반영됩니다.

## 코드가 궁금하시면

저장소: `github.com/sunic1616-lgtm/etch-AI-dashboard` (`main` 브랜치)
자세한 파일 구조와 실행 방법은 저장소 [`README.md`](README.md)에 정리해뒀습니다.
