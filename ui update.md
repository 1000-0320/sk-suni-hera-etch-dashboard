# UI Update

- 업데이트 날짜: 2026-08-10
- 대상 브랜치: `feature/dashboard-youngjin`
- 대상 시스템: Etch AI Decision Support System
- 배포 상태: 로컬 검증 중 (미배포)

## 기존 UI 개선

- 회사 공정 시뮬레이션용 로그인 화면과 DEMO/PRODUCTION 인증 전환 자리를 추가했습니다.
- 로그인 왼쪽 이미지는 `assets/login/semiconductor-etch-hero.png` 파일을 교체해 변경할 수 있습니다.
- 흰색 배경과 코랄·오렌지 포인트 컬러를 적용하고 카드, 버튼, 입력창, 사이드바의 디자인을 통일했습니다.
- Active Process를 헤더 아래에 배치하고 Process Dashboard의 Active View가 스크롤 중에도 보이도록 구성했습니다.
- Wafer 단면 Profile을 2×2 레이아웃으로 배치했습니다.
- 최근 품질 평가를 현재 로그인 세션에서 최대 5건까지 보관하며, 카드 내부 스크롤로 최근 기록을 확인할 수 있습니다.
- 최근 평가 카드에 Recipe, 변경 Parameter, 품질 점수, Pass Rate, CD/Depth 균일도, Defect를 표시합니다.

## 팀원 브랜치에서 반영한 분석 기능

`feature/dashboard-yunseo`의 기능 중 아래 다섯 항목만 현재 UI 구조에 맞춰 옮겼습니다.

1. **Rev별 품질 변화 그래프**
   - 실제 Wafer 측정 데이터를 기준으로 CD, Depth, 균일도, Pass Rate, Defect의 Rev별 변화를 표시합니다.
   - 선택한 단일 Recipe 필터와 무관하게 같은 공정·장비·Chamber의 Rev들을 비교합니다.
2. **개선된 Wafer 단면 Profile**
   - Extreme Edge → Center → Extreme Edge 순서와 Zone 배경 구분을 적용했습니다.
   - 축 표기를 간소화하고 Hover 정보에 Wafer와 Zone 정보를 추가했습니다.
3. **목표 대비 진단 카드**
   - 목표 대비 종합점수, 미달 품질 항목, 가장 취약한 Zone을 한눈에 표시합니다.
4. **파라미터별 추천 이유**
   - 각 추천값이 기존 예측 대비 어떤 지표를 개선하는지 표에 표시합니다.
   - 추천 순위와 방향, 현재값, 제안값, 목표점수 개선폭을 함께 제공합니다.
5. **상위 추천안 조합 및 재예측**
   - 상위 추천 파라미터를 동시에 적용한 전체 Recipe를 구성하고 기존 `predict()` 경로로 다시 예측합니다.
   - 현재 Recipe와 조합 Recipe의 품질 점수, Pass Rate, 균일도, Particle 예측 결과를 비교합니다.

## 로직 보호 범위

- 기존 모델 파일, 학습 데이터, 후보 탐색 방식, 추천 정렬 기준, 종합점수 계산식은 변경하지 않았습니다.
- `ml_engine/isolation_core.py`와 `ml_engine/trench_core.py`에는 화면 설명에 필요한 기존 계산 결과를 반환값으로 노출하는 변경만 적용했습니다.
- 조합 검증은 새 모델이나 별도 점수식을 사용하지 않고 기존 `predict()`를 다시 호출합니다.

## 로컬 검증

- Python 구문 검사 통과
- Git diff 공백 오류 검사 통과
- Isolation/Trench 두 공정의 Base Recipe 예측, 추천 5건, 추천 이유, 조합 Recipe 재예측 확인
- Rev 품질 변화 차트 5개와 Wafer Profile 4개 생성 확인
- Streamlit 로그인, Process Dashboard, 목표 진단, 추천 표, 조합 결과 화면 확인
- 배포·푸시는 사용자 승인 전까지 진행하지 않음
