# UI Update

업데이트 날짜: 2026-08-10

대상 브랜치: `feature/dashboard-youngjin`

대상 앱: Etch AI Decision Support System

## 업데이트 목적

공모전 시연과 사용성 개선을 위해 기존 Streamlit 공정 시뮬레이션 시스템의 화면 구조와 디자인을 정리했습니다. 예측 모델, 품질 계산식, 추천 알고리즘과 원본 데이터는 변경하지 않았습니다.

## 주요 변경 사항

### 1. 로그인 화면

- 회사 공정 시뮬레이션 시스템용 로그인 화면을 추가했습니다.
- 부서 선택, 아이디, 비밀번호, 로그인 버튼으로 구성했습니다.
- 회원가입 기능은 포함하지 않았습니다.
- DEMO/PRODUCTION 인증 모드를 분리하고, 향후 실제 계정 정보를 Streamlit Secrets에 등록할 수 있도록 예시 파일을 추가했습니다.
- 로그인 왼쪽 이미지는 `assets/login/semiconductor-etch-hero.png`만 교체하면 변경할 수 있습니다.

### 2. 전체 UI 테마

- 흰색 배경을 유지하면서 코랄·오렌지 계열의 포인트 색상을 적용했습니다.
- 헤더, 탭, 버튼, 카드, 입력창과 사이드바의 간격·모서리·그림자를 통일했습니다.
- 이모지 스티커를 미니멀한 선형 아이콘과 화살표로 교체했습니다.
- 실행 버튼은 `문구 →` 형식으로 통일했습니다.
- Process Dashboard의 강한 빨간색 그래프는 눈에 편한 파스텔 계열로 조정했습니다.

### 3. 화면 구조와 가독성

- Active Process 영역을 Decision Support Center 아래에 배치했습니다.
- Process Dashboard의 Active View를 스크롤 중에도 확인할 수 있도록 정리했습니다.
- Wafer 단면 Profile 그래프를 2×2 레이아웃으로 변경했습니다.
- Dashboard 섹션 제목과 아이콘의 색상·형태를 통일했습니다.

### 4. Recent Quality

- 최근 품질 평가 기록을 현재 로그인 세션에서 최대 5건까지 보관합니다.
- 다른 사용자가 로그인하거나 앱이 재시작되면 기록이 초기화됩니다.
- 카드 영역은 약 2건 높이로 유지하고, 추가 기록은 카드 내부에서 스크롤됩니다.
- Recipe, 변경된 공정 변수, 종합 품질 점수, Pass Rate, CD/Depth 균일도와 Defect를 표시합니다.
- 최신 결과와 직전 결과의 차이를 함께 표시합니다.
- 기록이 1건일 때 지표 HTML이 코드 문자열로 노출되던 문제를 수정했습니다.

### 5. 배포 준비

- Streamlit 테마 설정과 앱 상태 정보를 정리했습니다.
- 배포일은 현재 `미정`으로 표시합니다.
- `.streamlit/secrets.toml.example`에 향후 실제 계정 설정 위치를 마련했습니다.
- 실제 비밀번호나 비밀값은 저장소에 포함하지 않습니다.

## 변경하지 않은 영역

다음 핵심 로직과 데이터는 수정하지 않았습니다.

- `model.py`
- `ml_engine/`
- `data_utils.py`
- `data/`
- `requirements.txt`

## 확인 내용

- Python 구문 검사 통과
- Streamlit 로컬 실행 및 로그인 확인
- Isolation/Trench 공정 화면 전환 확인
- 예측·평가·추천 화면과 Process Dashboard 렌더링 확인
- Recent Quality의 1건·2건·여러 건 카드 표시 확인
- ML·데이터 보호 파일의 Git 변경 사항 없음 확인
