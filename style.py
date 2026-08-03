"""
Etch AI Decision Support System - 디자인/스타일 모듈
- Toss 스타일(연회색 배경 + 흰 카드 + 파란 포인트 + 큰 둥근 모서리 + 넉넉한 여백)을 참고한 공통 CSS
- 카드/상태뱃지/점수 히어로 렌더링 헬퍼
- 품질 지표 색상 판정(양호/주의/위험) 로직

차트 시리즈 색상(series1~3, good/warning/critical)은 접근성 검증된 값을 그대로 유지하고,
배경/텍스트/포인트 컬러 등 UI 크롬만 Toss 톤으로 조정했다.
"""

import streamlit as st

# ----------------------------------------------------------------------------
# 공통 색상 팔레트
# ----------------------------------------------------------------------------
COLORS = {
    "surface": "#ffffff",
    "page": "#f2f4f6",           # Toss 특유의 연회색 배경
    "text_primary": "#191f28",
    "text_secondary": "#4e5968",
    "muted": "#8b95a1",
    "gridline": "#e5e8eb",
    "accent": "#3182f6",         # Toss 블루
    "accent_soft": "#e8f3ff",
    "series1": "#2a78d6",        # 차트 시리즈(접근성 검증된 값 유지)
    "series2": "#eb6834",
    "series3": "#1baf7a",
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
}

# 품질 상태(🟢양호 / 🟡주의 / 🔴위험) 표시용 메타데이터
STATUS_META = {
    "good": {"emoji": "🟢", "label": "양호", "color": COLORS["good"], "bg": "rgba(12,163,12,0.08)"},
    "warn": {"emoji": "🟡", "label": "주의", "color": "#c98500", "bg": "rgba(250,178,25,0.14)"},
    "critical": {"emoji": "🔴", "label": "위험", "color": COLORS["critical"], "bg": "rgba(208,59,59,0.08)"},
}


def inject_custom_css():
    """앱 전역에 적용되는 커스텀 CSS(카드/버튼/타이틀 스타일)"""
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-family: "Pretendard", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        .stApp {{ background-color: {COLORS['page']}; }}
        .block-container {{ padding-top: 2rem; max-width: 1180px; }}

        .main-title {{
            color: {COLORS['text_primary']};
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0;
        }}
        .subtitle {{
            color: {COLORS['text_secondary']};
            margin-top: 0.2rem;
            margin-bottom: 1.4rem;
            font-size: 1rem;
        }}
        .section-title {{
            color: {COLORS['text_primary']};
            font-weight: 800;
            font-size: 1.15rem;
            letter-spacing: -0.01em;
            margin: 1.6rem 0 0.7rem 0;
        }}
        .section-caption {{
            color: {COLORS['muted']};
            font-size: 0.82rem;
            margin-top: -0.3rem;
            margin-bottom: 0.8rem;
        }}

        div[data-testid="stButton"] button {{
            border-radius: 14px;
            font-weight: 700;
            padding: 0.6rem 1rem;
        }}
        div[data-testid="stButton"] button[kind="primary"] {{
            background-color: {COLORS['accent']};
            border: none;
        }}

        section[data-testid="stSidebar"] {{
            background-color: {COLORS['surface']};
            border-right: 1px solid {COLORS['gridline']};
        }}

        div[data-testid="stExpander"] {{
            background-color: {COLORS['surface']};
            border: 1px solid {COLORS['gridline']};
            border-radius: 16px;
        }}

        /* 예측 결과 카드 (큰 숫자) */
        .metric-card {{
            background-color: {COLORS['surface']};
            border-radius: 20px;
            padding: 1.2rem 1rem;
            text-align: center;
            box-shadow: 0 2px 10px rgba(25,31,40,0.05);
        }}
        .metric-card .label {{ color: {COLORS['text_secondary']}; font-size: 0.85rem; font-weight: 700; }}
        .metric-card .value {{ color: {COLORS['accent']}; font-size: 2.2rem; font-weight: 800; margin: 0.2rem 0; letter-spacing: -0.02em; }}
        .metric-card .unit {{ color: {COLORS['muted']}; font-size: 0.78rem; }}

        /* 품질 지표 상태 카드 */
        .status-card {{
            border-radius: 20px;
            padding: 1.1rem;
            box-shadow: 0 2px 10px rgba(25,31,40,0.05);
        }}
        .status-card .label {{ color: {COLORS['text_secondary']}; font-size: 0.85rem; font-weight: 700; }}
        .status-card .value {{ font-size: 1.6rem; font-weight: 800; margin: 0.2rem 0; color: {COLORS['text_primary']}; }}
        .status-card .status-line {{ font-size: 0.85rem; font-weight: 800; }}

        /* AI 분석 / 이슈 카드 */
        .analysis-card {{
            background-color: {COLORS['surface']};
            border-left: 4px solid {COLORS['accent']};
            border-radius: 18px;
            padding: 1.1rem 1.3rem;
            line-height: 2;
            color: {COLORS['text_primary']};
            box-shadow: 0 2px 10px rgba(25,31,40,0.05);
        }}

        /* Process Summary / 목표 대비 오차 카드 */
        .summary-card {{
            background-color: {COLORS['surface']};
            border-radius: 18px;
            padding: 0.95rem;
            text-align: center;
            box-shadow: 0 2px 10px rgba(25,31,40,0.05);
        }}
        .summary-card .label {{ color: {COLORS['text_secondary']}; font-size: 0.78rem; font-weight: 700; }}
        .summary-card .value {{ color: {COLORS['text_primary']}; font-size: 1.35rem; font-weight: 800; margin-top: 0.25rem; }}

        /* 종합 품질 점수 히어로 */
        .score-hero {{
            background-color: {COLORS['surface']};
            border-radius: 24px;
            padding: 1.8rem 1.5rem;
            text-align: center;
            box-shadow: 0 4px 16px rgba(25,31,40,0.07);
        }}
        .score-hero .score-label {{ color: {COLORS['text_secondary']}; font-weight: 700; font-size: 0.95rem; }}
        .score-hero .score-value {{ font-size: 3.4rem; font-weight: 800; letter-spacing: -0.03em; margin: 0.3rem 0; }}
        .score-hero .score-sub {{ font-size: 0.85rem; font-weight: 700; }}

        /* 작은 서브 점수 카드 (Pass Rate/Uniformity/Target Proximity 기여도) */
        .subscore-card {{
            background-color: {COLORS['surface']};
            border-radius: 16px;
            padding: 0.9rem;
            text-align: center;
            box-shadow: 0 2px 10px rgba(25,31,40,0.05);
        }}
        .subscore-card .label {{ color: {COLORS['muted']}; font-size: 0.75rem; font-weight: 700; }}
        .subscore-card .value {{ font-size: 1.3rem; font-weight: 800; margin-top: 0.15rem; }}
        .subscore-card .weight {{ color: {COLORS['muted']}; font-size: 0.7rem; }}

        h3 {{ color: {COLORS['text_primary']}; font-weight: 800; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value, unit: str = ""):
    """큰 숫자 카드 (Top/Mid/Bottom CD, Depth 등)"""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="unit">{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_card(label: str, value):
    """Process Summary / 오차 카드"""
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(label: str, value_str: str, status_key: str):
    """색상(🟢🟡🔴)으로 상태를 표시하는 품질 지표 카드"""
    meta = STATUS_META[status_key]
    st.markdown(
        f"""
        <div class="status-card" style="background-color:{meta['bg']};">
            <div class="label">{label}</div>
            <div class="value">{value_str}</div>
            <div class="status-line" style="color:{meta['color']};">{meta['emoji']} {meta['label']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_hero(score_total: float, label: str = "종합 품질 점수"):
    """종합 품질 점수를 큰 숫자로 강조하는 히어로 카드"""
    status = get_score_status(score_total)
    meta = STATUS_META[status]
    st.markdown(
        f"""
        <div class="score-hero">
            <div class="score-label">{label}</div>
            <div class="score-value" style="color:{meta['color']};">{score_total:.1f}<span style="font-size:1.2rem;">점</span></div>
            <div class="score-sub" style="color:{meta['color']};">{meta['emoji']} {meta['label']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_subscore_card(label: str, value: float, weight_text: str):
    st.markdown(
        f"""
        <div class="subscore-card">
            <div class="label">{label}</div>
            <div class="value" style="color:{COLORS['accent']};">{value:.1f}</div>
            <div class="weight">{weight_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pill_card(label: str, ok: bool):
    """Pass/Fail 판정을 카드 형태로 표시 (목표 대비 만족 여부 등)"""
    color = COLORS["good"] if ok else COLORS["critical"]
    bg = "rgba(12,163,12,0.08)" if ok else "rgba(208,59,59,0.08)"
    text = "Pass" if ok else "Fail"
    icon = "✅" if ok else "❌"
    st.markdown(
        f"""
        <div class="summary-card" style="background-color:{bg};">
            <div class="label">{label}</div>
            <div class="value" style="color:{color};">{icon} {text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# 품질 지표 상태 판정 로직 (임계값은 추후 실제 Spec 기준에 맞게 조정 가능)
# ----------------------------------------------------------------------------
def get_pass_rate_status(value: float) -> str:
    """Overall Spec Pass Rate: 95↑ 양호 / 90~95 주의 / 90↓ 위험"""
    if value >= 95:
        return "good"
    elif value >= 90:
        return "warn"
    return "critical"


def get_cd_uniformity_status(value: float) -> str:
    """CD Uniformity(변동계수 CV%): 값이 낮을수록 균일(양호). 1.8↓ 양호 / 3.0↓ 주의 / 3.0↑ 위험
    (trench recipe 데이터셋 관측 범위 약 1.1~3.5%를 기준으로 설정한 근사 임계값)"""
    if value <= 1.8:
        return "good"
    elif value <= 3.0:
        return "warn"
    return "critical"


def get_depth_uniformity_status(value: float) -> str:
    """Depth Uniformity(변동계수 CV%): 값이 낮을수록 균일(양호). 8↓ 양호 / 15↓ 주의 / 15↑ 위험
    (trench recipe 데이터셋 관측 범위 약 2.5~30.5%를 기준으로 설정한 근사 임계값)"""
    if value <= 8:
        return "good"
    elif value <= 15:
        return "warn"
    return "critical"


def get_particle_status(is_particle: bool) -> str:
    """Particle: False 양호 / True 위험"""
    return "critical" if is_particle else "good"


def get_score_status(value: float) -> str:
    """종합 품질 점수: 85↑ 양호 / 70~85 주의 / 70↓ 위험"""
    if value >= 85:
        return "good"
    elif value >= 70:
        return "warn"
    return "critical"
