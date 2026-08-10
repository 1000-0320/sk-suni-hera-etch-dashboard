"""
Etch AI Decision Support System - 디자인/스타일 모듈
- 선택 가능한 공통 UI 테마와 카드/상태 컴포넌트 CSS
- 카드/상태뱃지/점수 히어로 렌더링 헬퍼
- 품질 지표 색상 판정(양호/주의/위험) 로직

good/warning/critical은 품질 판독을 위해 테마와 무관한 의미 색상으로 유지한다.
"""

from __future__ import annotations

import streamlit as st

# ----------------------------------------------------------------------------
# 공통 색상 팔레트
# ----------------------------------------------------------------------------
# 롤백: 아래 값을 "classic_blue"로 바꾸고 Streamlit을 재시작하면 기존 색상으로 복원된다.
ACTIVE_THEME = "racing_coral"

THEMES = {
    "classic_blue": {
        "surface": "#ffffff",
        "surface_soft": "#f8fafc",
        "control_bg": "#f4f6f9",
        "page": "#f5f7fb",
        "text_primary": "#141b2d",
        "text_secondary": "#536176",
        "muted": "#8a96a8",
        "gridline": "#e7ebf1",
        "accent": "#1f6ff0",
        "accent_soft": "#eaf2ff",
        "accent_secondary": "#51b9ff",
        "sidebar_start": "#071a35",
        "sidebar_end": "#123967",
        "sidebar_dark": "#071a35",
        "sidebar_shadow": "rgba(7,26,53,0.16)",
        "avatar_soft": "rgba(75,168,255,0.2)",
        "active_dot": "#21b582",
        "active_dot_soft": "rgba(33,181,130,0.1)",
        "history_border": "rgba(76,164,255,0.28)",
        "history_fill": "rgba(43,126,226,0.1)",
        "history_accent": "#8ecbff",
        "history_badge": "#7cc6ff",
        "history_badge_bg": "rgba(64,156,255,0.14)",
        "history_scroll": "rgba(124,198,255,0.42)",
        "shadow_soft": "rgba(20,27,45,0.05)",
        "shadow_card": "rgba(25,31,40,0.05)",
        "shadow_hero": "rgba(25,31,40,0.07)",
        "series1": "#2a78d6",
        "series2": "#eb6834",
        "series3": "#1baf7a",
        "chart_good": "#0ca30c",
        "chart_warning": "#fab219",
        "chart_critical": "#d03b3b",
        "wafer_colorscale": "Blues",
        "good": "#0ca30c",
        "warning": "#fab219",
        "critical": "#d03b3b",
    },
    "racing_coral": {
        "surface": "#ffffff",
        "surface_soft": "#fafafa",
        "control_bg": "#f7f3f1",
        "page": "#ffffff",
        "text_primary": "#171717",
        "text_secondary": "#4b4543",
        "muted": "#817875",
        "gridline": "#ebe7e5",
        "accent": "#e5483b",
        "accent_soft": "#ffe1cf",
        "accent_secondary": "#f47a3d",
        "sidebar_start": "#47211e",
        "sidebar_end": "#6d342d",
        "sidebar_dark": "#47211e",
        "sidebar_shadow": "rgba(71,33,30,0.18)",
        "avatar_soft": "rgba(244,122,61,0.22)",
        "active_dot": "#f47a3d",
        "active_dot_soft": "rgba(244,122,61,0.14)",
        "history_border": "rgba(244,122,61,0.34)",
        "history_fill": "rgba(244,122,61,0.12)",
        "history_accent": "#ffb184",
        "history_badge": "#ffc29a",
        "history_badge_bg": "rgba(244,122,61,0.18)",
        "history_scroll": "rgba(255,177,132,0.56)",
        "shadow_soft": "rgba(23,23,23,0.05)",
        "shadow_card": "rgba(23,23,23,0.06)",
        "shadow_hero": "rgba(23,23,23,0.08)",
        "series1": "#7899b8",
        "series2": "#73ad9f",
        "series3": "#a799c4",
        "chart_good": "#78a98e",
        "chart_warning": "#d7ad68",
        "chart_critical": "#c78383",
        "wafer_colorscale": [[0.0, "#f2f7f8"], [0.52, "#9fc7c4"], [1.0, "#617f9d"]],
        "good": "#0ca30c",
        "warning": "#fab219",
        "critical": "#d03b3b",
    },
}

COLORS = THEMES[ACTIVE_THEME]

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
        .stApp {{
            --primary-color: {COLORS['accent']};
            background-color: {COLORS['page']};
            color: {COLORS['text_primary']};
        }}
        [data-testid="stAppDeployButton"],
        [data-testid="stMainMenu"],
        #MainMenu,
        footer {{
            display: none !important;
        }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        .block-container {{ padding-top: 3.6rem; padding-bottom: 4rem; max-width: 1320px; }}

        /* 로그인 화면의 테마 톤을 이어받는 앱 헤더 */
        .app-shell-header {{ padding: 0 0 0.9rem; }}
        .app-shell-eyebrow {{
            color: {COLORS['accent']};
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.16em;
            margin-bottom: 0.55rem;
        }}
        .app-shell-title {{
            color: {COLORS['text_primary']};
            font-size: clamp(2rem, 3.3vw, 3rem);
            line-height: 1.05;
            font-weight: 850;
            letter-spacing: -0.055em;
            margin: 0;
        }}
        .app-shell-subtitle {{
            color: {COLORS['text_secondary']};
            margin-top: 0.75rem;
            font-size: 0.95rem;
        }}

        div[data-testid="stColumn"]:has(.process-selector-anchor) {{
            align-self: center;
            background: {COLORS['surface']};
            border: 1px solid {COLORS['gridline']};
            border-radius: 18px;
            padding: 0.9rem 1rem 0.75rem;
            box-shadow: 0 8px 24px {COLORS['shadow_soft']};
        }}
        .process-selector-anchor {{
            color: {COLORS['muted']};
            font-size: 0.66rem;
            font-weight: 850;
            letter-spacing: 0.14em;
            margin-bottom: 0.35rem;
        }}
        div[data-testid="stColumn"]:has(.process-selector-anchor) div[role="radiogroup"] {{
            gap: 0.35rem;
        }}
        div[data-testid="stColumn"]:has(.process-selector-anchor) div[role="radiogroup"] label {{
            background: {COLORS['control_bg']};
            border-radius: 10px;
            padding: 0.42rem 0.62rem;
            margin: 0;
        }}
        div[data-testid="stColumn"]:has(.process-selector-anchor) div[role="radiogroup"] label:has(input:checked) {{
            background: {COLORS['accent_soft']};
            color: {COLORS['accent']};
        }}
        label:has(input[type="radio"]:checked) > div:first-of-type > div > div:first-child {{
            background-color: {COLORS['accent']} !important;
        }}

        /* 시뮬레이터 모드: B안 미니멀 타깃 / 슬라이더 라인 아이콘 */
        .simulator-mode-anchor {{ display: none; }}
        div[data-testid="stElementContainer"]:has(.simulator-mode-anchor) + div[data-testid="stElementContainer"] div[role="radiogroup"] {{
            gap: 0.55rem 1.35rem;
        }}
        div[data-testid="stElementContainer"]:has(.simulator-mode-anchor) + div[data-testid="stElementContainer"] div[role="radiogroup"] label {{
            margin: 0;
        }}
        div[data-testid="stElementContainer"]:has(.simulator-mode-anchor) + div[data-testid="stElementContainer"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] {{
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
        }}
        div[data-testid="stElementContainer"]:has(.simulator-mode-anchor) + div[data-testid="stElementContainer"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"]::before {{
            content: "";
            display: inline-block;
            width: 1.22rem;
            height: 1.22rem;
            flex: 0 0 1.22rem;
            background-color: {COLORS['accent']};
        }}
        div[data-testid="stElementContainer"]:has(.simulator-mode-anchor) + div[data-testid="stElementContainer"] div[role="radiogroup"] label:nth-child(1) [data-testid="stMarkdownContainer"]::before {{
            -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Ccircle cx='12' cy='12' r='8.5' fill='none' stroke='black' stroke-width='1.6'/%3E%3Ccircle cx='12' cy='12' r='4.5' fill='none' stroke='black' stroke-width='1.6'/%3E%3Ccircle cx='12' cy='12' r='1.4' fill='black'/%3E%3C/svg%3E") center / contain no-repeat;
            mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Ccircle cx='12' cy='12' r='8.5' fill='none' stroke='black' stroke-width='1.6'/%3E%3Ccircle cx='12' cy='12' r='4.5' fill='none' stroke='black' stroke-width='1.6'/%3E%3Ccircle cx='12' cy='12' r='1.4' fill='black'/%3E%3C/svg%3E") center / contain no-repeat;
        }}
        div[data-testid="stElementContainer"]:has(.simulator-mode-anchor) + div[data-testid="stElementContainer"] div[role="radiogroup"] label:nth-child(2) [data-testid="stMarkdownContainer"]::before {{
            -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M4 7h16M4 12h16M4 17h16' fill='none' stroke='black' stroke-width='1.7' stroke-linecap='round'/%3E%3Ccircle cx='9' cy='7' r='1.7' fill='white' stroke='black' stroke-width='1.5'/%3E%3Ccircle cx='15' cy='12' r='1.7' fill='white' stroke='black' stroke-width='1.5'/%3E%3Ccircle cx='8' cy='17' r='1.7' fill='white' stroke='black' stroke-width='1.5'/%3E%3C/svg%3E") center / contain no-repeat;
            mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M4 7h16M4 12h16M4 17h16' fill='none' stroke='black' stroke-width='1.7' stroke-linecap='round'/%3E%3Ccircle cx='9' cy='7' r='1.7' fill='white' stroke='black' stroke-width='1.5'/%3E%3Ccircle cx='15' cy='12' r='1.7' fill='white' stroke='black' stroke-width='1.5'/%3E%3Ccircle cx='8' cy='17' r='1.7' fill='white' stroke='black' stroke-width='1.5'/%3E%3C/svg%3E") center / contain no-repeat;
        }}

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
            border-radius: 12px;
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
        section[data-testid="stSidebar"] > div {{ padding-top: 1.15rem; }}
        section[data-testid="stSidebar"] .stButton button {{ min-height: 2.6rem; }}

        .sidebar-user-card {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.8rem;
            margin-bottom: 0.55rem;
            background: linear-gradient(145deg, {COLORS['sidebar_start']} 0%, {COLORS['sidebar_end']} 100%);
            border-radius: 16px;
            color: #ffffff;
        }}
        .sidebar-user-avatar {{
            width: 2.35rem;
            height: 2.35rem;
            flex: 0 0 2.35rem;
            display: grid;
            place-items: center;
            border-radius: 12px;
            background: {COLORS['avatar_soft']};
            border: 1px solid rgba(255, 255, 255, 0.2);
            font-weight: 850;
        }}
        .sidebar-user-name {{ font-size: 0.84rem; font-weight: 800; }}
        .sidebar-user-role {{ color: rgba(255,255,255,0.62); font-size: 0.69rem; margin-top: 0.12rem; }}
        .sidebar-section-label {{
            color: {COLORS['muted']};
            font-size: 0.64rem;
            font-weight: 850;
            letter-spacing: 0.14em;
            margin: 1.35rem 0 0.55rem;
        }}
        .sidebar-context-card,
        .sidebar-data-card {{
            border: 1px solid {COLORS['gridline']};
            background: {COLORS['surface_soft']};
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
        }}
        .sidebar-context-card .label,
        .sidebar-data-card .label {{ color: {COLORS['muted']}; font-size: 0.68rem; font-weight: 750; }}
        .sidebar-context-card .value,
        .sidebar-data-card .value {{ color: {COLORS['text_primary']}; font-size: 0.86rem; font-weight: 820; margin-top: 0.2rem; }}
        .sidebar-context-card .dot,
        .sidebar-data-card .dot {{
            display: inline-block;
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 50%;
            background: {COLORS['active_dot']};
            margin-right: 0.4rem;
            box-shadow: 0 0 0 4px {COLORS['active_dot_soft']};
        }}
        .sidebar-footnote {{ color: {COLORS['muted']}; font-size: 0.68rem; line-height: 1.55; margin-top: 0.65rem; }}
        .sidebar-system-card {{
            border: 1px solid {COLORS['gridline']};
            background: {COLORS['surface_soft']};
            border-radius: 14px;
            padding: 0.78rem 0.88rem;
        }}
        .sidebar-system-head,
        .sidebar-system-row {{ display: flex; align-items: center; justify-content: space-between; gap: 0.6rem; }}
        .sidebar-system-head {{
            color: {COLORS['text_primary']};
            font-size: 0.75rem;
            font-weight: 850;
            padding-bottom: 0.48rem;
            margin-bottom: 0.4rem;
            border-bottom: 1px solid {COLORS['gridline']};
        }}
        .sidebar-system-head strong {{ color: {COLORS['accent']}; font-size: 0.65rem; }}
        .sidebar-system-row {{ color: {COLORS['muted']}; font-size: 0.62rem; padding: 0.16rem 0; }}
        .sidebar-system-row strong {{ color: {COLORS['text_secondary']}; font-size: 0.62rem; }}
        .sidebar-system-row strong.is-ready {{ color: {COLORS['good']}; }}

        section[data-testid="stSidebar"] div[data-testid="stExpander"]:has(.quality-history-anchor) {{
            background: {COLORS['sidebar_dark']};
            border: 0;
            border-radius: 18px;
            box-shadow: 0 12px 28px {COLORS['sidebar_shadow']};
            overflow: hidden;
        }}
        section[data-testid="stSidebar"] div[data-testid="stExpander"]:has(.quality-history-anchor) summary {{
            background: {COLORS['sidebar_end']} !important;
            color: #ffffff !important;
            font-size: 0.76rem;
            font-weight: 820;
            min-height: 2.65rem;
            padding: 0.62rem 0.8rem !important;
            border-radius: 18px 18px 0 0;
        }}
        section[data-testid="stSidebar"] div[data-testid="stExpander"]:has(.quality-history-anchor) summary p,
        section[data-testid="stSidebar"] div[data-testid="stExpander"]:has(.quality-history-anchor) summary span {{
            color: #ffffff !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stExpander"]:has(.quality-history-anchor) summary svg {{
            color: #ffffff !important;
            fill: #ffffff !important;
            stroke: #ffffff !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stExpander"]:has(.quality-history-anchor) details {{
            background: {COLORS['sidebar_dark']} !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stExpander"]:has(.quality-history-anchor) [data-testid="stExpanderDetails"] {{
            max-height: 34.5rem;
            overflow-y: auto;
            overscroll-behavior: contain;
            scrollbar-gutter: stable;
            padding: 0 0.5rem 0.75rem 0.65rem;
        }}
        section[data-testid="stSidebar"] div[data-testid="stExpander"]:has(.quality-history-anchor) [data-testid="stExpanderDetails"]:has(.quality-history-anchor.has-multiple) {{
            height: 34.5rem;
        }}
        section[data-testid="stSidebar"] div[data-testid="stExpander"]:has(.quality-history-anchor) [data-testid="stExpanderDetails"]::-webkit-scrollbar {{
            width: 0.34rem;
        }}
        section[data-testid="stSidebar"] div[data-testid="stExpander"]:has(.quality-history-anchor) [data-testid="stExpanderDetails"]::-webkit-scrollbar-track {{
            background: rgba(255,255,255,0.04);
            border-radius: 999px;
        }}
        section[data-testid="stSidebar"] div[data-testid="stExpander"]:has(.quality-history-anchor) [data-testid="stExpanderDetails"]::-webkit-scrollbar-thumb {{
            background: {COLORS['history_scroll']};
            border-radius: 999px;
        }}
        .quality-history-anchor {{ display: none; }}
        .quality-history-item {{
            border-top: 1px solid rgba(255,255,255,0.1);
            padding: 0.8rem 0.15rem 0;
            margin-top: 0.55rem;
            color: #ffffff;
        }}
        .quality-history-item.latest {{
            border: 1px solid {COLORS['history_border']};
            border-radius: 13px;
            background: {COLORS['history_fill']};
            padding: 0.75rem;
        }}
        .quality-history-head {{ display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }}
        .quality-history-head strong {{ font-size: 0.72rem; }}
        .quality-history-head span {{ color: rgba(255,255,255,0.48); font-size: 0.6rem; }}
        .quality-history-meta {{ color: rgba(255,255,255,0.55); font-size: 0.62rem; margin-top: 0.24rem; }}
        .quality-history-change {{
            color: rgba(255,255,255,0.72);
            font-size: 0.59rem;
            line-height: 1.45;
            margin-top: 0.32rem;
        }}
        .quality-history-change strong {{
            color: {COLORS['history_accent']};
            font-size: 0.56rem;
            letter-spacing: 0.04em;
            margin-right: 0.3rem;
        }}
        .quality-history-change.baseline {{ color: rgba(255,255,255,0.5); }}
        .quality-history-change.baseline strong {{ color: rgba(255,255,255,0.7); }}
        .quality-history-meta .badge {{
            color: {COLORS['history_badge']};
            background: {COLORS['history_badge_bg']};
            border-radius: 999px;
            padding: 0.2rem 0.48rem;
            font-size: 0.58rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            margin-left: 0.25rem;
        }}
        .quality-history-score {{
            display: flex;
            align-items: baseline;
            gap: 0.3rem;
            padding: 0.64rem 0 0.45rem;
        }}
        .quality-history-score strong {{ font-size: 1.55rem; font-weight: 850; letter-spacing: -0.04em; }}
        .quality-history-score span {{ color: rgba(255,255,255,0.55); font-size: 0.62rem; }}
        .quality-history-delta-title {{
            color: rgba(255,255,255,0.5);
            font-size: 0.52rem;
            font-weight: 850;
            letter-spacing: 0.1em;
            margin: 0.1rem 0 0.38rem;
        }}
        .quality-history-delta-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.34rem 0.5rem;
            padding: 0.5rem 0.55rem;
            margin-bottom: 0.52rem;
            border-radius: 9px;
            background: rgba(255,255,255,0.055);
        }}
        .quality-history-delta-grid span {{ display: flex; flex-direction: column; gap: 0.08rem; }}
        .quality-history-delta-grid small {{ color: rgba(255,255,255,0.46); font-size: 0.52rem; }}
        .quality-history-delta-grid strong {{ font-size: 0.62rem; }}
        .quality-history-delta-grid .is-improved {{ color: #74d6ad; }}
        .quality-history-delta-grid .is-worse {{ color: #ff9c93; }}
        .quality-history-delta-grid .is-neutral {{ color: rgba(255,255,255,0.66); }}
        .quality-history-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.4rem 0.6rem;
            padding-top: 0.5rem;
            border-top: 1px solid rgba(255,255,255,0.09);
        }}
        .quality-history-grid span {{ color: rgba(255,255,255,0.5); font-size: 0.58rem; }}
        .quality-history-grid strong {{ display: block; color: #ffffff; font-size: 0.68rem; margin-top: 0.12rem; }}
        .quality-history-empty {{
            color: rgba(255,255,255,0.56);
            font-size: 0.68rem;
            line-height: 1.55;
            padding: 0.2rem 0.15rem 0.55rem;
        }}

        /* 기본 Streamlit 탭을 제품형 세그먼트 내비게이션으로 정리 */
        div[data-testid="stTabs"] [role="tablist"] {{
            gap: 0.35rem;
            background: {COLORS['surface']};
            border: 1px solid {COLORS['gridline']};
            border-radius: 16px;
            padding: 0.38rem;
            box-shadow: 0 6px 20px {COLORS['shadow_soft']};
        }}
        div[data-testid="stTabs"] [role="tab"] {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            border-radius: 11px;
            min-height: 2.8rem;
            padding: 0 1.05rem;
            font-weight: 750;
        }}
        div[data-testid="stTabs"] [role="tab"]::before,
        button[data-baseweb="tab"]::before {{
            content: "";
            width: 1.05rem;
            height: 1.05rem;
            flex: 0 0 1.05rem;
            background-color: {COLORS['accent']};
            -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M4 4l8 8-8 8M10 4l8 8-8 8' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") center / contain no-repeat;
            mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M4 4l8 8-8 8M10 4l8 8-8 8' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") center / contain no-repeat;
        }}
        div[data-testid="stTabs"] [role="tab"]:nth-child(2)::before,
        button[data-baseweb="tab"]:nth-child(2)::before {{
            background-color: {COLORS['accent']};
        }}
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
            background: {COLORS['accent_soft']};
            color: {COLORS['accent']} !important;
            border-bottom-color: {COLORS['accent']} !important;
        }}
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"] > div:first-child {{
            color: {COLORS['accent']} !important;
        }}
        div[data-testid="stTabs"] [role="tab"] .react-aria-SelectionIndicator {{
            background-color: {COLORS['accent']} !important;
        }}
        div[data-testid="stTabs"] [role="tab"]:nth-child(2)[aria-selected="true"] {{
            background: {COLORS['accent_soft']};
            color: {COLORS['accent']} !important;
            border-bottom-color: {COLORS['accent']} !important;
        }}
        div[data-testid="stTabs"] [role="tab"]:nth-child(2)[aria-selected="true"] > div:first-child {{
            color: {COLORS['accent']} !important;
        }}
        div[data-baseweb="tab-list"] {{
            gap: 0.35rem;
            background: {COLORS['surface']};
            border: 1px solid {COLORS['gridline']};
            border-radius: 16px;
            padding: 0.38rem;
            box-shadow: 0 6px 20px {COLORS['shadow_soft']};
        }}
        button[data-baseweb="tab"] {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            border-radius: 11px;
            min-height: 2.8rem;
            padding: 0 1.05rem;
            font-weight: 750;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            background: {COLORS['accent_soft']};
            color: {COLORS['accent']};
        }}
        button[data-baseweb="tab"]:nth-child(2)[aria-selected="true"] {{
            background: {COLORS['accent_soft']};
            color: {COLORS['accent']};
        }}
        div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{ display: none; }}

        div[data-testid="stForm"] {{
            border: 1px solid {COLORS['gridline']};
            background: {COLORS['surface']};
            border-radius: 20px;
            box-shadow: 0 8px 24px {COLORS['shadow_soft']};
        }}

        div[data-testid="stLayoutWrapper"]:has(.dashboard-context-strip) {{
            position: sticky;
            top: 3.1rem;
            z-index: 100;
            margin: 0.7rem 0 0.3rem;
        }}
        div[data-testid="stLayoutWrapper"]:has(.dashboard-context-strip) > div[data-testid="stHorizontalBlock"] {{
            align-items: stretch;
            padding: 0.42rem;
            border: 1px solid {COLORS['gridline']};
            border-radius: 14px;
            background: rgba(255,255,255,0.96);
            box-shadow: 0 8px 22px {COLORS['shadow_soft']};
            backdrop-filter: blur(8px);
        }}
        div[data-testid="stTabs"] [role="tabpanel"],
        div[data-testid="stTabs"] [data-testid="stTabContent"] {{
            overflow: visible !important;
        }}
        .dashboard-context-strip {{
            display: flex;
            align-items: center;
            gap: 0.7rem;
            min-height: 2.55rem;
            padding: 0 0.72rem;
        }}
        .dashboard-context-strip span {{
            flex: 0 0 auto;
            color: {COLORS['accent']};
            font-size: 0.58rem;
            font-weight: 850;
            letter-spacing: 0.11em;
        }}
        .dashboard-context-strip strong {{
            min-width: 0;
            overflow: hidden;
            color: {COLORS['text_primary']};
            font-size: 0.78rem;
            white-space: nowrap;
            text-overflow: ellipsis;
        }}
        div[data-testid="stLayoutWrapper"]:has(.dashboard-context-strip) .stButton button {{
            min-height: 2.55rem;
            border-color: {COLORS['gridline']};
            background: {COLORS['surface']};
            color: {COLORS['text_secondary']};
            font-size: 0.7rem;
        }}

        .dashboard-section-title {{
            display: flex;
            align-items: center;
            gap: 0.58rem;
            margin: 1.4rem 0 0.78rem;
            color: {COLORS['text_primary']};
            font-size: 1.55rem;
            font-weight: 820;
            letter-spacing: -0.025em;
        }}
        .dashboard-section-icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2rem;
            height: 2rem;
            flex: 0 0 2rem;
            color: {COLORS['accent']};
        }}
        .dashboard-section-title.tone-blue .dashboard-section-icon {{
            color: {COLORS['accent']};
        }}
        .dashboard-section-icon svg {{
            width: 100%;
            height: 100%;
            overflow: visible;
            stroke: currentColor;
            stroke-width: 1.55;
            stroke-linecap: round;
            stroke-linejoin: round;
        }}
        .dashboard-section-icon svg .wafer-detail {{
            stroke-width: 1.05;
        }}

        /* Process Dashboard — Best Case 카드 */
        .dashboard-best-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            background: {COLORS['good']};
            color: #ffffff;
            border-radius: 999px;
            padding: 0.22rem 0.75rem;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.06em;
        }}
        .dashboard-best-recipe {{
            margin-left: 0.6rem;
            color: {COLORS['text_primary']};
            font-size: 1.2rem;
            font-weight: 850;
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
            box-shadow: 0 2px 10px {COLORS['shadow_card']};
        }}
        .metric-card .label {{ color: {COLORS['text_secondary']}; font-size: 0.85rem; font-weight: 700; }}
        .metric-card .value {{ color: {COLORS['text_primary']}; font-size: 2.2rem; font-weight: 800; margin: 0.2rem 0; letter-spacing: -0.02em; }}
        .metric-card .unit {{ color: {COLORS['muted']}; font-size: 0.78rem; }}

        /* 품질 지표 상태 카드 */
        .status-card {{
            border-radius: 20px;
            padding: 1.1rem;
            box-shadow: 0 2px 10px {COLORS['shadow_card']};
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
            box-shadow: 0 2px 10px {COLORS['shadow_card']};
        }}

        /* Process Summary / 목표 대비 오차 카드 */
        .summary-card {{
            background-color: {COLORS['surface']};
            border-radius: 18px;
            padding: 0.95rem;
            text-align: center;
            box-shadow: 0 2px 10px {COLORS['shadow_card']};
        }}
        .summary-card .label {{ color: {COLORS['text_secondary']}; font-size: 0.78rem; font-weight: 700; }}
        .summary-card .value {{ color: {COLORS['text_primary']}; font-size: 1.35rem; font-weight: 800; margin-top: 0.25rem; }}

        /* 종합 품질 점수 히어로 */
        .score-hero {{
            background-color: {COLORS['surface']};
            border-radius: 24px;
            padding: 1.8rem 1.5rem;
            text-align: center;
            box-shadow: 0 4px 16px {COLORS['shadow_hero']};
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
            box-shadow: 0 2px 10px {COLORS['shadow_card']};
        }}
        .subscore-card .label {{ color: {COLORS['muted']}; font-size: 0.75rem; font-weight: 700; }}
        .subscore-card .value {{ font-size: 1.3rem; font-weight: 800; margin-top: 0.15rem; }}
        .subscore-card .weight {{ color: {COLORS['muted']}; font-size: 0.7rem; }}

        /* Parameter History — Revision 카드 */
        .history-card {{
            background-color: {COLORS['surface']};
            border-left: 4px solid {COLORS['accent']};
            border-radius: 18px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 2px 10px {COLORS['shadow_card']};
        }}
        .history-card-head {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            flex-wrap: wrap;
        }}
        .history-card-head .revision {{
            color: {COLORS['text_primary']};
            font-size: 1.15rem;
            font-weight: 800;
        }}
        .history-tag {{
            display: inline-block;
            color: {COLORS['accent']};
            background: {COLORS['accent_soft']};
            border-radius: 999px;
            padding: 0.18rem 0.62rem;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.01em;
        }}
        .history-card-notes {{
            color: {COLORS['text_primary']};
            font-size: 0.92rem;
            line-height: 1.55;
            margin-top: 0.5rem;
        }}
        .history-param-list {{
            color: {COLORS['text_secondary']};
            font-size: 0.82rem;
            line-height: 1.75;
            margin-top: 0.45rem;
        }}
        .history-param-list .param-heading {{
            color: {COLORS['muted']};
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        .history-card-footer {{
            display: flex;
            align-items: center;
            gap: 0.9rem;
            flex-wrap: wrap;
            margin-top: 0.7rem;
            padding-top: 0.65rem;
            border-top: 1px solid {COLORS['gridline']};
            color: {COLORS['text_secondary']};
            font-size: 0.82rem;
            font-weight: 700;
        }}
        .history-card-footer .highlight {{ color: {COLORS['text_primary']}; font-weight: 800; }}
        .history-empty {{
            background-color: {COLORS['surface_soft']};
            border: 1px dashed {COLORS['gridline']};
            border-radius: 18px;
            padding: 1.6rem;
            text-align: center;
            color: {COLORS['text_secondary']};
            font-size: 0.95rem;
            font-weight: 600;
        }}
        .history-detail-row {{
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            padding: 0.4rem 0;
            border-bottom: 1px solid {COLORS['gridline']};
            flex-wrap: wrap;
        }}
        .history-detail-row:last-child {{ border-bottom: none; }}
        .history-detail-row .param-label {{ color: {COLORS['text_primary']}; font-weight: 800; font-size: 0.85rem; min-width: 11rem; }}
        .history-detail-row .value-change {{ color: {COLORS['text_secondary']}; font-size: 0.85rem; }}
        .history-detail-row .delta {{ font-weight: 800; font-size: 0.85rem; }}
        .history-detail-row .delta.is-up {{ color: {COLORS['chart_critical']}; }}
        .history-detail-row .delta.is-down {{ color: {COLORS['accent']}; }}

        div[data-testid="stMetricValue"] {{ color: {COLORS['text_primary']} !important; }}

        h3 {{ color: {COLORS['text_primary']}; font-weight: 800; }}

        @media (max-width: 900px) {{
            .block-container {{ padding-top: 3.1rem; }}
            div[data-testid="stColumn"]:has(.process-selector-anchor) {{ width: 100% !important; }}
        }}
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
            <div class="value" style="color:{COLORS['text_primary']};">{value:.1f}</div>
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


def history_tag_badge_html(tag: str | None) -> str:
    """Change_Notes에서 추출한 변경 단계 태그(1차/최종/확정/완료/양산 후보)를 pill 뱃지 HTML로.
    tag가 없으면 빈 문자열(호출부에서 그냥 이어붙이면 됨)."""
    if not tag:
        return ""
    return f'<span class="history-tag">{tag}</span>'


def dashboard_best_badge_html(recipe_label: str) -> str:
    """Process Dashboard의 Best Case 카드용 뱃지. 색상뿐 아니라 "BEST" 문구로도 의미를 전달한다."""
    return f'<span class="dashboard-best-badge">🏆 BEST</span><span class="dashboard-best-recipe">{recipe_label}</span>'


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
