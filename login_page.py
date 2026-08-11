"""테스트용 로그인 랜딩 페이지 UI.

실제 인증을 수행하지 않는다. 부서/아이디/비밀번호 입력 여부만 확인한 뒤
현재 Streamlit 세션에서 대시보드 화면으로 전환한다.
"""

from __future__ import annotations

import base64
import mimetypes
from html import escape
from pathlib import Path

import streamlit as st

from app_config import get_auth_mode, get_auth_mode_label, verify_login
from style import ACTIVE_THEME


# 이 파일만 같은 이름으로 교체하면 HERA 마스코트가 사용되는 모든 위치가 변경된다.
HERA_MASCOT_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "brand"
    / "hera-mascot.png"
)
DEPARTMENT_OPTIONS = ("관리자",)
LOGIN_THEMES = {
    "classic_blue": {
        "page": "#ffffff",
        "hero_text": "#ffffff",
        "hero_overlay_top": "rgba(4,15,35,0.08)",
        "hero_overlay_bottom": "rgba(3,13,31,0.78)",
        "hero_fallback_start": "#051326",
        "hero_fallback_mid": "#07356a",
        "hero_fallback_end": "#0b75cf",
        "hero_accent": "#51b9ff",
        "hero_mark_bg": "rgba(8,37,73,0.32)",
        "hero_status": "#55d9a8",
        "hero_status_soft": "rgba(85,217,168,0.15)",
        "accent": "#3182f6",
        "accent_button": "#1f6ff0",
        "accent_shadow": "rgba(31,111,240,0.22)",
        "focus_shadow": "rgba(49,130,246,0.1)",
        "text_primary": "#191f28",
        "text_secondary": "#6b7684",
        "label": "#333d4b",
        "input_bg": "#f9fafb",
        "border": "#e5e8eb",
        "divider": "#edf0f2",
        "muted": "#8b95a1",
    },
    "racing_coral": {
        "page": "#ffffff",
        "hero_text": "#fffaf6",
        "hero_overlay_top": "rgba(116,35,27,0.14)",
        "hero_overlay_bottom": "rgba(64,20,17,0.84)",
        "hero_fallback_start": "#431714",
        "hero_fallback_mid": "#a9362f",
        "hero_fallback_end": "#f47a3d",
        "hero_accent": "#ff9a5b",
        "hero_mark_bg": "rgba(91,30,25,0.38)",
        "hero_status": "#55d9a8",
        "hero_status_soft": "rgba(85,217,168,0.15)",
        "accent": "#e5483b",
        "accent_button": "#e5483b",
        "accent_shadow": "rgba(229,72,59,0.24)",
        "focus_shadow": "rgba(229,72,59,0.13)",
        "text_primary": "#31201d",
        "text_secondary": "#765b55",
        "label": "#4b332f",
        "input_bg": "#fff8f2",
        "border": "#efddd2",
        "divider": "#f2e2d9",
        "muted": "#9b7d76",
    },
}
LOGIN_COLORS = LOGIN_THEMES[ACTIVE_THEME]
PRIVATE_SESSION_KEYS = (
    "prediction_result",
    "prediction_inputs",
    "prediction_targets",
    "prediction_evaluation",
    "prediction_recommendation",
    "target_mode_baseline",
    "target_mode_suggestion",
)


def _clear_private_session_data() -> None:
    """로그인 사용자가 바뀔 때 이전 평가 결과와 최근 기록을 제거한다."""
    for key in PRIVATE_SESSION_KEYS:
        st.session_state[key] = None
    st.session_state.quality_history = []


def _image_data_uri(image_path: Path) -> str:
    """로컬 이미지를 CSS background-image에서 사용할 data URI로 변환한다."""
    if not image_path.exists():
        return ""

    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@st.cache_data(show_spinner=False)
def get_hera_mascot_uri() -> str:
    """로그인과 앱 헤더가 공유하는 HERA 마스코트 data URI를 반환한다."""
    return _image_data_uri(HERA_MASCOT_PATH)


def _inject_login_css() -> None:
    st.markdown(
        f"""
        <style>
        .login-page-marker {{ display: none; }}

        .stApp:has(.login-page-marker) {{
            background: {LOGIN_COLORS['page']};
        }}
        .stApp:has(.login-page-marker) .block-container {{
            max-width: none;
            padding: 0;
        }}
        .stApp:has(.login-page-marker) header[data-testid="stHeader"],
        .stApp:has(.login-page-marker) section[data-testid="stSidebar"],
        .stApp:has(.login-page-marker) [data-testid="collapsedControl"],
        .stApp:has(.login-page-marker) footer {{
            display: none;
        }}

        div[data-testid="stHorizontalBlock"]:has(.login-hero) {{
            min-height: 100vh;
            gap: 0;
            align-items: stretch;
        }}
        div[data-testid="stColumn"]:has(.login-hero),
        div[data-testid="stColumn"]:has(.login-form-anchor) {{
            min-width: 0;
        }}
        div[data-testid="stColumn"]:has(.login-hero) > div {{
            height: 100%;
        }}
        div[data-testid="stColumn"]:has(.login-form-anchor) {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            background: {LOGIN_COLORS['page']};
            padding: clamp(3rem, 7vw, 7.5rem);
        }}

        .login-hero {{
            position: relative;
            isolation: isolate;
            overflow: hidden;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: clamp(2rem, 4vw, 4.2rem);
            color: {LOGIN_COLORS['hero_text']};
            background:
                radial-gradient(circle at 79% 25%, rgba(255, 246, 232, 0.25) 0 11%, transparent 11.5%),
                radial-gradient(circle at 79% 25%, transparent 0 19%, rgba(255, 219, 192, 0.18) 19.3% 19.8%, transparent 20.1%),
                linear-gradient(145deg, #4b1715 0%, #982d28 48%, #ef7143 100%);
            box-sizing: border-box;
        }}
        .login-hero::before {{
            content: "";
            position: absolute;
            z-index: -2;
            inset: 0;
            opacity: 0.24;
            background-image:
                linear-gradient(rgba(255,255,255,0.18) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.18) 1px, transparent 1px);
            background-size: 52px 52px;
            mask-image: linear-gradient(125deg, transparent 8%, black 54%, transparent 96%);
        }}
        .login-hero::after {{
            content: "";
            position: absolute;
            z-index: -1;
            right: -12%;
            bottom: -25%;
            width: 82%;
            aspect-ratio: 1;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,248,235,0.32) 0%, rgba(255,218,190,0.12) 45%, transparent 70%);
        }}
        .login-hero__eyebrow {{
            display: flex;
            align-items: center;
            gap: 0.7rem;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.17em;
            opacity: 0.94;
        }}
        .login-hero__eyebrow span {{
            display: inline-block;
            width: 2rem;
            height: 2px;
            background: {LOGIN_COLORS['hero_accent']};
        }}
        .login-hero__stage {{
            position: relative;
            flex: 1 1 auto;
            min-height: 0;
        }}
        .login-hero__copy {{
            position: relative;
            z-index: 3;
            max-width: 29rem;
            margin-top: clamp(2.2rem, 9vh, 7rem);
        }}
        .login-hero__mark {{
            width: auto;
            min-width: 4.4rem;
            height: 3.25rem;
            display: inline-grid;
            place-items: center;
            padding: 0 0.9rem;
            margin-bottom: 1.1rem;
            border: 1px solid rgba(255, 255, 255, 0.55);
            border-radius: 14px;
            background: {LOGIN_COLORS['hero_mark_bg']};
            backdrop-filter: blur(12px);
            font-weight: 800;
            letter-spacing: -0.04em;
        }}
        .login-hero h2 {{
            margin: 0;
            color: {LOGIN_COLORS['hero_text']};
            font-size: clamp(3.25rem, 5.2vw, 5.1rem);
            font-weight: 900;
            line-height: 0.95;
            letter-spacing: -0.045em;
            text-wrap: balance;
            text-shadow: 0 5px 24px rgba(53, 12, 9, 0.22);
        }}
        .login-hero__subtitle {{
            max-width: 24rem;
            margin-top: 1.05rem;
            color: rgba(255, 255, 255, 0.86);
            font-size: clamp(1.05rem, 1.55vw, 1.4rem);
            font-weight: 720;
            line-height: 1.35;
            letter-spacing: 0.005em;
        }}
        .login-hero p {{
            max-width: 27rem;
            margin: 1.1rem 0 0;
            color: rgba(255, 255, 255, 0.78);
            font-size: 0.95rem;
            line-height: 1.7;
        }}
        .login-hero__mascot {{
            position: absolute;
            z-index: 2;
            right: clamp(-6.2rem, -7vw, -3rem);
            bottom: clamp(-3.4rem, -4vh, -1.8rem);
            width: min(96%, 40rem);
            max-height: 76vh;
            object-fit: contain;
            object-position: right bottom;
            filter: drop-shadow(0 24px 30px rgba(55, 9, 7, 0.32));
        }}
        .login-hero__footer {{
            position: relative;
            z-index: 4;
            display: flex;
            align-items: center;
            gap: 0.55rem;
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.72rem;
            letter-spacing: 0.08em;
        }}
        .login-hero__status {{
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 50%;
            background: {LOGIN_COLORS['hero_status']};
            box-shadow: 0 0 0 5px {LOGIN_COLORS['hero_status_soft']};
        }}

        .login-form-anchor {{ display: none; }}
        .login-heading {{ max-width: 32rem; margin-bottom: 2rem; }}
        .login-heading__eyebrow {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 0.8rem;
            color: {LOGIN_COLORS['accent']};
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.15em;
        }}
        .login-mode-badge {{
            display: inline-flex;
            align-items: center;
            min-height: 1.45rem;
            padding: 0.15rem 0.48rem;
            border-radius: 999px;
            background: {LOGIN_COLORS['focus_shadow']};
            color: {LOGIN_COLORS['accent']};
            font-size: 0.58rem;
            font-weight: 850;
            letter-spacing: 0.08em;
        }}
        .login-heading h1 {{
            margin: 0;
            color: {LOGIN_COLORS['text_primary']};
            font-size: clamp(2rem, 3vw, 3rem);
            line-height: 1.1;
            letter-spacing: -0.05em;
        }}
        .login-heading p {{
            margin: 0.85rem 0 0;
            color: {LOGIN_COLORS['text_secondary']};
            font-size: 0.93rem;
        }}

        div[data-testid="stColumn"]:has(.login-form-anchor) div[data-testid="stForm"] {{
            max-width: 32rem;
            padding: 0;
            border: 0;
            background: transparent;
        }}
        div[data-testid="stColumn"]:has(.login-form-anchor) label {{
            color: {LOGIN_COLORS['label']};
            font-size: 0.82rem;
            font-weight: 750;
        }}
        div[data-testid="stColumn"]:has(.login-form-anchor) [data-baseweb="select"] > div,
        div[data-testid="stColumn"]:has(.login-form-anchor) [data-baseweb="input"] > div {{
            min-height: 3.25rem;
            border-color: {LOGIN_COLORS['border']};
            border-radius: 12px;
            background: {LOGIN_COLORS['input_bg']};
        }}
        div[data-testid="stColumn"]:has(.login-form-anchor) [data-baseweb="select"] > div:focus-within,
        div[data-testid="stColumn"]:has(.login-form-anchor) [data-baseweb="input"] > div:focus-within {{
            border-color: {LOGIN_COLORS['accent']};
            background: {LOGIN_COLORS['page']};
            box-shadow: 0 0 0 3px {LOGIN_COLORS['focus_shadow']};
        }}
        div[data-testid="stColumn"]:has(.login-form-anchor) div[data-testid="stFormSubmitButton"] button {{
            min-height: 3.35rem;
            margin-top: 0.8rem;
            border: 0;
            border-radius: 12px;
            background: {LOGIN_COLORS['accent_button']};
            box-shadow: 0 9px 20px {LOGIN_COLORS['accent_shadow']};
            font-size: 0.94rem;
        }}
        .login-assurance {{
            max-width: 32rem;
            margin-top: 1.5rem;
            padding-top: 1.25rem;
            border-top: 1px solid {LOGIN_COLORS['divider']};
            color: {LOGIN_COLORS['muted']};
            font-size: 0.75rem;
            line-height: 1.6;
        }}

        @media (max-width: 760px) {{
            div[data-testid="stHorizontalBlock"]:has(.login-hero) {{
                min-height: auto;
                flex-direction: column;
            }}
            div[data-testid="stColumn"]:has(.login-hero),
            div[data-testid="stColumn"]:has(.login-form-anchor) {{
                width: 100% !important;
                flex: 1 1 auto !important;
            }}
            .login-hero {{
                min-height: 20rem;
                padding: 2rem;
            }}
            .login-hero__copy {{
                max-width: 72%;
                margin-top: 2.1rem;
            }}
            .login-hero h2 {{ font-size: clamp(2.8rem, 12vw, 3.6rem); }}
            .login-hero__subtitle {{
                max-width: 72%;
                margin-top: 0.65rem;
                font-size: 0.98rem;
            }}
            .login-hero__mark {{ height: 2.65rem; min-width: 3.9rem; margin-bottom: 0.75rem; }}
            .login-hero__mascot {{
                right: -2.8rem;
                bottom: -2.2rem;
                width: min(57%, 18rem);
                max-height: 16rem;
            }}
            .login-hero__copy p {{ display: none; }}
            div[data-testid="stColumn"]:has(.login-form-anchor) {{
                min-height: calc(100vh - 20rem);
                padding: 2.25rem 1.5rem 3rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_login_page() -> None:
    """템플릿형 로그인 화면을 그리고 성공 시 현재 대시보드로 전환한다."""
    _inject_login_css()
    auth_mode = get_auth_mode()
    auth_mode_label = get_auth_mode_label()
    mascot_uri = get_hera_mascot_uri()
    mascot_html = (
        f'<img class="login-hero__mascot" src="{mascot_uri}" alt="HERA mascot">'
        if mascot_uri
        else ""
    )
    st.markdown('<span class="login-page-marker"></span>', unsafe_allow_html=True)

    hero_column, form_column = st.columns([0.92, 1.28], gap=None)

    with hero_column:
        st.markdown(
            f"""
            <section class="login-hero">
                <div class="login-hero__eyebrow"><span></span> PROCESS INTELLIGENCE</div>
                <div class="login-hero__stage">
                    <div class="login-hero__copy">
                        <h2>HERA</h2>
                        <div class="login-hero__subtitle">Hynix Etch Recipe Advisor</div>
                    </div>
                    {mascot_html}
                </div>
                <div class="login-hero__footer">
                    <span class="login-hero__status"></span>
                    ETCH PROCESS SIMULATION SYSTEM
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    with form_column:
        st.markdown(
            f"""
            <div class="login-form-anchor"></div>
            <div class="login-heading">
                <div class="login-heading__eyebrow">
                    SECURE WORKSPACE
                    <span class="login-mode-badge">{auth_mode_label} MODE</span>
                </div>
                <h1>Login</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("test_login_form", clear_on_submit=False):
            department = st.selectbox("부서 선택", DEPARTMENT_OPTIONS)
            user_id = st.text_input("아이디", placeholder="아이디를 입력하세요")
            password = st.text_input("비밀번호", type="password", placeholder="비밀번호를 입력하세요")
            submitted = st.form_submit_button("로그인  →", type="primary", use_container_width=True)

        if submitted:
            is_valid, error_message = verify_login(user_id, password)
            if is_valid:
                _clear_private_session_data()
                st.session_state.authenticated = True
                st.session_state.authenticated_user = user_id.strip()
                st.session_state.authenticated_department = department
                st.query_params["auth"] = "1"
                st.rerun()
            else:
                st.error(error_message)

        st.markdown(
            f"""
            <div class="login-assurance">
                {'DEMO MODE · 아이디와 비밀번호 입력 여부만 확인합니다.' if auth_mode == 'demo' else 'PRODUCTION MODE · Streamlit Secrets에 등록된 계정만 허용합니다.'}<br>
                회사 공정 시뮬레이션 전용 화면이며 계정 생성 기능은 제공하지 않습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar_logout() -> None:
    """로그인된 사용자 정보와 로그아웃 버튼을 사이드바 최상단에 표시한다."""
    user_id = st.session_state.get("authenticated_user") or "관리자"
    department = st.session_state.get("authenticated_department") or "관리자"
    initial = user_id[:1].upper()
    auth_mode_label = get_auth_mode_label()

    st.markdown(
        f"""
        <div class="sidebar-user-card">
            <div class="sidebar-user-avatar">{escape(initial)}</div>
            <div>
                <div class="sidebar-user-name">{escape(user_id)}</div>
                <div class="sidebar-user-role">{escape(department)} · {auth_mode_label} session</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("← 로그아웃", key="sidebar_logout", use_container_width=True):
        _clear_private_session_data()
        st.session_state.authenticated = False
        st.session_state.authenticated_user = None
        st.session_state.authenticated_department = None
        st.query_params.pop("auth", None)
        st.rerun()
