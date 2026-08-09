"""배포 환경과 인증 모드를 관리하는 UI 설정 모듈.

기본값은 공모전 시연용 DEMO 모드다. Streamlit Community Cloud의
Secrets에 값을 등록하면 코드 변경 없이 PRODUCTION 인증으로 전환할 수 있다.
"""

from __future__ import annotations

import hmac

import streamlit as st


APP_VERSION = "v1.0.0"
DEPLOYMENT_DATE = "미정"
DEFAULT_AUTH_MODE = "demo"


def _secret_section(name: str) -> dict:
    """Secrets 파일이 없는 로컬 환경에서도 안전하게 빈 설정을 반환한다."""
    try:
        return dict(st.secrets.get(name, {}))
    except Exception:
        return {}


def get_auth_mode() -> str:
    """`demo` 또는 `production` 인증 모드를 반환한다."""
    configured = str(_secret_section("app").get("auth_mode", DEFAULT_AUTH_MODE)).strip().lower()
    return configured if configured in {"demo", "production"} else DEFAULT_AUTH_MODE


def get_auth_mode_label() -> str:
    return "DEMO" if get_auth_mode() == "demo" else "PRODUCTION"


def production_credentials_configured() -> bool:
    auth = _secret_section("auth")
    return bool(str(auth.get("admin_user", "")).strip() and str(auth.get("admin_password", "")))


def verify_login(user_id: str, password: str) -> tuple[bool, str]:
    """현재 인증 모드에 따라 로그인 입력값을 확인한다.

    DEMO는 공모전 시연을 위해 비어 있지 않은 입력을 허용한다. PRODUCTION은
    Streamlit Secrets의 `[auth]` 값과 상수 시간 비교를 수행한다.
    """
    normalized_user = user_id.strip()
    if not normalized_user or not password:
        return False, "아이디와 비밀번호를 모두 입력해 주세요."

    if get_auth_mode() == "demo":
        return True, ""

    auth = _secret_section("auth")
    expected_user = str(auth.get("admin_user", "")).strip()
    expected_password = str(auth.get("admin_password", ""))
    if not expected_user or not expected_password:
        return False, "운영 로그인 정보가 설정되지 않았습니다. Streamlit Secrets의 [auth] 항목을 확인해 주세요."

    valid_user = hmac.compare_digest(normalized_user, expected_user)
    valid_password = hmac.compare_digest(password, expected_password)
    if valid_user and valid_password:
        return True, ""
    return False, "아이디 또는 비밀번호가 올바르지 않습니다."
