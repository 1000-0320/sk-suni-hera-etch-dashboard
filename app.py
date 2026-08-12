"""
Etch AI Decision Support System
반도체 Etch 공정 품질 예측 및 의사결정 지원 시스템

isolation / trench / gate / metal 4개 공정을 선택할 수 있고, 학습된 RandomForest/XGBoost 모델로
실제 예측을 수행한다 (model.predict() 내부가 ml_engine의 학습된 모델을 호출).

업로드 워크북은 3개 시트 구조를 전제로 한다: Recipe_Master / Wafer_Summary / Site_Level_Raw

실행: streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import datetime
from html import escape
from time import perf_counter
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from app_config import APP_VERSION, DEPLOYMENT_DATE, get_auth_mode_label
from model import (
    predict, compute_composite_score, evaluate_against_target, score_recipe_versions,
    recommend_best_recipe, recommend_parameter_adjustments, build_stage_diff_table, generate_dashboard_analysis,
    format_parameter_label, generate_recommendation_reason,
    apply_recommended_changes, build_combined_recipe_table,
    find_layer_references,
)
from data_utils import (
    generate_dummy_workbook, load_bundled_workbook, is_valid_workbook, load_required_sheets,
    get_zone_summary, get_recipe_stage_table, stage_inputs_from_recipe,
    ordered_recipe_versions, get_default_targets, get_dashboard_spec_targets, get_representative_chamber,
    PROCESS_STAGE_DEFS, PROCESS_LABELS,
    REQUIRED_SHEETS, PARTICLE_DEFECT_THRESHOLD,
)
from history_utils import (
    PARAM_TYPE_LABELS, get_equipment_models, get_detail_param_options, format_detail_param_label,
    build_history_results,
)
from style import (
    inject_custom_css, render_metric_card, render_summary_card, render_status_badge,
    render_score_hero, render_subscore_card, render_pill_card, render_history_kpi_card,
    get_pass_rate_status, get_cd_uniformity_status, get_depth_uniformity_status, get_particle_status,
    get_score_status, STATUS_META, history_tag_badge_html, dashboard_best_badge_html, COLORS,
)
from charts import (
    build_cd_bar_chart, build_gauge_chart, build_variation_gauge,
    build_cd_trend_chart, build_depth_trend_chart,
    build_cd_uniformity_trend_chart, build_depth_uniformity_trend_chart,
    build_pass_rate_trend_chart, build_particle_chart,
    build_rev_cd_trend_chart, build_rev_depth_trend_chart,
    build_rev_uniformity_trend_chart, build_rev_pass_rate_chart, build_rev_defect_chart,
    build_recipe_score_chart, build_rev_comparison_chart, build_wafer_profile_chart,
    build_zone_cd_chart, build_zone_depth_chart, build_zone_spread_chart,
    build_zone_pass_rate_chart, build_zone_defect_chart,
    build_cd_comparison_chart, build_score_comparison_chart,
)
from login_page import get_hera_mascot_uri, render_login_page, render_sidebar_logout

st.set_page_config(page_title="Etch AI Decision Support System", page_icon="🧪", layout="wide")
inject_custom_css()

# Wafer 단면 Profile에서 한 번에 보여줄 지표들 (표시용 라벨 -> Site_Level_Raw 실제 컬럼명)
WAFER_MAP_METRICS = {
    "Top CD": "Top_CD_nm", "Mid CD": "Mid_CD_nm", "Bottom CD": "Bottom_CD_nm", "Depth": "Depth_nm",
}
TARGET_MODE_LABEL = "목표품질 달성을 위한 레시피 변경점 추천"
DIRECT_MODE_LABEL = "레시피 조건을 직접 입력하여 품질 평가"

# 공정별 Before/After 단면 스키매틱 이미지 (멘토 피드백: 텍스트 카드 대신 그림으로 직관적 표현)
PROCESS_SCHEMATIC_IMAGES = {
    "isolation": "assets/process/isolation_schematic.png",
    "trench": "assets/process/trench_schematic.png",
    "gate": "assets/process/gate_schematic.png",
    "metal": "assets/process/metal_schematic.png",
}


# ==============================================================================
# 세션 상태 초기화
# ==============================================================================
def init_session_state():
    defaults = {
        "authenticated": False,
        "authenticated_user": None,
        "authenticated_department": None,
        "process": "trench",
        "uploaded_filename": None,
        "sheet_names": [],
        "workbook": None,  # {"Recipe_Master":df, "Wafer_Summary":df, "Site_Level_Raw":df}
        "workbook_process": None,  # 업로드된 workbook이 어느 공정 기준으로 검증됐는지
        "prediction_result": None,
        "prediction_inputs": None,
        "prediction_targets": None,
        "prediction_evaluation": None,
        "prediction_recommendation": None,
        "target_mode_baseline": None,
        "target_mode_suggestion": None,
        "target_mode_combo": None,
        "quality_history": [],
        "prediction_running": False,
        "last_run_duration": None,
        "last_run_error": None,
        "last_run_kind": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def get_active_workbook() -> dict:
    """현재 선택된 공정에 맞는 워크북을 반환.
    업로드된 워크북이 유효(현재 공정 기준 3-시트 구조 일치)하면 그것을,
    아니면 레포에 함께 배포된 실측 데이터를 기본값으로 사용한다."""
    process = st.session_state.process
    if is_valid_workbook(st.session_state.workbook, process) and st.session_state.workbook_process == process:
        return st.session_state.workbook
    return load_bundled_workbook(process)


def using_real_data() -> bool:
    process = st.session_state.process
    return is_valid_workbook(st.session_state.workbook, process) and st.session_state.workbook_process == process


_DASHBOARD_WAFER_ICONS = {
    "summary": """
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="8.5" />
            <path d="M8 4.5v15M12 3.5v17M16 4.5v15M4.5 8h15M3.5 12h17M4.5 16h15" class="wafer-detail" />
            <path d="M10 20.2h4" />
        </svg>
    """,
    "score": """
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="8.5" />
            <path d="M7 15v2M11 11v6M15 7v10" />
            <path d="M10 20.2h4" />
        </svg>
    """,
    "quality": """
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="8.5" />
            <path d="M6.5 14.5l3-3 2.6 2 4.9-5" />
            <path d="M10 20.2h4" />
        </svg>
    """,
    "map": """
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="8.5" />
            <circle cx="12" cy="12" r="4.5" class="wafer-detail" />
            <path d="M12 3.5v17M3.5 12h17" class="wafer-detail" />
            <path d="M10 20.2h4" />
        </svg>
    """,
    "zone": """
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="8.5" />
            <path d="M12 3.5V12l6 6M12 12H3.5M12 12l-6 6" class="wafer-detail" />
            <path d="M10 20.2h4" />
        </svg>
    """,
    "ai": """
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="8.5" />
            <path d="M8 8.5l4 3.5 4-3.5M8 15.5l4-3.5 4 3.5" class="wafer-detail" />
            <circle cx="8" cy="8.5" r="1" /><circle cx="16" cy="8.5" r="1" />
            <circle cx="12" cy="12" r="1" /><circle cx="8" cy="15.5" r="1" /><circle cx="16" cy="15.5" r="1" />
            <path d="M10 20.2h4" />
        </svg>
    """,
    "trophy": '<span style="font-size:1.4rem;line-height:1;">🏆</span>',
}


def render_dashboard_section_title(title: str, icon: str, tone: str = "coral") -> None:
    """Process Dashboard 섹션 제목을 통일된 웨이퍼 라인 아이콘으로 표시한다."""
    st.markdown(
        f"""
        <div class="dashboard-section-title tone-{escape(tone)}">
            <span class="dashboard-section-icon">{_DASHBOARD_WAFER_ICONS[icon]}</span>
            <span>{escape(title)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 0. 공정 선택 (isolation / trench / gate / metal) — 화면 전체에 영향
# ==============================================================================
def create_process_selector():
    options = list(PROCESS_STAGE_DEFS.keys())
    short_labels = {"isolation": "Isolation", "trench": "Trench", "gate": "Gate", "metal": "Metal"}
    mascot_uri = get_hera_mascot_uri()
    mascot_visual = (
        f'<img src="{mascot_uri}" alt="HERA mascot">'
        if mascot_uri
        else '<div class="hera-header-fallback">HERA</div>'
    )

    header_copy_col, header_mascot_col = st.columns([1.5, 0.5], gap="large")
    with header_copy_col:
        st.markdown(
            """
            <div class="app-shell-header">
                <div class="app-shell-eyebrow">ETCH PROCESS INTELLIGENCE</div>
                <h1 class="app-shell-title">Decision Support Center</h1>
                <div class="app-shell-subtitle">반도체 Etch 공정 예측, 품질 평가와 의사결정을 하나의 흐름에서 관리합니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_mascot_col:
        st.markdown(
            f"""
            <aside class="hera-header-card" aria-label="HERA brand mascot">
                <div class="hera-header-copy">
                    <span>ETCH RECIPE ADVISOR</span>
                    <strong>HERA</strong>
                    <small>Process intelligence guide</small>
                </div>
                {mascot_visual}
            </aside>
            """,
            unsafe_allow_html=True,
        )

    selector_col, equipment_col = st.columns([1, 1], gap="large")
    with selector_col:
        st.markdown('<div class="process-selector-anchor">ACTIVE PROCESS</div>', unsafe_allow_html=True)
        selected = st.radio(
            "분석할 공정",
            options,
            index=options.index(st.session_state.process),
            format_func=lambda key: short_labels.get(key, key.title()),
            horizontal=True,
            label_visibility="collapsed",
            key="process_selector",
        )

    if selected != st.session_state.process:
        st.session_state.process = selected
        st.session_state.prediction_result = None
        st.session_state.prediction_recommendation = None
        st.session_state.target_mode_baseline = None
        st.session_state.target_mode_suggestion = None
        st.session_state.target_mode_combo = None

    process = st.session_state.process
    equipment_options = sorted(get_active_workbook()["Wafer_Summary"]["Equipment_Model"].unique())
    equipment_key = f"top_equipment_{process}"
    if st.session_state.get(equipment_key) not in equipment_options and equipment_options:
        st.session_state[equipment_key] = equipment_options[0]

    with equipment_col:
        st.markdown('<div class="process-selector-anchor">EQUIPMENT</div>', unsafe_allow_html=True)
        st.radio(
            "장비 선택",
            equipment_options,
            horizontal=True,
            label_visibility="collapsed",
            key=equipment_key,
        )


# ==============================================================================
# 1. 사이드바 — Excel 업로드 / 시트 목록 / 데이터 불러오기
# ==============================================================================
def create_sidebar():
    with st.sidebar:
        render_sidebar_logout()
        process = st.session_state.process

        st.markdown('<div class="sidebar-section-label">ACTIVE PROCESS</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sidebar-context-card">
                <div class="label">현재 분석 공정</div>
                <div class="value"><span class="dot"></span>{PROCESS_LABELS[process]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-label">DATA WORKSPACE</div>', unsafe_allow_html=True)
        with st.expander("Excel 데이터 연결", expanded=False):
            uploaded_file = st.file_uploader(
                "Excel 파일 (.xlsx)",
                type=["xlsx"],
                key=f"uploader_{process}",
            )

            if uploaded_file is not None:
                st.session_state.uploaded_filename = uploaded_file.name
                st.success(f"선택됨: {uploaded_file.name}")

                try:
                    excel_file = pd.ExcelFile(uploaded_file)
                    st.session_state.sheet_names = excel_file.sheet_names

                    st.markdown("**시트 확인**")
                    for sheet in st.session_state.sheet_names:
                        mark = "✅" if sheet in REQUIRED_SHEETS else "•"
                        st.markdown(f"- {mark} {sheet}")

                    if st.button("데이터 불러오기", type="primary", use_container_width=True):
                        sheets = load_required_sheets(excel_file)
                        if is_valid_workbook(sheets, process):
                            st.session_state.workbook = sheets
                            st.session_state.workbook_process = process
                            n_wafer = len(sheets["Wafer_Summary"])
                            n_site = len(sheets["Site_Level_Raw"])
                            st.success(f"로드 완료 · Wafer {n_wafer}장 / Site {n_site}행")
                        else:
                            st.error(
                                f"현재 선택한 공정({PROCESS_LABELS[process]})의 시트/컬럼 구조와 맞지 않습니다."
                            )
                except Exception as e:
                    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
            else:
                st.caption("파일을 선택하지 않으면 기본 제공 실측 데이터를 사용합니다.")

        if using_real_data():
            data_label = "업로드한 실측 데이터"
            data_detail = st.session_state.uploaded_filename or "사용자 데이터"
        else:
            data_label = "기본 제공 실측 데이터"
            data_detail = "Repository bundled source"

        st.markdown(
            f"""
            <div class="sidebar-data-card">
                <div class="label">현재 데이터 소스</div>
                <div class="value"><span class="dot"></span>{data_label}</div>
                <div class="sidebar-footnote">{data_detail}</div>
            </div>
            <div class="sidebar-footnote">
                필요 시트 · Recipe_Master / Wafer_Summary / Site_Level_Raw<br>
                Etch AI Decision Support System v1.0
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-label">RECENT QUALITY</div>', unsafe_allow_html=True)
        recent_quality_slot = st.empty()

        st.markdown('<div class="sidebar-section-label">SYSTEM STATUS</div>', unsafe_allow_html=True)
        model_status = "READY"
        source_status = "UPLOADED" if using_real_data() else "BUNDLED"
        st.markdown(
            f"""
            <div class="sidebar-system-card">
                <div class="sidebar-system-head">
                    <span>Etch AI</span><strong>{APP_VERSION}</strong>
                </div>
                <div class="sidebar-system-row"><span>Auth mode</span><strong>{get_auth_mode_label()}</strong></div>
                <div class="sidebar-system-row"><span>Model</span><strong class="is-ready">{model_status}</strong></div>
                <div class="sidebar-system-row"><span>Data source</span><strong>{source_status}</strong></div>
                <div class="sidebar-system-row"><span>배포일</span><strong>{DEPLOYMENT_DATE}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return recent_quality_slot


def summarize_recipe_changes(
    inputs: dict,
    recipe_master_df: pd.DataFrame,
    recipe: str,
    stage_defs: list,
) -> list[str]:
    """기준 Recipe와 직접 입력값을 비교해 사람이 읽기 쉬운 변경 목록을 만든다."""
    baseline = stage_inputs_from_recipe(recipe_master_df, recipe, stage_defs)
    changes = []

    for stage in stage_defs:
        stage_key = stage["key"]
        input_prefix = stage_key.lower()
        parameters = [
            (f"{input_prefix}_time", "Etch Time", "s"),
            (f"{input_prefix}_rf_bias", "RF Bias", "W"),
            (f"{input_prefix}_pressure", "Pressure", "mT"),
        ]
        def _gas_short_name(gas_col: str, stage_key: str = stage_key) -> str:
            prefix = f"{stage_key}_"
            name = gas_col[len(prefix):] if gas_col.startswith(prefix) else gas_col
            return name[: -len("_sccm")] if name.endswith("_sccm") else name

        parameters.extend(
            (gas_col.lower(), f"{_gas_short_name(gas_col)} Flow", "sccm")
            for gas_col in stage["gas_cols"]
        )

        for input_key, label, unit in parameters:
            current = inputs.get(input_key)
            reference = baseline.get(input_key)
            if current is None or reference is None:
                continue
            if abs(float(current) - float(reference)) > 1e-9:
                changes.append(f"{stage_key} {label} {float(current):g}{unit}")

    return changes


def append_quality_history(
    result: dict,
    evaluation: dict,
    recipe: str,
    inputs: dict,
    recipe_master_df: pd.DataFrame,
    stage_defs: list,
) -> None:
    """현재 사용자 세션에 최근 직접 품질 평가를 최대 5개까지 보관한다."""
    if not result or result.get("_error") or not evaluation:
        return

    changes = summarize_recipe_changes(inputs, recipe_master_df, recipe, stage_defs)
    entry = {
        "evaluated_at": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%m.%d %H:%M"),
        "process": PROCESS_LABELS[st.session_state.process],
        "recipe": recipe,
        "changes": changes[:2],
        "change_count": len(changes),
        "score": evaluation.get("score", {}).get("total"),
        "pass_rate": result.get("Overall Spec Pass Rate"),
        "cd_uniformity": result.get("CD Uniformity"),
        "depth_uniformity": result.get("Depth Uniformity"),
        "defect_count": result.get("Defect Count"),
    }
    history = list(st.session_state.get("quality_history", []))
    history.insert(0, entry)
    st.session_state.quality_history = history[:5]


def render_recent_quality_sidebar(slot):
    """클릭하면 현재 사용자 세션의 최근 품질 평가 최대 5개를 보여준다."""
    history = list(st.session_state.get("quality_history", []))[:5]

    with slot.container():
        with st.expander(f"최근에 실시한 품질 평가 · {len(history)}/5", expanded=False):
            history_size_class = " has-multiple" if len(history) >= 2 else ""
            st.markdown(
                f'<span class="quality-history-anchor{history_size_class}"></span>',
                unsafe_allow_html=True,
            )
            if not history:
                st.markdown(
                    """
                    <div class="quality-history-empty">
                        아직 평가 기록이 없습니다.<br>
                        공정 조건을 입력하고 품질 평가를 실행해 주세요.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                return

            def metric(value, suffix=""):
                if value is None:
                    return "—"
                return f"{float(value):.1f}{suffix}"

            def delta_item(label, current, previous, suffix="", higher_is_better=True):
                if current is None or previous is None:
                    return f'<span><small>{label}</small><strong class="is-neutral">—</strong></span>'
                delta = float(current) - float(previous)
                improved = delta > 0 if higher_is_better else delta < 0
                worsened = delta < 0 if higher_is_better else delta > 0
                state_class = "is-improved" if improved else "is-worse" if worsened else "is-neutral"
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                return (
                    f'<span><small>{label}</small>'
                    f'<strong class="{state_class}">{arrow} {delta:+.1f}{suffix}</strong></span>'
                )

            for index, entry in enumerate(history):
                latest_badge = '<span class="badge">LATEST</span>' if index == 0 else ""
                process_name = escape(str(entry["process"]))
                evaluated_at = escape(str(entry["evaluated_at"]))
                recipe_name = escape(str(entry["recipe"]))
                changes = [escape(str(change)) for change in entry.get("changes", [])]
                change_count = int(entry.get("change_count", len(changes)))
                if changes:
                    change_text = " · ".join(changes)
                    remaining_count = max(0, change_count - len(changes))
                    if remaining_count:
                        change_text += f" · 외 {remaining_count}개"
                    change_html = f'<div class="quality-history-change"><strong>변경</strong>{change_text}</div>'
                else:
                    change_html = (
                        '<div class="quality-history-change baseline">'
                        '<strong>변경 없음</strong>기준 Recipe 조건'
                        '</div>'
                    )
                score_value = entry.get("score")
                score_color = (
                    STATUS_META[get_score_status(float(score_value))]["color"]
                    if score_value is not None
                    else "#ffffff"
                )
                comparison_html = ""
                if index == 0 and len(history) >= 2:
                    previous = history[1]
                    comparison_html = (
                        '<div class="quality-history-delta-title">LATEST VS PREVIOUS</div>'
                        '<div class="quality-history-delta-grid">'
                        f"{delta_item('종합 점수', entry.get('score'), previous.get('score'), '점')}"
                        f"{delta_item('Pass Rate', entry.get('pass_rate'), previous.get('pass_rate'), '%p')}"
                        f"{delta_item('CD 균일도', entry.get('cd_uniformity'), previous.get('cd_uniformity'), '%p', False)}"
                        f"{delta_item('Defect', entry.get('defect_count'), previous.get('defect_count'), '', False)}"
                        '</div>'
                    )
                card_html = (
                    f'<div class="quality-history-item{" latest" if index == 0 else ""}">'
                    '<div class="quality-history-head">'
                    f'<strong>{process_name}</strong><span>{evaluated_at}</span>'
                    '</div>'
                    f'<div class="quality-history-meta">Recipe · {recipe_name} {latest_badge}</div>'
                    f'{change_html}'
                    '<div class="quality-history-score">'
                    f'<strong style="color:{score_color};">{metric(score_value)}</strong>'
                    '<span>종합 품질 점수</span>'
                    '</div>'
                    f'{comparison_html}'
                    '<div class="quality-history-grid">'
                    f'<span>Pass Rate <strong>{metric(entry["pass_rate"], "%")}</strong></span>'
                    f'<span>CD 균일도 <strong>{metric(entry["cd_uniformity"], "%")}</strong></span>'
                    f'<span>Depth 균일도 <strong>{metric(entry["depth_uniformity"], "%")}</strong></span>'
                    f'<span>Defect <strong>{metric(entry["defect_count"])}</strong></span>'
                    '</div></div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)


# ==============================================================================
# 2. 공정 조건 입력 패널 (A. 기본정보 / B. 장비 / C. 목표 품질 / D. 현재 Recipe)
# ==============================================================================
def create_input_panel():
    process = st.session_state.process
    stage_defs = PROCESS_STAGE_DEFS[process]
    workbook = get_active_workbook()
    wafer_df = workbook["Wafer_Summary"]
    recipe_df = workbook["Recipe_Master"]

    # ---- 공정 기본 정보 (텍스트 카드 대신 Before/After 단면 스키매틱 — 멘토 피드백) ----
    st.markdown("<div class='section-title'>공정 기본 정보</div>", unsafe_allow_html=True)
    schematic_path = PROCESS_SCHEMATIC_IMAGES.get(process)
    if schematic_path and os.path.exists(schematic_path):
        st.image(schematic_path, use_container_width=True)
    layer_names = " → ".join(s["label"] for s in stage_defs)
    st.caption(
        f"{PROCESS_LABELS[process]} · {len(stage_defs)}-Step · Layer 순서: {layer_names}"
    )

    # ---- 시뮬레이터 모드 (먼저 선택 — 멘토 피드백: 무엇을 할지 먼저 고르게) ----
    st.markdown("<div class='section-title'>시뮬레이터 모드</div>", unsafe_allow_html=True)
    st.markdown('<span class="simulator-mode-anchor"></span>', unsafe_allow_html=True)
    mode_label = st.radio(
        "무엇을 하고 싶으신가요?",
        [TARGET_MODE_LABEL, DIRECT_MODE_LABEL],
        horizontal=True, key=f"mode_{process}",
    )

    # ---- Recipe 선택 (Equipment는 상단 ACTIVE PROCESS 옆 선택을 그대로 사용, Chamber는 자동 대표값 — 멘토 피드백: 모든 Chamber 동일 조건으로 가정) ----
    st.markdown("<div class='section-title'>Recipe 선택</div>", unsafe_allow_html=True)
    equipment_options = sorted(wafer_df["Equipment_Model"].unique())
    equipment = st.session_state.get(f"top_equipment_{process}") or (equipment_options[0] if equipment_options else None)
    chamber = get_representative_chamber(wafer_df, equipment)
    recipe_options = ordered_recipe_versions(recipe_df)
    recipe = st.selectbox("Recipe (참고용 베이스라인)", recipe_options, key=f"recipe_{process}")
    st.caption(f"Equipment: **{equipment}** · Chamber는 모든 Chamber가 동일하다는 가정으로 대표값(`{chamber}`)을 자동 사용합니다.")

    stage_table = get_recipe_stage_table(recipe_df, recipe, stage_defs)
    if not stage_table.empty:
        st.markdown(f"**Recipe '{recipe}' 실제 Stage 조건 (참고용)**")
        st.dataframe(stage_table, use_container_width=True, hide_index=True)

    # ---- 목표 품질 설정 (현재 기준값 → 변경 목표 형식) ----
    st.markdown("<div class='section-title'>목표 품질 설정</div>", unsafe_allow_html=True)
    st.caption(
        f"현재 기준값은 '{recipe}' Recipe의 실측 평균입니다. 원하는 변경 목표를 입력하면 "
        "해당 목표에 가까워지기 위한 Recipe 변경점을 추천합니다."
    )

    # current_targets: Equipment/Base Recipe 선택 조합에서 매 rerun마다 새로 계산되는 읽기 전용 기준값.
    current_targets = get_default_targets(wafer_df, recipe_df, recipe)
    defaults = current_targets  # 아래 Uniformity/Pass Rate/Defect Count 입력은 기존 그대로 이 값을 참조한다.

    # edited_targets: 사용자가 입력한 변경 목표(각 number_input의 session_state 값 자체가 저장소 역할).
    # Equipment 또는 Base Recipe가 바뀔 때만 current_targets로 재초기화하고, 그 외 rerun에서는 그대로 유지한다.
    baseline_sig_key = f"target_baseline_sig_{process}"
    baseline_sig = (equipment, recipe)
    if st.session_state.get(baseline_sig_key) != baseline_sig:
        st.session_state[baseline_sig_key] = baseline_sig
        for field, value in current_targets.items():
            st.session_state[f"target_input_{field}_{process}"] = value

    def _target_change_row(label: str, unit: str, field: str, decimals: int = 1):
        """'{label} ({unit})' 제목 아래, 라벨 행과 값 행을 별도 columns로 나눠 '현재 기준값(읽기 전용) →
        변경 목표(입력)'를 배치한다. 두 행이 동일한 [1.1, 0.3, 1.1] 비율과 동일한 높이(2.5rem) wrapper를
        쓰기 때문에, padding을 미세조정하지 않아도 화살표·입력창과 같은 가로선에 맞는다."""
        st.markdown(f"**{label} ({unit})**")

        label_cols = st.columns([1.1, 0.3, 1.1])
        with label_cols[0]:
            st.markdown("<div class='target-label'>현재 기준값</div>", unsafe_allow_html=True)
        with label_cols[2]:
            st.markdown("<div class='target-label'>변경 목표</div>", unsafe_allow_html=True)

        value_cols = st.columns([1.1, 0.3, 1.1], vertical_alignment="top")
        with value_cols[0]:
            st.markdown(
                f"<div class='current-target-value'>{current_targets[field]:.{decimals}f}</div>",
                unsafe_allow_html=True,
            )
        with value_cols[1]:
            st.markdown("<div class='target-arrow'>&rarr;</div>", unsafe_allow_html=True)
        with value_cols[2], st.container(key=f"target_value_input_{field}_{process}"):
            return st.number_input(
                "변경 목표", key=f"target_input_{field}_{process}", format=f"%.{decimals}f",
                label_visibility="collapsed",
            )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_top_cd = _target_change_row("Top CD", "nm", "target_top_cd")
    with c2:
        target_mid_cd = _target_change_row("Mid CD", "nm", "target_mid_cd")
    with c3:
        target_bottom_cd = _target_change_row("Bottom CD", "nm", "target_bottom_cd")
    with c4:
        target_depth = _target_change_row("Depth", "nm", "target_depth")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        max_cd_uniformity = st.number_input(
            "Maximum CD Uniformity (%)", value=defaults["max_cd_uniformity"], step=0.5, format="%.1f", key=f"mcdu_{process}",
        )
    with c6:
        max_depth_uniformity = st.number_input(
            "Maximum Depth Uniformity (%)", value=defaults["max_depth_uniformity"], step=0.5, format="%.1f", key=f"mdu_{process}",
        )
    with c7:
        min_pass_rate = st.number_input(
            "Minimum Pass Rate (%)", value=defaults["min_pass_rate"], step=1.0, format="%.1f", key=f"mpr_{process}",
        )
    with c8:
        max_defect_count = st.number_input(
            "Maximum Defect Count (건)", value=int(defaults["max_defect_count"]), step=1, format="%d", key=f"mdc_{process}",
        )

    targets = {
        "target_top_cd": target_top_cd, "target_mid_cd": target_mid_cd,
        "target_bottom_cd": target_bottom_cd, "target_depth": target_depth,
        "max_cd_uniformity": max_cd_uniformity, "max_depth_uniformity": max_depth_uniformity,
        "min_pass_rate": min_pass_rate, "max_defect_count": max_defect_count,
    }

    stage_defaults = stage_inputs_from_recipe(recipe_df, recipe, stage_defs)
    baseline_inputs = {"equipment": equipment, "chamber": chamber, "recipe": recipe, **stage_defaults}

    # ---- 레시피 변경 (모드별 입력/추천 — 멘토 피드백: 기능 두 가지를 명확히 분리) ----
    if mode_label == TARGET_MODE_LABEL:
        st.caption("위 Recipe를 출발점으로, 목표 품질에 가까워지려면 파라미터를 어떻게 바꾸면 좋을지 AI가 추천합니다.")
        run = st.button(
            "변경점 추천 실행 →",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("prediction_running", False),
        )
        return "target", baseline_inputs, targets, run, stage_defs, recipe, workbook

    # ---- 현재 공정 조건 입력 (모든 Stage를 한 번에 펼쳐서 표시 — 멘토 피드백 반영) ----
    st.markdown("<div class='section-title'>현재 공정 조건 입력</div>", unsafe_allow_html=True)
    st.caption(f"{PROCESS_LABELS[process]} 전체 {len(stage_defs)}개 Stage를 한 화면에서 바로 확인·수정할 수 있습니다.")

    inputs = {"equipment": equipment, "chamber": chamber, "recipe": recipe}
    match = recipe_df[recipe_df["Recipe_Version"] == recipe]
    recipe_row = match.iloc[0] if not match.empty else None

    for stage in stage_defs:
        key = stage["key"].lower()
        st.markdown(f"**{stage['label']}**")
        d1, d2, d3 = st.columns(3)
        with d1:
            inputs[f"{key}_time"] = st.number_input(
                "Etch Time (s)", value=stage_defaults.get(f"{key}_time", 0.0),
                key=f"{key}_time_{process}_{recipe}",
            )
        with d2:
            inputs[f"{key}_rf_bias"] = st.number_input(
                "RF Bias (W)", value=stage_defaults.get(f"{key}_rf_bias", 0.0),
                key=f"{key}_rf_bias_{process}_{recipe}",
            )
        with d3:
            inputs[f"{key}_pressure"] = st.number_input(
                "Pressure (mT)", value=stage_defaults.get(f"{key}_pressure", 0.0),
                key=f"{key}_pressure_{process}_{recipe}",
            )

        gas_cols_ui = st.columns(len(stage["gas_cols"]))
        for gas_col_widget, gas_col_name in zip(gas_cols_ui, stage["gas_cols"]):
            gas_prefix = f"{stage['key']}_"
            gas_label = gas_col_name[len(gas_prefix):] if gas_col_name.startswith(gas_prefix) else gas_col_name
            gas_label = gas_label[: -len("_sccm")] if gas_label.endswith("_sccm") else gas_label
            default_val = float(recipe_row[gas_col_name]) if recipe_row is not None and gas_col_name in recipe_row.index else 0.0
            input_key = gas_col_name.lower()
            with gas_col_widget:
                inputs[input_key] = st.number_input(
                    f"{gas_label} Flow (sccm)", value=default_val,
                    key=f"{input_key}_{process}_{recipe}",
                )
        st.markdown("---")

    submitted = st.button(
        "예측 · 평가 · 추천 실행 →",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.get("prediction_running", False),
    )

    return "recipe", inputs, targets, submitted, stage_defs, recipe, workbook


# ==============================================================================
# Output A. 현재 Recipe 품질 예측
# ==============================================================================
def show_prediction(result: dict):
    st.markdown("<div class='section-title'>현재 Recipe 품질 예측</div>", unsafe_allow_html=True)

    if result.get("_error"):
        st.error(f"예측할 수 없습니다: {result['_error']}")
        return

    cols = st.columns(4)
    items = [
        ("Top CD", result["Top CD"], "nm"), ("Mid CD", result["Mid CD"], "nm"),
        ("Bottom CD", result["Bottom CD"], "nm"), ("Depth", result["Depth"], "nm"),
    ]
    for col, (label, value, unit) in zip(cols, items):
        with col:
            render_metric_card(label, value, unit)

    st.markdown("**품질 지표**")
    q_cols = st.columns(4)
    with q_cols[0]:
        render_status_badge("CD Uniformity (CV%)", f'{result["CD Uniformity"]}%', get_cd_uniformity_status(result["CD Uniformity"]))
    with q_cols[1]:
        render_status_badge("Depth Uniformity (CV%)", f'{result["Depth Uniformity"]}%', get_depth_uniformity_status(result["Depth Uniformity"]))
    with q_cols[2]:
        render_status_badge("Overall Pass Rate", f'{result["Overall Spec Pass Rate"]}%', get_pass_rate_status(result["Overall Spec Pass Rate"]))
    with q_cols[3]:
        render_status_badge("Particle 발생 확률", f'{result["Particle Probability"]}%', get_particle_status(result["Particle"]))
    st.caption(
        f"※ Uniformity는 변동계수(CV%) 기준으로 값이 낮을수록 좋습니다. "
        f"모델 기준 상대 품질지수(v3) {result.get('_quality_index_v3_pct', '-')}점, "
        f"가장 취약한 위치: {result.get('_worst_zone', '-')} Zone."
    )
    if result.get("_warnings"):
        for w in result["_warnings"]:
            st.warning(w)

    chart_cols = st.columns(3)
    with chart_cols[0]:
        st.plotly_chart(build_cd_bar_chart(result), use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(build_gauge_chart("Overall Pass Rate", result["Overall Spec Pass Rate"]), use_container_width=True)
    with chart_cols[2]:
        st.plotly_chart(
            build_variation_gauge("CD Uniformity (CV%)", result["CD Uniformity"], good_th=1.8, warn_th=3.0, max_range=6),
            use_container_width=True,
        )


# ==============================================================================
# Output A-1. 목표 대비 진단 (목표 품질 모드 전용)
# ==============================================================================
def show_target_diagnosis(baseline_result: dict, targets: dict):
    """현재 Recipe 예측을 목표 품질과 비교해 핵심 미달 항목을 요약한다."""
    evaluation = evaluate_against_target(baseline_result, targets)
    failed_items = [name for name, satisfied in evaluation["satisfied"].items() if not satisfied]
    issue_text = " · ".join(failed_items) if failed_items else "모든 목표 기준 충족"

    st.markdown("<div class='section-title'>목표 대비 진단</div>", unsafe_allow_html=True)
    columns = st.columns(3)
    with columns[0]:
        render_score_hero(evaluation["score"]["total"], label="목표 대비 종합점수")
    with columns[1]:
        render_summary_card("주요 미달 항목", issue_text)
    with columns[2]:
        render_summary_card("예측상 취약 Zone", baseline_result.get("_worst_zone", "—"))


# ==============================================================================
# Output B. 목표 대비 Recipe 평가
# ==============================================================================
def show_target_evaluation(evaluation: dict):
    st.markdown("<div class='section-title'>목표 대비 Recipe 평가</div>", unsafe_allow_html=True)

    score = evaluation["score"]
    col_hero, col_sub = st.columns([1, 2])
    with col_hero:
        render_score_hero(score["total"])
    with col_sub:
        st.markdown("<br>", unsafe_allow_html=True)
        sub_cols = st.columns(3)
        with sub_cols[0]:
            render_subscore_card("Pass Rate 기여", score["pass_rate_score"], "가중치 60%")
        with sub_cols[1]:
            render_subscore_card("Uniformity 기여", score["uniformity_score"], "가중치 25%")
        with sub_cols[2]:
            render_subscore_card("목표 근접도 기여", score["target_proximity_score"], "가중치 15%")

    st.markdown("**항목별 목표 대비 오차**")
    err_cols = st.columns(4)
    for col, name in zip(err_cols, ["Top CD", "Mid CD", "Bottom CD", "Depth"]):
        with col:
            e = evaluation["errors"][name]
            sign = "+" if e > 0 else ""
            render_summary_card(f"{name} Error", f"{sign}{e:.1f}nm")

    st.markdown("**Spec 및 목표 만족 여부**")
    pill_cols = st.columns(4)
    for col, name in zip(pill_cols, ["CD Uniformity", "Depth Uniformity", "Pass Rate", "Defect Count"]):
        with col:
            render_pill_card(name, evaluation["satisfied"][name])

    st.markdown("**주요 미달 항목 및 개선 우선순위**")
    if evaluation["issues"]:
        html = "<div class='analysis-card'>" + "".join(
            f"<div>{i + 1}. {m}</div>" for i, m in enumerate(evaluation["issues"])
        ) + "</div>"
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.success("모든 목표 기준을 만족했습니다.")


# ==============================================================================
# Output C. 최적 Recipe 추천
# ==============================================================================
def show_recommendation(current_inputs: dict, recommendation: dict, recipe_df: pd.DataFrame, stage_defs):
    st.markdown("<div class='section-title'>최적 Recipe 추천</div>", unsafe_allow_html=True)

    if recommendation is None:
        st.info("추천할 다른 Recipe 이력이 없습니다.")
        return

    st.markdown(f"현재 조건(Equipment/Chamber)에서 알려진 Recipe 중 **{recommendation['recipe']}**의 종합 품질 점수가 가장 높게 예상됩니다.")

    diff_rows = build_stage_diff_table(current_inputs, recommendation["inputs"], recipe_df, stage_defs)
    if diff_rows:
        st.markdown("**추천 Recipe Parameter (현재값 대비 변경점)**")
        st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("현재 입력값과 추천 Recipe의 Stage 조건이 동일합니다.")

    st.markdown("**추천 Recipe 예상 품질 결과**")
    result = recommendation["result"]
    cols = st.columns(5)
    items = [
        ("Top CD", result["Top CD"], "nm"), ("Mid CD", result["Mid CD"], "nm"),
        ("Bottom CD", result["Bottom CD"], "nm"), ("Depth", result["Depth"], "nm"),
        ("Pass Rate", result["Overall Spec Pass Rate"], "%"),
    ]
    for col, (label, value, unit) in zip(cols, items):
        with col:
            render_metric_card(label, value, unit)

    render_score_hero(recommendation["score"]["total"], label=f"추천 Recipe '{recommendation['recipe']}' 예상 품질 점수")


# ==============================================================================
# Output D. 현재 Recipe와 추천 Recipe 비교
# ==============================================================================
def show_comparison(current_result: dict, current_score: dict, recommendation: dict):
    st.markdown("<div class='section-title'>현재 Recipe와 추천 Recipe 비교</div>", unsafe_allow_html=True)

    if recommendation is None:
        st.info("비교할 추천 Recipe가 없습니다.")
        return

    rec_result = recommendation["result"]
    rec_score = recommendation["score"]

    rows = [
        {"품질 항목": "Bottom CD (nm)", "현재 Recipe": current_result["Bottom CD"], "추천 Recipe": rec_result["Bottom CD"],
         "개선 효과": f"{rec_result['Bottom CD'] - current_result['Bottom CD']:+.1f}nm"},
        {"품질 항목": "Depth (nm)", "현재 Recipe": current_result["Depth"], "추천 Recipe": rec_result["Depth"],
         "개선 효과": f"{rec_result['Depth'] - current_result['Depth']:+.1f}nm"},
        {"품질 항목": "Pass Rate (%)", "현재 Recipe": current_result["Overall Spec Pass Rate"], "추천 Recipe": rec_result["Overall Spec Pass Rate"],
         "개선 효과": f"{rec_result['Overall Spec Pass Rate'] - current_result['Overall Spec Pass Rate']:+.1f}%p"},
        {"품질 항목": "종합 품질 점수", "현재 Recipe": current_score["total"], "추천 Recipe": rec_score["total"],
         "개선 효과": f"{rec_score['total'] - current_score['total']:+.1f}점"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(build_cd_comparison_chart(current_result, rec_result), use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(build_score_comparison_chart(current_score["total"], rec_score["total"]), use_container_width=True)


# ==============================================================================
# Output E. 파라미터별 조정 제안 (AI 추천 — Recipe 단위가 아니라 파라미터 단위)
# ==============================================================================
def show_parameter_recommendations(suggestion: dict, targets: dict, stage_defs: list):
    st.markdown("<div class='section-title'>파라미터별 조정 제안</div>", unsafe_allow_html=True)
    st.caption(
        "입력한 조건을 출발점으로 파라미터를 하나씩 바꾸며 목표 품질에 가까워지는 방향을 찾았습니다. "
        "추천 이유는 기존 품질 점수의 예측 전후 변화를 설명한 결과입니다."
    )

    if suggestion is None:
        return
    if suggestion.get("error"):
        st.error(f"추천을 계산할 수 없습니다: {suggestion['error']}")
        return

    recommendations = suggestion.get("recommendations") or []
    if not recommendations:
        st.info("현재 조건에서 더 좋아질 수 있는 파라미터 조정을 찾지 못했습니다.")
        return

    direction_kr = {"increase": "▲ 증가", "decrease": "▼ 감소"}
    top = recommendations[0]
    st.markdown(f"**추천 조정 {len(recommendations)}건**")
    st.caption(
        f"가장 큰 개선 후보 · {format_parameter_label(top['parameter'], stage_defs)} "
        f"{direction_kr.get(top['direction'], top['direction'])}"
    )

    rows = []
    for rank, recommendation in enumerate(recommendations, start=1):
        try:
            reason = generate_recommendation_reason(suggestion, recommendation, targets)
        except (KeyError, TypeError, ValueError):
            reason = "목표 품질 점수 개선"
        rows.append({
            "순위": rank,
            "Parameter": format_parameter_label(recommendation["parameter"], stage_defs),
            "방향": direction_kr.get(recommendation["direction"], recommendation["direction"]),
            "현재값": float(recommendation["current"]),
            "제안값": float(recommendation["proposed"]),
            "목표점수 개선": float(recommendation["target_composite_score_gain_pct_point"]),
            "추천 이유": reason,
        })
    display_df = pd.DataFrame(rows)

    def style_recommendation_row(row):
        base = "background-color:rgba(229,72,59,0.05);" if row["순위"] <= 3 else ""
        styles = [base] * len(row)
        column_index = {name: index for index, name in enumerate(row.index)}
        styles[column_index["제안값"]] = base + "color:#d83f33;background-color:#fff0e8;font-weight:700;"
        if row["목표점수 개선"] > 0:
            styles[column_index["목표점수 개선"]] = base + "color:#37835b;font-weight:700;"
        return styles

    styled = (
        display_df.style
        .format({"순위": "{:d}", "현재값": "{:g}", "제안값": "{:g}", "목표점수 개선": "{:+.2f}점"})
        .apply(style_recommendation_row, axis=1)
        .hide(axis="index")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(
        f"출발점 점수 {suggestion['baseline_target_score_pct']:.1f}점 · "
        "각 행은 다른 조건을 유지한 One-Factor-at-a-Time 결과이며, 여러 변경안은 아래에서 함께 재검증합니다."
    )


def show_combined_recipe_section(
    baseline_inputs: dict,
    baseline_result: dict,
    suggestion: dict,
    targets: dict,
    stage_defs: list,
    process: str,
    workbook: dict,
):
    """상위 추천안을 동시에 적용하고 기존 predict()로 조합 결과를 다시 검증한다."""
    recommendations = (suggestion or {}).get("recommendations") or []
    if not recommendations:
        return

    recommendation_count = len(recommendations)
    target_signature = tuple(sorted((key, float(value)) for key, value in targets.items()))

    def build_combo(allow_out_of_range: bool = False):
        applied_inputs, changed_parameters = apply_recommended_changes(baseline_inputs, recommendations, stage_defs)
        with st.status("추천 변경안 조합을 다시 예측하는 중...", expanded=False) as combo_status:
            combo_result = predict(
                applied_inputs,
                workbook["Wafer_Summary"],
                workbook["Recipe_Master"],
                process=process,
                allow_out_of_range=allow_out_of_range,
            )
            if combo_result.get("_error"):
                combo_status.update(label="조합 Recipe 확인이 필요합니다.", state="error", expanded=False)
            else:
                combo_status.update(label="조합 Recipe 재예측이 완료됐습니다.", state="complete", expanded=False)
        st.session_state.target_mode_combo = {
            "process": process,
            "equipment": baseline_inputs.get("equipment"),
            "chamber": baseline_inputs.get("chamber"),
            "base_recipe": baseline_inputs.get("recipe"),
            "target_signature": target_signature,
            "applied_inputs": applied_inputs,
            "changed_parameters": changed_parameters,
            "combo_result": combo_result,
            "change_count": recommendation_count,
        }

    if st.button(
        f"상위 {recommendation_count}개 변경안 조합 검증 →",
        type="primary",
        use_container_width=True,
        key="build_combined_recipe",
    ):
        build_combo(allow_out_of_range=False)

    combo = st.session_state.get("target_mode_combo")
    expected_state = (
        combo is not None
        and combo.get("process") == process
        and combo.get("equipment") == baseline_inputs.get("equipment")
        and combo.get("chamber") == baseline_inputs.get("chamber")
        and combo.get("base_recipe") == baseline_inputs.get("recipe")
        and combo.get("target_signature") == target_signature
    )
    if not expected_state:
        return

    combo_result = combo["combo_result"]
    if combo_result.get("_out_of_range"):
        st.warning(f"⚠️ {combo_result['_error']}")
        st.caption("학습 데이터 범위를 벗어난 조합이라 예측 신뢰도가 낮을 수 있습니다.")
        if st.button("그래도 이 조합으로 실행", key="force_out_of_range_combo"):
            build_combo(allow_out_of_range=True)
            st.rerun()
        return
    if combo_result.get("_error"):
        st.error(f"조합 Recipe를 예측할 수 없습니다: {combo_result['_error']}")
        return

    st.markdown("---")
    st.markdown("<div class='section-title'>추천 변경안 적용 Recipe</div>", unsafe_allow_html=True)
    st.caption(
        f"기준 Recipe · {combo['base_recipe']} | 변경 Parameter · {combo['change_count']}개 | "
        "추천안들을 동시에 적용한 뒤 다시 예측한 결과입니다."
    )

    table_df = pd.DataFrame(build_combined_recipe_table(baseline_inputs, combo["applied_inputs"], stage_defs))

    def style_combo_row(row):
        styles = [""] * len(row)
        if abs(row["기준 Recipe"] - row["추천 적용값"]) > 1e-9:
            column_index = {name: index for index, name in enumerate(row.index)}
            styles[column_index["추천 적용값"]] = "color:#d83f33;background-color:#fff0e8;font-weight:700;"
        return styles

    styled_table = (
        table_df.style
        .format({"기준 Recipe": "{:g}", "추천 적용값": "{:g}"})
        .apply(style_combo_row, axis=1)
        .hide(axis="index")
    )
    st.dataframe(styled_table, use_container_width=True, hide_index=True)

    baseline_evaluation = evaluate_against_target(baseline_result, targets)
    combo_evaluation = evaluate_against_target(combo_result, targets)
    comparison_rows = [
        {
            "품질 항목": "목표 대비 종합점수",
            "현재 Recipe": baseline_evaluation["score"]["total"],
            "변경안 적용 Recipe": combo_evaluation["score"]["total"],
            "변화": combo_evaluation["score"]["total"] - baseline_evaluation["score"]["total"],
        },
        {
            "품질 항목": "Overall Pass Rate [%]",
            "현재 Recipe": baseline_result["Overall Spec Pass Rate"],
            "변경안 적용 Recipe": combo_result["Overall Spec Pass Rate"],
            "변화": combo_result["Overall Spec Pass Rate"] - baseline_result["Overall Spec Pass Rate"],
        },
        {
            "품질 항목": "CD Uniformity [%]",
            "현재 Recipe": baseline_result["CD Uniformity"],
            "변경안 적용 Recipe": combo_result["CD Uniformity"],
            "변화": combo_result["CD Uniformity"] - baseline_result["CD Uniformity"],
        },
        {
            "품질 항목": "Depth Uniformity [%]",
            "현재 Recipe": baseline_result["Depth Uniformity"],
            "변경안 적용 Recipe": combo_result["Depth Uniformity"],
            "변화": combo_result["Depth Uniformity"] - baseline_result["Depth Uniformity"],
        },
        {
            "품질 항목": "Particle 발생 확률 [%]",
            "현재 Recipe": baseline_result["Particle Probability"],
            "변경안 적용 Recipe": combo_result["Particle Probability"],
            "변화": combo_result["Particle Probability"] - baseline_result["Particle Probability"],
        },
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(
        comparison_df.style.format(
            {"현재 Recipe": "{:.1f}", "변경안 적용 Recipe": "{:.1f}", "변화": "{:+.1f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )


# ==============================================================================
# 2-1 / 2-2. Process Dashboard — 조건 선택 + Summary
# ==============================================================================
def reset_process_dashboard_filters(process: str) -> None:
    """Process Dashboard의 필터를 해당 공정 기본값으로 되돌린다."""
    for key in (f"top_equipment_{process}", f"pd_chamber_{process}", f"pd_recipe_{process}"):
        st.session_state.pop(key, None)


def create_process_dashboard_equipment_selector(workbook: dict, process: str):
    """Equipment는 상단 ACTIVE PROCESS 옆 공용 선택을 그대로 쓰고, 그 Equipment의 모든 Chamber 데이터를 합쳐서 집계한다.
    (Chamber별로 대표 1개만 쓰면 나머지 Chamber 데이터가 화면에서 통째로 빠지는 문제가 있어,
    "대표 Chamber 자동 선택" 대신 "해당 Equipment의 Chamber 전체 통합"으로 바꿨다.)"""
    wafer_df = workbook["Wafer_Summary"]

    equipment_options = sorted(wafer_df["Equipment_Model"].unique())
    equipment = st.session_state.get(f"top_equipment_{process}") or (equipment_options[0] if equipment_options else None)

    chambers = sorted(wafer_df.loc[wafer_df["Equipment_Model"] == equipment, "Chamber_ID"].dropna().unique())
    chambers_text = escape(", ".join(chambers)) if chambers else "—"
    st.markdown(
        f"""
        <div class="dashboard-context-strip">
            <span>ACTIVE VIEW</span>
            <strong>{escape(PROCESS_LABELS[process])} · {escape(str(equipment))} · Chamber {len(chambers)}개 통합 ({chambers_text})</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not chambers:
        equipment_chamber_wafer = wafer_df.iloc[0:0]
    else:
        equipment_chamber_wafer = wafer_df[wafer_df["Equipment_Model"] == equipment]

    return equipment, chambers, equipment_chamber_wafer


def filter_dashboard_by_rev(
    workbook: dict, equipment_chamber_wafer: pd.DataFrame,
    equipment: str, chambers: list, selected_rev: str | None,
):
    """선택된 Rev로 Wafer_Summary / Site_Level_Raw를 좁혀 상세 섹션에 전달한다.
    Chamber는 하나로 좁히지 않고 해당 Equipment의 모든 Chamber를 그대로 포함한다."""
    if not chambers or selected_rev is None:
        empty_wafer = equipment_chamber_wafer.iloc[0:0]
        return empty_wafer, workbook["Site_Level_Raw"].iloc[0:0]

    filtered_wafer = equipment_chamber_wafer[equipment_chamber_wafer["Recipe_Version"] == selected_rev]

    site_df = workbook["Site_Level_Raw"]
    filtered_site = site_df[
        (site_df["Equipment_Model"] == equipment)
        & (site_df["Chamber_ID"].isin(chambers))
        & (site_df["Recipe_Version"] == selected_rev)
    ]
    return filtered_wafer, filtered_site


def show_process_summary(filtered_wafer: pd.DataFrame):
    render_dashboard_section_title("선택 Recipe Summary", "summary")
    if filtered_wafer.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    total_wafers = filtered_wafer["Wafer_ID"].nunique()
    avg_pass_rate = filtered_wafer["Overall_Spec_Pass_Rate_pct"].mean()
    avg_cd = filtered_wafer[["Top_CD_Mean_nm", "Mid_CD_Mean_nm", "Bottom_CD_Mean_nm"]].mean().mean()
    avg_depth = filtered_wafer["Depth_Mean_nm"].mean()
    avg_uniformity = filtered_wafer[
        ["Top_CD_Uniformity_pct", "Mid_CD_Uniformity_pct", "Bottom_CD_Uniformity_pct", "Depth_Uniformity_pct"]
    ].mean().mean()
    particle_count = int((filtered_wafer["Total_Defect_Count"] > PARTICLE_DEFECT_THRESHOLD).sum())

    chambers_used = sorted(filtered_wafer["Chamber_ID"].dropna().unique())
    row1 = st.columns(5, gap="medium")
    row1_items = [
        ("Equipment", filtered_wafer["Equipment_Model"].iloc[0]),
        ("Chamber (통합)", ", ".join(chambers_used) if chambers_used else "—"),
        ("Recipe", filtered_wafer["Recipe_Version"].iloc[0]),
        ("Total Wafer 수", f"{total_wafers}"),
        ("평균 Pass Rate", f"{avg_pass_rate:.1f}%"),
    ]
    for col, (label, value) in zip(row1, row1_items):
        with col:
            render_summary_card(label, value, variant="summary-card-recipe")

    row2 = st.columns(4, gap="medium")
    row2_items = [
        ("평균 CD", f"{avg_cd:.1f} nm"),
        ("평균 Depth", f"{avg_depth:.1f} nm"),
        ("평균 Uniformity(CV%)", f"{avg_uniformity:.2f}%"),
        (f"Particle 발생 건수 (Defect>{PARTICLE_DEFECT_THRESHOLD})", f"{particle_count} 건"),
    ]
    for col, (label, value) in zip(row2, row2_items):
        with col:
            render_summary_card(label, value, variant="summary-card-recipe")


# ==============================================================================
# 2-3. Rev별 스코어링 순위 (멘토 피드백: 현재 어떤 레시피가 가장 좋은지 바로 알 수 있게)
# ==============================================================================
_SCOREBOARD_BEST_BOLD_COLS = {"순위", "Recipe", "종합 점수"}


def _style_scoreboard_rows(display_df: pd.DataFrame, recipe_order: list, best_rev: str, selected_rev: str | None):
    """1위(Best) 행은 핵심 값을 항상 bold로, 선택 상태에 따라 배경을 다르게 강조한다.
    - 선택된 행이 Best면: 연한 금색 배경.
    - 선택된 행이 Best가 아니면: 선택 행만 accent 배경(Best 행의 전체 배경 강조는 해제),
      Best 행은 Recipe 셀의 🏆 표기와 bold로만 계속 구분된다."""
    def _row_style(row):
        recipe = recipe_order[row.name]
        is_best = recipe == best_rev
        is_selected = recipe == selected_rev
        if is_selected and is_best:
            bg = f"background-color: {COLORS['gold_soft']};"
        elif is_selected:
            bg = f"background-color: {COLORS['accent_soft']};"
        else:
            bg = ""
        return [
            bg + (" font-weight: 800;" if is_best and col in _SCOREBOARD_BEST_BOLD_COLS else "")
            for col in row.index
        ]
    return display_df.style.apply(_row_style, axis=1)


def show_recipe_scoreboard(equipment_chamber_wafer: pd.DataFrame, targets: dict, recipe_df: pd.DataFrame, process: str):
    """Rev별 종합 품질 점수 테이블. 순위 산정과 정렬만 담당하는 표시 레이어이며,
    종합 점수 계산식(compute_composite_score)과 실측 집계 로직(score_recipe_versions)은 그대로 둔다."""
    render_dashboard_section_title("Recipe별 종합 품질 점수", "trophy", "blue")
    st.caption("모델 예측이 아니라 실제 측정된 Wafer 결과를 Recipe(Rev)별로 집계한 종합 품질 점수입니다.")

    rev_state_key = "dashboard_selected_rev"
    table_key = f"pd_rev_table_{process}"

    if equipment_chamber_wafer.empty:
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")
        st.session_state.pop(rev_state_key, None)
        return pd.DataFrame(), None

    scoreboard = score_recipe_versions(equipment_chamber_wafer, targets)
    if scoreboard.empty:
        st.info("Rev 스코어를 계산할 데이터가 없습니다.")
        st.session_state.pop(rev_state_key, None)
        return scoreboard, None

    # 동점이면 기존 Recipe/Rev 자연 순서(Base, Rev1, Rev2, ... Rev9, Rev10 ...)로 안정적으로 정렬.
    # 종합 점수 값 자체는 건드리지 않고, 표시 순서(순위)만 결정한다.
    natural_order = {r: i for i, r in enumerate(ordered_recipe_versions(recipe_df, scoreboard["Recipe"].unique()))}
    scoreboard = scoreboard.assign(_natural_order=scoreboard["Recipe"].map(natural_order))
    scoreboard = (
        scoreboard.sort_values(["종합 점수", "_natural_order"], ascending=[False, True], kind="mergesort")
        .drop(columns="_natural_order")
        .reset_index(drop=True)
    )

    options = scoreboard["Recipe"].tolist()
    best_rev = options[0]

    # 위젯을 다시 그리기 전에, 이번 rerun에 이미 반영된 클릭 결과를 먼저 읽어 선택 상태를 갱신한다.
    # (st.session_state[table_key]는 dataframe 위젯이 자체 관리하는 값이라 프로그램적으로 쓸 수는 없고 읽기만 가능하다.)
    pending = st.session_state.get(table_key)
    if pending is not None:
        clicked_rows = pending.get("selection", {}).get("rows", [])
        if clicked_rows:
            st.session_state[rev_state_key] = scoreboard.iloc[clicked_rows[0]]["Recipe"]

    if st.session_state.get(rev_state_key) not in options:
        st.session_state[rev_state_key] = best_rev
    selected_rev = st.session_state[rev_state_key]

    display_df = pd.DataFrame({
        "순위": scoreboard.index + 1,
        "Recipe": [f"🏆 {r}" if r == best_rev else r for r in scoreboard["Recipe"]],
        "종합 점수": scoreboard["종합 점수"],
        "Wafer 수": scoreboard["Wafer 수"],
        "Top CD": scoreboard["Top CD"],
        "Mid CD": scoreboard["Mid CD"],
        "Bottom CD": scoreboard["Bottom CD"],
        "Depth": scoreboard["Depth"],
        "Overall Spec Pass Rate": scoreboard["Overall Spec Pass Rate"],
    })
    styled = _style_scoreboard_rows(display_df, options, best_rev, selected_rev)

    st.plotly_chart(build_recipe_score_chart(scoreboard, selected_rev), use_container_width=True)
    show_dashboard_best_case(scoreboard)
    st.markdown(
        f"<p style='color:{COLORS['text_primary']}; font-weight:800; margin:0 0 0.5rem 0;'>"
        "✅ 선택 — 왼쪽 체크박스를 누르면 해당 Recipe의 상세 정보를 확인할 수 있습니다.</p>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        styled,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key=table_key,
        column_config={
            "순위": st.column_config.NumberColumn("순위", format="%d", width="small"),
            "Recipe": st.column_config.TextColumn("Recipe"),
            "종합 점수": st.column_config.NumberColumn("종합 점수", format="%.1f"),
            "Wafer 수": st.column_config.NumberColumn("Wafer 수", format="%d"),
            "Top CD": st.column_config.NumberColumn("Top CD", format="%.1f"),
            "Mid CD": st.column_config.NumberColumn("Mid CD", format="%.1f"),
            "Bottom CD": st.column_config.NumberColumn("Bottom CD", format="%.1f"),
            "Depth": st.column_config.NumberColumn("Depth", format="%.1f"),
            "Overall Spec Pass Rate": st.column_config.NumberColumn("Overall Spec Pass Rate", format="%.1f"),
        },
    )
    return scoreboard, selected_rev


def show_dashboard_best_case(scoreboard: pd.DataFrame) -> None:
    """스코어보드 1위 Recipe/Rev를 Best Case 카드로 강조해, 진입 즉시 최고 성능 Rev를 알 수 있게 한다."""
    if scoreboard is None or scoreboard.empty:
        return
    best = scoreboard.iloc[0]
    with st.container(border=True):
        st.markdown(dashboard_best_badge_html(escape(str(best["Recipe"]))), unsafe_allow_html=True)
        cols = st.columns(4)
        metrics = [
            ("종합 품질 점수", f"{best['종합 점수']:.1f}점"),
            ("Pass Rate", f"{best['Overall Spec Pass Rate']:.1f}%"),
            ("CD Uniformity", f"{best['CD Uniformity']:.2f}%"),
            ("Depth Uniformity", f"{best['Depth Uniformity']:.2f}%"),
        ]
        for col, (label, value) in zip(cols, metrics):
            with col:
                render_summary_card(label, value, variant="summary-card-best")


# ==============================================================================
# 2-4. Recipe 비교/차이 드릴다운 (멘토 피드백: 선택한 Recipe가 1위/평균 대비 얼마나 차이나는지)
# ==============================================================================
def show_recipe_comparison(scoreboard: pd.DataFrame, selected_rev: str | None) -> None:
    """스코어링 표에서 선택한 Recipe를 1위 Recipe·전체 평균과 비교해 Δ를 보여준다.
    scoreboard/selected_rev는 show_recipe_scoreboard()가 이미 계산해둔 값을 그대로 받아 쓴다."""
    render_dashboard_section_title("Recipe 비교", "score", "blue")
    if scoreboard is None or scoreboard.empty or selected_rev is None:
        st.info("비교할 Recipe가 없습니다.")
        return

    selected_row = scoreboard[scoreboard["Recipe"] == selected_rev].iloc[0]
    best_row = scoreboard.iloc[0]
    compare_cols = ["종합 점수", "Overall Spec Pass Rate", "CD Uniformity", "Depth Uniformity"]
    avg_row = scoreboard[compare_cols].mean()

    st.caption(f"선택한 Recipe **{selected_rev}**를 1위 Recipe **{best_row['Recipe']}** 및 전체 평균과 비교합니다.")

    metrics = [
        ("종합 점수", "종합 점수", "점"),
        ("Pass Rate", "Overall Spec Pass Rate", "%"),
        ("CD Uniformity", "CD Uniformity", "%"),
        ("Depth Uniformity", "Depth Uniformity", "%"),
    ]
    rows = [
        {
            "품질 항목": label,
            "선택 Recipe": f"{selected_row[col]:.2f}{unit}",
            "1위 대비 Δ": f"{selected_row[col] - best_row[col]:+.2f}{unit}",
            "전체 평균 대비 Δ": f"{selected_row[col] - avg_row[col]:+.2f}{unit}",
        }
        for label, col, unit in metrics
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.plotly_chart(
        build_rev_comparison_chart(
            selected_rev, selected_row["종합 점수"], best_row["Recipe"], best_row["종합 점수"], avg_row["종합 점수"],
        ),
        use_container_width=True,
    )


def show_rev_quality_trends(
    equipment_chamber_wafer: pd.DataFrame,
    scoreboard: pd.DataFrame,
    recipe_df: pd.DataFrame,
    targets: dict,
    selected_rev: str | None = None,
):
    """현재 Equipment/Chamber의 전체 Recipe를 Rev 순서로 집계해 비교한다."""
    render_dashboard_section_title("Rev별 품질 변화", "quality")
    st.caption(
        "선택한 Equipment의 실제 Wafer 결과를 Rev별로 집계했습니다. "
        "위 Recipe 필터와 관계없이 전체 Rev의 변화 방향을 비교합니다."
    )
    if equipment_chamber_wafer.empty or scoreboard is None or scoreboard.empty:
        st.info("Rev별 품질 변화를 표시할 데이터가 없습니다.")
        return

    recipe_order = ordered_recipe_versions(
        recipe_df,
        equipment_chamber_wafer["Recipe_Version"].unique(),
    )
    if not recipe_order:
        st.info("Rev별 품질 변화를 표시할 Recipe가 없습니다.")
        return

    rev_scoreboard = scoreboard.set_index("Recipe").reindex(recipe_order).dropna(how="all").reset_index()
    particle_rows = []
    for recipe_version in rev_scoreboard["Recipe"]:
        group = equipment_chamber_wafer[
            equipment_chamber_wafer["Recipe_Version"] == recipe_version
        ]
        particle_rows.append({
            "Recipe": recipe_version,
            "평균 Defect Count": float(group["Total_Defect_Count"].mean()),
            "Particle 발생 Wafer 수": int((group["Total_Defect_Count"] > PARTICLE_DEFECT_THRESHOLD).sum()),
            "Wafer 수": int(group["Wafer_ID"].nunique()) if "Wafer_ID" in group.columns else len(group),
        })
    rev_particle_df = pd.DataFrame(particle_rows)

    row1 = st.columns(2, gap="large")
    with row1[0]:
        st.plotly_chart(build_rev_cd_trend_chart(rev_scoreboard, targets, selected_rev), use_container_width=True)
    with row1[1]:
        st.plotly_chart(build_rev_depth_trend_chart(rev_scoreboard, targets, selected_rev), use_container_width=True)

    row2 = st.columns(2, gap="large")
    with row2[0]:
        st.plotly_chart(build_rev_uniformity_trend_chart(rev_scoreboard, selected_rev), use_container_width=True)
    with row2[1]:
        st.plotly_chart(build_rev_pass_rate_chart(rev_scoreboard, selected_rev), use_container_width=True)

    st.plotly_chart(build_rev_defect_chart(rev_particle_df, selected_rev), use_container_width=True)


# ==============================================================================
# 3. 품질 결과 시각화 (Wafer 단위 추이) — 각 그래프는 독립 카드
# ==============================================================================
def show_quality_visualization(filtered_wafer: pd.DataFrame, targets: dict):
    render_dashboard_section_title("Wafer 단위 품질 결과", "quality")
    if filtered_wafer.empty:
        return

    wafer_df = filtered_wafer.copy()
    if "Eval_Timestamp" in wafer_df.columns:
        wafer_df = wafer_df.sort_values("Eval_Timestamp")
    wafer_df["Wafer_Label"] = wafer_df["Lot_ID"].astype(str) + " / " + wafer_df["Wafer_ID"].astype(str)

    row1 = st.columns(2)
    with row1[0]:
        st.plotly_chart(build_cd_trend_chart(wafer_df, targets), use_container_width=True)
    with row1[1]:
        st.plotly_chart(build_depth_trend_chart(wafer_df, targets), use_container_width=True)

    row2 = st.columns(2)
    with row2[0]:
        st.plotly_chart(build_cd_uniformity_trend_chart(wafer_df), use_container_width=True)
    with row2[1]:
        st.plotly_chart(build_depth_uniformity_trend_chart(wafer_df), use_container_width=True)

    row3 = st.columns(2)
    with row3[0]:
        st.plotly_chart(build_pass_rate_trend_chart(wafer_df), use_container_width=True)
    with row3[1]:
        st.plotly_chart(build_particle_chart(wafer_df), use_container_width=True)


# ==============================================================================
# 4. Wafer 단면 Profile
# ==============================================================================
def show_wafer_profile(filtered_site: pd.DataFrame):
    render_dashboard_section_title("Wafer 단면 Profile", "map")
    if filtered_site.empty:
        st.info("선택한 조건에 해당하는 Site 데이터가 없습니다.")
        return
    st.caption("x축 = Point 번호. Edge → Center → Edge 순서라 Zone 평균보다 정확한 수치 비교가 쉽습니다.")
    profile_items = list(WAFER_MAP_METRICS.items())
    for row_start in range(0, len(profile_items), 2):
        profile_cols = st.columns(2, gap="large")
        for col, (label, metric_col) in zip(profile_cols, profile_items[row_start:row_start + 2]):
            with col:
                st.plotly_chart(build_wafer_profile_chart(filtered_site, metric_col, label), use_container_width=True)


# ==============================================================================
# 5. Zone 분석
# ==============================================================================
def show_zone_analysis(filtered_site: pd.DataFrame):
    render_dashboard_section_title("Zone별 분석", "zone")
    if filtered_site.empty:
        return None

    zone_summary = get_zone_summary(filtered_site)

    row1 = st.columns(2)
    with row1[0]:
        st.plotly_chart(build_zone_cd_chart(zone_summary), use_container_width=True)
    with row1[1]:
        st.plotly_chart(build_zone_depth_chart(zone_summary), use_container_width=True)

    row2 = st.columns(2)
    with row2[0]:
        st.plotly_chart(build_zone_spread_chart(zone_summary), use_container_width=True)
    with row2[1]:
        st.plotly_chart(build_zone_pass_rate_chart(zone_summary), use_container_width=True)

    st.plotly_chart(build_zone_defect_chart(zone_summary), use_container_width=True)
    st.caption("※ 이 데이터셋에는 Zone별 Uniformity%가 별도로 없어, 같은 Zone 내 Site간 표준편차(Spread)를 균일도 대체 지표로 사용했습니다.")

    return zone_summary


# ==============================================================================
# 6. AI 분석 (Process Dashboard, Rule Base) — 순서상 맨 위로 (멘토 피드백 반영)
# ==============================================================================
def show_dashboard_ai_analysis(filtered_wafer: pd.DataFrame, zone_summary: pd.DataFrame):
    render_dashboard_section_title("AI 분석", "ai", "blue")
    messages = generate_dashboard_analysis(filtered_wafer, zone_summary)
    html = "<div class='analysis-card'>" + "".join(f"<div>• {m}</div>" for m in messages) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


MATERIAL_CROSS_SECTION_COLORS = {
    "Mask": "#6D4C41", "SiO2": "#64B5F6", "SiON": "#4FC3F7", "SOC": "#A1887F",
    "Si": "#78909C", "PolySi": "#FFB74D", "Al": "#CFD8DC", "TiN": "#FFD54F",
    "Metal": "#C0CA33", "Substrate": "#4E342E", "_default": "#B39DDB",
}

NON_ETCH_MATERIALS = {"Mask", "Substrate"}


def _render_layer_cross_section(layer_materials, etch_indices):
    def block(material, etched, is_last):
        color = MATERIAL_CROSS_SECTION_COLORS.get(material, MATERIAL_CROSS_SECTION_COLORS["_default"])
        extra = (
            "opacity:0.45;"
            "background-image:repeating-linear-gradient(45deg, rgba(0,0,0,0.2) 0 6px, transparent 6px 12px);"
            if etched else ""
        )
        label = f"{material} (Etched)" if etched else material
        border = "" if is_last else "border-bottom:1px solid rgba(255,255,255,0.6);"
        return (
            f"<div style='background:{color}; {extra} {border} padding:16px 10px; "
            f"text-align:center; font-size:0.85rem; font-weight:700; color:#222;'>{label}</div>"
        )

    def stack(materials, etch_set):
        blocks = "".join(
            block(m, i in etch_set, i == len(materials) - 1) for i, m in enumerate(materials)
        )
        return (
            "<div style='border:1px solid rgba(0,0,0,0.15); border-radius:6px; overflow:hidden; "
            f"box-shadow:0 2px 6px rgba(0,0,0,0.12);'>{blocks}</div>"
        )

    etch_set = set(etch_indices)
    before_col, after_col = st.columns(2)
    with before_col:
        st.caption("입력 Layer 구조")
        st.markdown(stack(layer_materials, set()), unsafe_allow_html=True)
    with after_col:
        st.caption("Etch 결과 프리뷰 (Concept)")
        st.markdown(stack(layer_materials, etch_set), unsafe_allow_html=True)
    st.caption("현재 이미지는 입력한 Layer 구조를 바탕으로 생성한 개념 단면도입니다. 실제 Etch 형상 예측 결과는 아닙니다.")


def show_new_process_preview():
    """신규 공정 품질 예측 (Phase 1 스캐폴드) — 정식 AI 예측이 아니라
    입력한 레이어(물질) 순서를 기존 4개 공정의 Stage들과 대조해 참고값을 보여주고,
    향후 예측 기능이 붙을 자리(입력 폼·단면도·결과 카드)를 미리 구성한다."""
    st.markdown("<div class='section-title'>신규 공정 품질 예측</div>", unsafe_allow_html=True)
    st.caption(
        "아직 AI 모델이 없는 새 공정을, Etch 순서대로 레이어(물질)를 선택하고 조건을 입력해서 미리 감을 잡는 기능입니다. "
        "**AI 예측이 아니며 점수화도 하지 않습니다.**"
    )
    st.caption("예: SiO2를 먼저 Etch하고 PolySi를 그다음 Etch한다면 → 레이어 1 SiO2, 레이어 2 PolySi")

    st.markdown("#### STEP 1. 신규 공정 기본 정보")
    st.text_input("신규 공정명", key="new_process_name", placeholder="예: New Gate Etch Test")
    st.text_area("공정 설명", key="new_process_desc", placeholder="예: SiO2 Open 이후 PolySi를 Etch하는 신규 Gate 공정")
    current_process = st.session_state.get("process")
    current_equipment = st.session_state.get(f"top_equipment_{current_process}", "미선택")
    st.caption(f"현재 선택된 Equipment: **{current_equipment}** (사이드바 EQUIPMENT 선택값을 그대로 참고합니다)")

    st.markdown("#### STEP 2. Layer 구조")
    OTHER_MATERIAL_OPTION = "기타 (직접 입력)"
    STANDARD_MATERIALS = ["Mask", "SiO2", "PolySi", "Si", "SiON", "SOC", "Al", "TiN", "Metal", "Substrate"]
    material_options = sorted(
        set(STANDARD_MATERIALS)
        | {stage["material"] for stage_defs in PROCESS_STAGE_DEFS.values() for stage in stage_defs}
    ) + [OTHER_MATERIAL_OPTION]

    if "new_process_layer_count" not in st.session_state:
        st.session_state.new_process_layer_count = 2

    label_col, add_col, remove_col = st.columns([4, 1, 1])
    with label_col:
        st.caption(f"레이어 {st.session_state.new_process_layer_count}개")
    with add_col:
        if st.button("+ 레이어", disabled=st.session_state.new_process_layer_count >= 8, use_container_width=True):
            st.session_state.new_process_layer_count += 1
            st.rerun()
    with remove_col:
        if st.button("− 레이어", disabled=st.session_state.new_process_layer_count <= 1, use_container_width=True):
            st.session_state.new_process_layer_count -= 1
            st.rerun()

    st.caption("레이어 1이 가장 먼저 Etch되는(웨이퍼 표면과 가장 가까운) 레이어이며, 아래로 갈수록 나중에 Etch되는 레이어입니다.")

    layer_materials = []
    layer_count = st.session_state.new_process_layer_count
    for i in range(layer_count):
        row_label_col, row_input_col = st.columns([1, 5])
        with row_label_col:
            st.markdown(f"<div style='padding-top:1.9rem; font-weight:600;'>레이어 {i + 1}</div>", unsafe_allow_html=True)
        with row_input_col:
            selected_material = st.selectbox(
                f"레이어 {i + 1} 물질", material_options,
                key=f"new_process_layer_{i}", label_visibility="collapsed",
            )
            if selected_material == OTHER_MATERIAL_OPTION:
                custom_material = st.text_input(
                    f"레이어 {i + 1} 물질명 직접 입력", key=f"new_process_layer_{i}_custom",
                    label_visibility="collapsed", placeholder="예: HfO2",
                ).strip()
                selected_material = custom_material or OTHER_MATERIAL_OPTION
        layer_materials.append(selected_material)
        if i < layer_count - 1:
            st.markdown("<div style='text-align:center; color:#999; font-size:1.3rem;'>↓</div>", unsafe_allow_html=True)

    etch_indices = [i for i, m in enumerate(layer_materials) if m not in NON_ETCH_MATERIALS]

    st.markdown("#### STEP 3. Etch Recipe 입력")
    if not etch_indices:
        st.caption("Etch 대상 레이어가 없습니다 (Mask/Substrate만으로 구성됨).")
    for step_no, i in enumerate(etch_indices, start=1):
        material = layer_materials[i]
        with st.expander(f"Step {step_no} · {material} Etch", expanded=False):
            time_col, rf_col, pressure_col = st.columns(3)
            with time_col:
                st.number_input("Time (sec)", min_value=0.0, step=1.0, key=f"new_process_time_{i}")
            with rf_col:
                st.number_input("RF Bias (W)", min_value=0.0, step=10.0, key=f"new_process_rf_{i}")
            with pressure_col:
                st.number_input("Pressure (mT)", min_value=0.0, step=1.0, key=f"new_process_pressure_{i}")
            with st.expander("상세 Recipe 조건 (Gas)", expanded=False):
                st.text_input("Gas Type", key=f"new_process_gas_type_{i}", placeholder="예: CHF3 / C4F8 / O2")
                st.text_input("Gas Flow", key=f"new_process_gas_flow_{i}", placeholder="예: 30 / 10 / 5 sccm")

    result = find_layer_references(layer_materials)
    layer_matches = result["layer_matches"]
    overall = result["overall_reference"]

    st.markdown(f"#### STEP 4. 기존 유사 Recipe 참고 ({len(layer_matches)}개 레이어)")
    st.markdown("###### 레이어별 참고 Recipe 조건")
    for i, (material_label, layer) in enumerate(zip(layer_materials, layer_matches), start=1):
        with st.expander(f"레이어 {i}: {material_label}", expanded=True):
            if not layer["matches"]:
                st.warning("이 물질과 겹치는 기존 Stage를 찾지 못했습니다.")
                continue
            for match in layer["matches"]:
                st.markdown(f"**{match['process_label']} · {match['stage_label']}**")
                param_cols = st.columns(len(match["param_ranges"]) + len(match["gas_ranges"]))
                col_idx = 0
                for label, (lo, hi) in match["param_ranges"].items():
                    with param_cols[col_idx]:
                        render_summary_card(label, f"{lo:g} ~ {hi:g}")
                    col_idx += 1
                for label, (lo, hi) in match["gas_ranges"].items():
                    with param_cols[col_idx]:
                        render_summary_card(label, f"{lo:g} ~ {hi:g}")
                    col_idx += 1
            st.caption("위 범위는 해당 Stage가 16개 Recipe에서 실제로 관측된 값의 최소~최대입니다 (예측값 아님).")

    st.markdown("###### 종합 참고 품질 (가장 비슷한 기존 공정)")
    if overall is None:
        st.warning("입력한 레이어 구성과 겹치는 기존 공정을 찾지 못해 종합 참고 품질을 계산할 수 없습니다.")
    else:
        st.markdown(
            f"요청한 {overall['requested_count']}개 레이어 중 **{overall['overlap_count']}개**가 겹치는, "
            f"가장 비슷한 기존 공정: **{overall['process_label']}** "
            f"(이 공정의 Stage 물질 구성: {', '.join(overall['process_materials'])})"
        )
        ref_cols = st.columns(6)
        ref_items = [
            ("Top CD (nm)", overall["avg_top_cd_nm"]), ("Mid CD (nm)", overall["avg_mid_cd_nm"]),
            ("Bottom CD (nm)", overall["avg_bottom_cd_nm"]), ("Depth (nm)", overall["avg_depth_nm"]),
            ("Uniformity (%)", overall["avg_uniformity_pct"]), ("Pass Rate (%)", overall["avg_pass_rate_pct"]),
        ]
        for col, (label, value) in zip(ref_cols, ref_items):
            with col:
                render_summary_card(label, value)
        st.caption(
            f"위 수치는 {overall['process_label']} 공정 전체 Wafer {overall['wafer_count']}장의 실측 평균입니다 — "
            "입력한 레이어 조합 전용 예측값이 아니라, 가장 비슷한 기존 공정 하나를 그대로 보여주는 참고값입니다."
        )

    st.markdown("#### STEP 5. Before / After 단면도 (Concept Preview)")
    _render_layer_cross_section(layer_materials, etch_indices)

    st.markdown("#### STEP 6. 신규 공정 품질 예측")
    if st.button("신규 공정 품질 예측", type="primary"):
        st.info(
            "임의의 새 Layer 조합에 대한 진짜 품질 예측 모델은 아직 없습니다 "
            "(Stage 단위로 분해된 품질 데이터가 없고, 기존 공정 4개만으로는 조합 다양성이 부족합니다). "
            "대신 아래 STEP 7은 가장 비슷한 기존 공정의 실측 평균을 예시로 보여줍니다."
        )

    st.markdown("#### STEP 7. 예상 품질")
    if overall is None:
        st.warning("참고할 만큼 겹치는 기존 공정을 찾지 못해 예시 품질을 보여줄 수 없습니다.")
    else:
        ref_items = [
            ("Top CD (nm)", overall["avg_top_cd_nm"]), ("Mid CD (nm)", overall["avg_mid_cd_nm"]),
            ("Bottom CD (nm)", overall["avg_bottom_cd_nm"]), ("Depth (nm)", overall["avg_depth_nm"]),
            ("Uniformity (%)", overall["avg_uniformity_pct"]), ("Pass Rate (%)", overall["avg_pass_rate_pct"]),
        ]
        for row_start in range(0, len(ref_items), 3):
            stub_cols = st.columns(3)
            for col, (label, value) in zip(stub_cols, ref_items[row_start:row_start + 3]):
                with col:
                    render_summary_card(label, value)
        st.caption(
            f"위 수치는 실제 예측값이 아니라, 가장 비슷한 기존 공정(**{overall['process_label']}**)의 "
            f"Wafer {overall['wafer_count']}장 실측 평균을 예시로 가져온 것입니다. "
            "진짜 신규 공정 예측 모델은 Layer 단위 데이터 축적 후 연결할 예정입니다."
        )


# ==============================================================================
# 7. Parameter 변경 이력 조회 (Parameter Change History) — 과거 파라미터 변경 이력 + 당시 품질 조회
# ==============================================================================
def create_parameter_history_selectors(workbook: dict, process: str):
    st.markdown(
        """
        <div class="app-shell-header">
            <div class="app-shell-eyebrow">PARAMETER CHANGE HISTORY</div>
            <h1 class="app-shell-title" style="font-size:1.9rem;">Parameter Change History</h1>
            <div class="app-shell-subtitle">저장된 레시피의 파라미터 변경 이력과 당시 품질 결과를 조회합니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    equipment_options = ["전체"] + get_equipment_models(workbook["Wafer_Summary"])

    col1, col2, col3 = st.columns(3)
    with col1:
        equipment = st.selectbox("Equipment Model", equipment_options, key=f"ph_equipment_{process}")
    with col2:
        param_type = st.selectbox("변경 파라미터", ["전체"] + PARAM_TYPE_LABELS, key=f"ph_param_type_{process}")
    with col3:
        detail_cols = get_detail_param_options(workbook["Recipe_Master"], param_type)
        all_label = f"전체 {param_type}" if param_type and param_type != "전체" else "전체"
        detail_labels = [all_label] + [format_detail_param_label(c, process) for c in detail_cols]
        # key에 process와 param_type을 포함시켜, 공정/유형을 바꿔 옵션 목록이 달라져도 이전
        # 선택값이 새 목록에 없어서 나는 StreamlitAPIException을 피한다 (매번 새 위젯 취급).
        detail_choice = st.selectbox(
            "세부 파라미터 (선택사항)", detail_labels,
            key=f"ph_detail_{process}_{param_type}",
            disabled=not detail_cols,
        )
        detail_param = detail_cols[detail_labels.index(detail_choice) - 1] if detail_choice != all_label else None

    submitted = st.button("이력 조회", key=f"ph_search_{process}", type="primary")
    return equipment, param_type, detail_param, submitted


def _strip_blank_lines(html: str) -> str:
    """조립된 HTML 문자열에서 빈 줄(공백만 있는 줄 포함)을 제거한다.

    카드/행 HTML은 선택적 조각({badge}, {eval_period_html} 등)이 빈 문자열일 때 그 줄이
    공백만 남는데, Streamlit의 마크다운 파서는 HTML 블록 중간의 빈 줄을 블록의 끝으로
    해석해 버린다 — 그러면 그 뒤 태그들이 렌더링되지 않고 그대로 텍스트로 노출된다."""
    return "\n".join(line for line in html.split("\n") if line.strip())


def _render_html(html: str) -> None:
    st.markdown(_strip_blank_lines(html), unsafe_allow_html=True)


def _history_value_text(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return escape(str(value))


def _history_metric_text(value, unit: str = "", decimals: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}{unit}"


def _no_negative_zero(value: float, decimals: int = 1) -> float:
    """반올림 후 -0.0이 되는 값을 0.0으로 바로잡는다 (예: -0.03 -> "-0.0"으로 보이는 문제 방지)."""
    return round(value, decimals) + 0.0


def _history_value_change_text(row: dict) -> str:
    """"9.4 → 10.0 s" 처럼 단위를 값 뒤에 한 번만 붙인다."""
    text = f"{_history_value_text(row['prev_value'])} → {_history_value_text(row['new_value'])}"
    unit = row.get("unit")
    if unit and (row["prev_value"] is not None or row["new_value"] is not None):
        text += f" {unit}"
    return text


def _history_delta_text(row: dict) -> str:
    if row["abs_delta"] is None:
        return "N/A"
    sign = "+" if row["abs_delta"] > 0 else ""
    pct_text = f"{sign}{row['pct_change']:.1f}%" if row["pct_change"] is not None else "N/A"
    return f"{sign}{row['abs_delta']:g} ({pct_text})"


def _history_delta_class(row: dict) -> str:
    if row["abs_delta"] is None:
        return ""
    return "is-up" if row["abs_delta"] > 0 else "is-down" if row["abs_delta"] < 0 else ""


def _major_change_row_html(row: dict) -> str:
    """카드의 "주요 변경" 한 줄 — 값만 보여주고 변화량/%는 넣지 않는다(상세 보기에서만)."""
    selected_class = " is-selected" if row.get("selected") else ""
    badge = '<span class="history-select-badge">선택</span>' if row.get("selected") else ""
    return _strip_blank_lines(f"""
        <div class="history-major-row{selected_class}">
            <span class="param-label">{escape(row['short_label'])}</span>
            <span>
                <span class="value-change">{_history_value_change_text(row)}</span>
                {badge}
            </span>
        </div>
        """)


def _detail_row_lg_html(row: dict) -> str:
    """상세 보기의 큰 파라미터 변경 행 — 라벨/뱃지가 첫 줄, 값 변화/변화량이 둘째 줄."""
    selected_class = " is-selected" if row.get("selected") else ""
    badge = '<span class="history-select-badge">선택 파라미터</span>' if row.get("selected") else ""
    return _strip_blank_lines(f"""
        <div class="history-param-row-lg{selected_class}">
            <div class="row-top">
                <span class="param-label">{escape(row['label'])}</span>
                {badge}
            </div>
            <div class="row-bottom">
                <span class="value-change">{_history_value_text(row['prev_value'])} → {_history_value_text(row['new_value'])}</span>
                <span class="delta {_history_delta_class(row)}">{escape(_history_delta_text(row))}</span>
            </div>
        </div>
        """)


def render_history_revision_card(entry: dict) -> None:
    """Revision 카드 — Change Notes 전문은 넣지 않고, 실제 파라미터 값 변화를 바로 보여준다.
    세부 파라미터를 골랐으면 그 항목을 맨 위에 강조하고(같은 유형의 다른 Step은 그 아래),
    "전체"로 조회했으면 최대 2개까지만 보여주고 나머지는 "외 N건"으로 축약한다."""
    revision = entry["revision"]
    quality = entry["quality"]
    card_rows = entry["card_rows"]
    has_selection = any(row.get("selected") for row in card_rows)

    if has_selection:
        shown_rows, more_count = card_rows, 0
    else:
        shown_rows, more_count = card_rows[:2], max(0, len(card_rows) - 2)

    major_rows_html = "".join(_major_change_row_html(row) for row in shown_rows)
    more_html = f'<div class="history-major-more">외 {more_count}건</div>' if more_count else ""

    eval_period = quality.get("eval_period")
    eval_period_html = f'<span class="history-eval-period">{escape(eval_period)}</span>' if eval_period else "<span></span>"

    pass_rate_text = _history_metric_text(quality["pass_rate_mean"], "%")
    equipment_text = " / ".join(quality["equipment_models"]) if quality["equipment_models"] else "—"

    _render_html(f"""
        <div class="history-card">
            <div class="history-card-head">
                <span class="history-card-head-title">
                    <span class="revision">{escape(str(revision))}</span>
                    {history_tag_badge_html(entry.get("stage_tag"))}
                </span>
                {eval_period_html}
            </div>
            <div class="history-major-heading">주요 변경</div>
            {major_rows_html}
            {more_html}
            <div class="history-card-footer">
                <span>Pass Rate <span class="highlight">{pass_rate_text}</span></span>
                <span>평가 Wafer <span class="highlight">{quality['wafer_count']}장</span></span>
                <span class="highlight">{escape(equipment_text)}</span>
            </div>
        </div>
        """)

    with st.expander(f"{revision} 변경 상세 보기"):
        render_history_detail_view(entry)


def render_history_detail_view(entry: dict) -> None:
    """상세 보기 전체 — 1행 2열(왼쪽 변경 상세 / 오른쪽 품질 변화 요약) → 당시 품질 KPI →
    비교표 → 평가 환경 순서로, 스크롤 없이 "무엇을 바꿨고 품질이 어떻게 달라졌는지" 먼저 보이게 한다."""
    revision = entry["revision"]
    notes = entry.get("change_notes")

    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        with st.container(border=True):
            st.markdown(f"#### {revision} 변경 상세")
            st.markdown("**변경 목적**")
            st.write(notes if isinstance(notes, str) and notes.strip() else "기록 없음")
            st.markdown("**파라미터 변경 상세**")
            for row in entry["detail_rows"]:
                _render_html(_detail_row_lg_html(row))

    with right_col:
        with st.container(border=True):
            st.markdown("#### 이전 Revision 대비 품질 변화")
            render_history_quality_change_summary(entry)

    st.markdown("")
    st.markdown("##### 당시 품질")
    render_history_kpi_section(entry["quality"], entry.get("kpi_target_diff"))

    st.markdown("##### 변경 전/후 실제 품질 비교")
    render_history_comparison_table(entry)

    st.markdown("##### 평가 환경")
    quality = entry["quality"]
    st.write(f"Equipment Model: {', '.join(quality['equipment_models']) if quality['equipment_models'] else '—'}")
    st.write(f"Chamber ID: {', '.join(quality['chambers']) if quality['chambers'] else '—'}")
    st.write(f"평가 기간: {quality['eval_period'] or '—'}")


def _comparison_item_text(row: dict) -> tuple:
    """개선/확인 필요 요약 한 줄의 (지표 라벨, 값 변화 텍스트)를 만든다.
    CD/Depth는 Target과의 절대 오차(부호 없이) 변화로, 나머지는 실측값 자체의 변화로 보여준다."""
    if row["mode"] == "error":
        label = f"{row['label']} Target 오차"
        if row["prev_error"] is None or row["curr_error"] is None:
            return label, "데이터 없음"
        return label, f"{abs(row['prev_error']):.1f} {row['unit']} → {abs(row['curr_error']):.1f} {row['unit']}"

    if row["prev_value"] is None or row["curr_value"] is None:
        return row["label"], "데이터 없음"
    delta_unit = "%p" if row["unit"] == "%" else row["unit"]
    return (
        row["label"],
        f"{row['prev_value']:.1f}{row['unit']} → {row['curr_value']:.1f}{row['unit']} "
        f"({_no_negative_zero(row['delta']):+.1f}{delta_unit})",
    )


def render_history_quality_change_summary(entry: dict) -> None:
    """오른쪽 카드 본문 — 표보다 결론(개선/확인 필요)을 먼저 보여준다.
    Change_Notes는 전혀 참고하지 않고 실제 Wafer_Summary 집계값만 사용한다."""
    prev_revision = entry.get("prev_revision")
    comparison = entry.get("comparison")

    if prev_revision:
        st.markdown(f'<div class="history-quality-subtitle">{escape(str(prev_revision))} → {escape(str(entry["revision"]))}</div>', unsafe_allow_html=True)

    if not comparison:
        st.info("이전 Revision 평가 데이터가 없어 품질 변화를 비교할 수 없습니다.")
        return

    improved = [r for r in comparison if r["classification"] == "개선"]
    attention = [r for r in comparison if r["classification"] == "확인 필요"]

    if improved:
        st.markdown('<div class="history-quality-section-heading is-improve">개선</div>', unsafe_allow_html=True)
        for row in improved:
            label, value_text = _comparison_item_text(row)
            st.markdown(
                f"""
                <div class="history-quality-item is-improve">
                    <span class="icon">✓</span>
                    <span class="body">
                        <div class="metric-label">{escape(label)}</div>
                        <div class="metric-values">{escape(value_text)}</div>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if attention:
        st.markdown('<div class="history-quality-section-heading is-attention">확인 필요</div>', unsafe_allow_html=True)
        for row in attention:
            label, value_text = _comparison_item_text(row)
            st.markdown(
                f"""
                <div class="history-quality-item is-attention">
                    <span class="icon">△</span>
                    <span class="body">
                        <div class="metric-label">{escape(label)}</div>
                        <div class="metric-values">{escape(value_text)}</div>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not improved and not attention:
        st.caption("이전 Revision과 비교했을 때 뚜렷한 변화가 없습니다 (모든 지표가 오차 범위 내에서 유지).")


def _kpi_card_html(label: str, value_text: str, subtext: str = "") -> str:
    """당시 품질" 카드 하나. subtext가 없어도 빈 영역을 그대로 두어(칸을 지우지 않음)
    카드 8개의 높이가 CSS(min-height)로 항상 맞춰지게 한다."""
    return f"""
        <div class="kpi-card">
            <div class="kpi-label">{escape(label)}</div>
            <div class="kpi-value">{escape(value_text)}</div>
            <div class="kpi-subtext">{escape(subtext)}</div>
        </div>
        """


def _kpi_cd_depth_card(label: str, key: str, actual_raw, kpi_target_diff: dict | None) -> str:
    """CD/Depth 카드 — 평균과 Target 오차를 "같은 반올림 결과"로 계산해서, 화면에 보이는
    두 숫자를 사용자가 직접 빼봐도 항상 맞게 만든다(예: 249.9 / Target -0.1 = 250.0 - 0.1... 이 아니라
    반올림된 249.9와 반올림된 Target을 먼저 만들고 그 차이를 다시 반올림해 보여준다)."""
    if actual_raw is None:
        return _kpi_card_html(label, "—")
    rounded_actual = round(actual_raw, 1)
    subtext = ""
    target_raw = (kpi_target_diff or {}).get(key, {}).get("target")
    if target_raw is not None:
        rounded_target = round(target_raw, 1)
        delta = _no_negative_zero(rounded_actual - rounded_target)
        subtext = f"Target {delta:+.1f} nm"
    return _kpi_card_html(label, _history_metric_text(rounded_actual, " nm"), subtext)


def render_history_kpi_section(quality: dict, kpi_target_diff: dict | None) -> None:
    """당시 품질 KPI — 8개 카드를 하나의 CSS Grid 컨테이너에 렌더링한다(Streamlit st.columns를
    쓰지 않아 열 너비 어긋남·카드 겹침이 생기지 않는다). 데스크톱 4열 x 2행, 태블릿 2열, 모바일 1열
    (.kpi-grid의 media query가 처리). CD/Depth에는 Target 대비 오차를 함께 보여준다
    (Spec 범위가 없어 충족/이탈 판정은 하지 않는다)."""
    cards_html = "".join([
        _kpi_cd_depth_card("Top CD 평균", "top_cd", quality["top_cd_mean"], kpi_target_diff),
        _kpi_cd_depth_card("Mid CD 평균", "mid_cd", quality["mid_cd_mean"], kpi_target_diff),
        _kpi_cd_depth_card("Bottom CD 평균", "bottom_cd", quality["bottom_cd_mean"], kpi_target_diff),
        _kpi_cd_depth_card("Depth 평균", "depth", quality["depth_mean"], kpi_target_diff),
        _kpi_card_html("Overall Pass Rate", _history_metric_text(quality["pass_rate_mean"], "%")),
        _kpi_card_html("Total Defect Count 평균", _history_metric_text(quality["defect_count_mean"], "건")),
        _kpi_card_html("Uniformity 평균", _history_metric_text(quality["uniformity_mean"], "%")),
        _kpi_card_html("Wafer Count", f"{quality['wafer_count']}장"),
    ])
    _render_html(f'<div class="kpi-grid">{cards_html}</div>')


def render_history_comparison_table(entry: dict) -> None:
    """변경 전/후 실제 품질 비교표 — 오른쪽 카드 요약의 계산 근거를 사용자가 직접 검증할 수 있게
    전체 지표를 표로 보여준다."""
    comparison = entry.get("comparison")
    if not comparison:
        st.caption("이전 Revision 평가 데이터가 없어 비교표를 만들 수 없습니다.")
        return

    prev_label = f"{entry['prev_revision']} 평균"
    curr_label = f"{entry['revision']} 평균"

    def _fmt(value, unit):
        return "-" if value is None else f"{_no_negative_zero(value):.1f}{unit}"

    rows = []
    for row in comparison:
        unit = row["unit"]
        rows.append({
            "지표": row["label"],
            prev_label: _fmt(row["prev_value"], unit),
            curr_label: _fmt(row["curr_value"], unit),
            "Target": _fmt(row["target"], unit) if row["mode"] == "error" else "-",
            "이전 Target 오차": _fmt(row["prev_error"], unit) if row["mode"] == "error" else "-",
            "현재 Target 오차": _fmt(row["curr_error"], unit) if row["mode"] == "error" else "-",
            "변화량": (
                "-" if row["delta"] is None else
                f"{_no_negative_zero(row['delta']):+.1f}{'%p' if unit == '%' else unit}"
            ),
            "판정": row["classification"] or "-",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def show_parameter_history_results(recipe_df: pd.DataFrame, wafer_df: pd.DataFrame, process: str,
                                    equipment: str, param_type: str, detail_param: str | None = None) -> None:
    results = build_history_results(
        recipe_df, wafer_df, process, param_type, equipment, detail_param,
        targets=get_default_targets(wafer_df, recipe_df),
    )

    if results["revision_count"] == 0:
        st.markdown(
            '<div class="history-empty">선택한 조건과 일치하는 파라미터 변경 이력이 없습니다.</div>',
            unsafe_allow_html=True,
        )
        return

    title = f"{param_type} 변경 이력" if param_type and param_type != "전체" else "전체 파라미터 변경 이력"
    st.markdown(f"#### {title}")
    st.caption(
        f"총 {results['revision_count']}개 Revision · 평가 Wafer {results['total_wafer_count']}장 · "
        f"세부 파라미터 변경 {results['matched_param_count']}건"
    )

    summary_cols = st.columns(3)
    for col, (label, value) in zip(summary_cols, [
        ("검색된 Revision 수", f"{results['revision_count']}개"),
        ("평가 Wafer 수", f"{results['total_wafer_count']}장"),
        ("세부 파라미터 변경 건수", f"{results['matched_param_count']}건"),
    ]):
        with col:
            render_history_kpi_card(label, value)

    st.markdown("")
    for entry in reversed(results["revisions"]):  # 최신 Revision이 위로 오게
        render_history_revision_card(entry)


# ==============================================================================
# 메인 실행부
# ==============================================================================
def main():
    # 페이지 새로고침(F5)은 세션을 새로 시작시키지만, 로그인 직후 URL에 남겨둔
    # 플래그가 있으면 다시 로그인 화면을 보여주지 않고 바로 복원한다.
    if not st.session_state.authenticated and st.query_params.get("auth") == "1":
        st.session_state.authenticated = True

    if not st.session_state.authenticated:
        render_login_page()
        return

    create_process_selector()
    recent_quality_slot = create_sidebar()
    process = st.session_state.process
    stage_defs = PROCESS_STAGE_DEFS[process]

    tab1, tab2, tab3, tab4 = st.tabs([
        "공정 예측 · 평가 · 추천", "Process Dashboard", "Parameter 변경 이력 조회", "신규 공정 품질 예측",
    ])

    # ---- Tab 1: 시뮬레이터 (모드별로 분리 — 멘토 피드백) ----
    with tab1:
        workbook = get_active_workbook()
        mode, inputs, targets, submitted, stage_defs, recipe, _ = create_input_panel()

        if mode == "target":
            # 모드 1: 목표 품질 → 레시피 변경점 추천 (Output E 중심)
            def _run_target_recommendation():
                started_at = perf_counter()
                st.session_state.prediction_running = True
                st.session_state.last_run_error = None
                st.session_state.last_run_kind = "target"
                st.session_state.target_mode_combo = None
                with st.status("변경점 추천을 준비하는 중...", expanded=True) as run_status:
                    try:
                        run_status.write("선택한 Recipe와 목표 품질을 확인하고 있습니다.")
                        baseline_result = predict(
                            inputs,
                            workbook["Wafer_Summary"],
                            workbook["Recipe_Master"],
                            process=process,
                        )
                        run_status.write("모델 예측 결과를 바탕으로 파라미터 변경점을 탐색하고 있습니다.")
                        suggestion = (
                            recommend_parameter_adjustments(inputs, targets, process=process)
                            if not baseline_result.get("_error")
                            else {"error": baseline_result.get("_error")}
                        )
                        st.session_state.target_mode_baseline = baseline_result
                        st.session_state.target_mode_suggestion = suggestion
                        st.session_state.target_mode_targets = dict(targets)
                        if baseline_result.get("_error"):
                            st.session_state.last_run_error = str(baseline_result["_error"])
                            run_status.update(label="입력값 또는 모델 상태를 확인해 주세요.", state="error", expanded=False)
                        else:
                            run_status.update(label="변경점 추천이 완료됐습니다.", state="complete", expanded=False)
                    except Exception as exc:
                        st.session_state.target_mode_baseline = None
                        st.session_state.target_mode_suggestion = None
                        st.session_state.last_run_error = f"{type(exc).__name__}: {exc}"
                        run_status.update(label="변경점 추천 중 오류가 발생했습니다.", state="error", expanded=False)
                    finally:
                        st.session_state.last_run_duration = perf_counter() - started_at
                        st.session_state.prediction_running = False

            if submitted:
                _run_target_recommendation()

            if st.session_state.get("last_run_kind") == "target":
                if st.session_state.get("last_run_error"):
                    st.error(f"실행 실패 · {st.session_state.last_run_error}")
                    if st.button("↻ 변경점 추천 다시 시도", key="retry_target_run"):
                        _run_target_recommendation()
                elif st.session_state.get("last_run_duration") is not None:
                    st.caption(f"✅ 최근 변경점 추천 완료 · {st.session_state.last_run_duration:.1f}초")

            baseline_result = st.session_state.get("target_mode_baseline")
            if baseline_result is not None:
                # run_targets: 마지막 '변경점 추천 실행' 클릭 시점의 목표값 스냅샷.
                # 아래 진단/추천표/조합검증은 전부 이 스냅샷 기준으로 통일해서 보여준다 — 그래야
                # 목표값을 입력창에서 바꾸기만 하고 아직 재실행하지 않은 상태에서 일부 섹션은 새 목표로,
                # 일부는 이전 목표로 채점되는 불일치(멘토 피드백 8/12)가 생기지 않는다.
                run_targets = st.session_state.get("target_mode_targets", targets)
                if targets != run_targets:
                    st.warning(
                        "목표 품질 값이 마지막 추천 실행 이후 변경되었습니다. 아래 결과는 이전 목표값 "
                        "기준입니다 — 새 목표로 다시 채점하려면 '변경점 추천 실행'을 다시 눌러주세요."
                    )
                st.markdown("---")
                st.markdown(f"<div class='section-title'>'{recipe}' 기준 현재 예측 품질</div>", unsafe_allow_html=True)
                show_prediction(baseline_result)
                if not baseline_result.get("_error"):
                    st.markdown("---")
                    show_target_diagnosis(baseline_result, run_targets)
                    st.markdown("---")
                    suggestion = st.session_state.get("target_mode_suggestion")
                    show_parameter_recommendations(suggestion, run_targets, stage_defs)
                    st.markdown("---")
                    show_combined_recipe_section(
                        inputs,
                        baseline_result,
                        suggestion,
                        run_targets,
                        stage_defs,
                        process,
                        workbook,
                    )
            else:
                st.info("목표 품질을 확인하고 '변경점 추천 실행' 버튼을 눌러주세요.")

        else:
            # 모드 2: 레시피 조건 직접 입력 → 품질 평가 (Output A~D)
            def _run_prediction(allow_out_of_range: bool = False):
                started_at = perf_counter()
                st.session_state.prediction_running = True
                st.session_state.last_run_error = None
                st.session_state.last_run_kind = "evaluation"
                with st.status("예측 · 평가 · 추천을 실행하는 중...", expanded=True) as run_status:
                    try:
                        run_status.write("공정 입력값과 모델 범위를 확인하고 있습니다.")
                        result = predict(
                            inputs, workbook["Wafer_Summary"], workbook["Recipe_Master"],
                            process=process, allow_out_of_range=allow_out_of_range,
                        )
                        run_status.write("예측 결과를 목표 품질과 비교하고 있습니다.")
                        evaluation = evaluate_against_target(result, targets) if not result.get("_error") else None
                        recommendation = None
                        if not result.get("_error"):
                            run_status.write("현재 조건과 가장 적합한 Recipe를 비교하고 있습니다.")
                            recommendation = recommend_best_recipe(
                                inputs["recipe"], inputs, targets, workbook["Wafer_Summary"], workbook["Recipe_Master"], process=process
                            )

                        st.session_state.prediction_result = result
                        st.session_state.prediction_inputs = inputs
                        st.session_state.prediction_targets = targets
                        st.session_state.prediction_evaluation = evaluation
                        st.session_state.prediction_recommendation = recommendation
                        append_quality_history(
                            result,
                            evaluation,
                            recipe,
                            inputs,
                            workbook["Recipe_Master"],
                            stage_defs,
                        )

                        if result.get("_error"):
                            if not result.get("_out_of_range"):
                                st.session_state.last_run_error = str(result["_error"])
                            run_status.update(label="입력값 확인이 필요합니다.", state="error", expanded=False)
                        else:
                            run_status.update(label="예측 · 평가 · 추천이 완료됐습니다.", state="complete", expanded=False)
                    except Exception as exc:
                        st.session_state.prediction_result = None
                        st.session_state.prediction_evaluation = None
                        st.session_state.prediction_recommendation = None
                        st.session_state.last_run_error = f"{type(exc).__name__}: {exc}"
                        run_status.update(label="실행 중 오류가 발생했습니다.", state="error", expanded=False)
                    finally:
                        st.session_state.last_run_duration = perf_counter() - started_at
                        st.session_state.prediction_running = False

            if submitted:
                _run_prediction(allow_out_of_range=False)

            if st.session_state.get("last_run_kind") == "evaluation":
                if st.session_state.get("last_run_error"):
                    st.error(f"실행 실패 · {st.session_state.last_run_error}")
                    if st.button("↻ 예측 · 평가 · 추천 다시 시도", key="retry_evaluation_run"):
                        _run_prediction(allow_out_of_range=False)
                elif st.session_state.get("last_run_duration") is not None:
                    st.caption(f"✅ 최근 품질 평가 완료 · {st.session_state.last_run_duration:.1f}초")

            result = st.session_state.prediction_result
            if result is not None:
                if result.get("_out_of_range"):
                    st.markdown("---")
                    st.warning(f"⚠️ {result['_error']}")
                    st.caption("학습 데이터 밖의 값이라 예측 신뢰도가 떨어질 수 있습니다. 그래도 이 값으로 결과를 보고 싶다면 아래 버튼을 누르세요.")
                    if st.button("그래도 이 값으로 실행", key="force_out_of_range_run"):
                        _run_prediction(allow_out_of_range=True)
                        st.rerun()
                else:
                    st.markdown("---")
                    show_prediction(result)
                    if not result.get("_error"):
                        st.markdown("---")
                        show_target_evaluation(st.session_state.prediction_evaluation)
                        st.markdown("---")
                        show_recommendation(
                            st.session_state.prediction_inputs, st.session_state.prediction_recommendation,
                            workbook["Recipe_Master"], stage_defs,
                        )
                        st.markdown("---")
                        show_comparison(
                            st.session_state.prediction_result,
                            st.session_state.prediction_evaluation["score"],
                            st.session_state.prediction_recommendation,
                        )
            else:
                st.info("공정 조건과 목표 품질을 입력하고 '예측 · 평가 · 추천 실행' 버튼을 눌러주세요.")

    # ---- Tab 2: Process Dashboard ----
    # Equipment 선택(Chamber 자동) → Rev별 종합 품질 점수 + Best Case(항상 노출) → Rev 선택
    # → 상세 섹션(AI 분석/Summary/시각화/Wafer Map/Zone 분석)은 expander로 접어 초기 화면을 짧게 유지한다.
    with tab2:
        workbook = get_active_workbook()
        equipment, chambers, equipment_chamber_wafer = create_process_dashboard_equipment_selector(workbook, process)

        dashboard_targets = get_default_targets(workbook["Wafer_Summary"], workbook["Recipe_Master"])
        dashboard_targets.update(get_dashboard_spec_targets(process))
        scoreboard, selected_rev = show_recipe_scoreboard(
            equipment_chamber_wafer, dashboard_targets, workbook["Recipe_Master"], process,
        )

        filtered_wafer, filtered_site = filter_dashboard_by_rev(
            workbook, equipment_chamber_wafer, equipment, chambers, selected_rev,
        )

        with st.expander("AI 분석", expanded=False):
            zone_summary_preview = get_zone_summary(filtered_site) if not filtered_site.empty else None
            show_dashboard_ai_analysis(filtered_wafer, zone_summary_preview)

        with st.expander("Recipe 비교", expanded=False):
            show_recipe_comparison(scoreboard, selected_rev)

        with st.expander("선택 Recipe Summary", expanded=False):
            show_process_summary(filtered_wafer)

        with st.expander("Rev별 품질 변화", expanded=False):
            show_rev_quality_trends(
                equipment_chamber_wafer, scoreboard, workbook["Recipe_Master"], dashboard_targets, selected_rev,
            )

        with st.expander("Wafer 단위 품질 결과", expanded=False):
            show_quality_visualization(filtered_wafer, dashboard_targets)

        with st.expander("Wafer 단면 Profile", expanded=False):
            show_wafer_profile(filtered_site)

        with st.expander("Zone별 분석", expanded=False):
            show_zone_analysis(filtered_site)

    # ---- Tab 3: Parameter 변경 이력 조회 (Parameter Change History) ----
    with tab3:
        workbook = get_active_workbook()
        equipment, param_type, detail_param, submitted = create_parameter_history_selectors(workbook, process)
        if submitted:
            st.session_state[f"ph_searched_{process}"] = True

        st.markdown("---")
        if st.session_state.get(f"ph_searched_{process}"):
            show_parameter_history_results(
                workbook["Recipe_Master"], workbook["Wafer_Summary"], process, equipment, param_type, detail_param,
            )
        else:
            st.info("Equipment Model과 변경 파라미터를 선택한 뒤 '이력 조회' 버튼을 눌러주세요.")

    # ---- Tab 4: 신규 공정 프리뷰 (유사도 기반, 편법 — 정식 예측 아님) ----
    with tab4:
        show_new_process_preview()

    render_recent_quality_sidebar(recent_quality_slot)


if __name__ == "__main__":
    main()
