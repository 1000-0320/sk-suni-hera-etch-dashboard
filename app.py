"""
Etch AI Decision Support System
반도체 Etch 공정 품질 예측 및 의사결정 지원 시스템

isolation / trench 두 공정을 선택할 수 있고, 학습된 RandomForest/XGBoost 모델로
실제 예측을 수행한다 (model.predict() 내부가 ml_engine의 학습된 모델을 호출).

업로드 워크북은 3개 시트 구조를 전제로 한다: Recipe_Master / Wafer_Summary / Site_Level_Raw

실행: streamlit run app.py
"""

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
    ordered_recipe_versions, get_default_targets, PROCESS_STAGE_DEFS, PROCESS_LABELS,
    REQUIRED_SHEETS, PARTICLE_DEFECT_THRESHOLD,
)
from style import (
    inject_custom_css, render_metric_card, render_summary_card, render_status_badge,
    render_score_hero, render_subscore_card, render_pill_card,
    get_pass_rate_status, get_cd_uniformity_status, get_depth_uniformity_status, get_particle_status,
    get_score_status, STATUS_META,
)
from charts import (
    build_cd_bar_chart, build_gauge_chart, build_variation_gauge,
    build_cd_trend_chart, build_depth_trend_chart,
    build_cd_uniformity_trend_chart, build_depth_uniformity_trend_chart,
    build_pass_rate_trend_chart, build_particle_chart,
    build_rev_cd_trend_chart, build_rev_depth_trend_chart,
    build_rev_uniformity_trend_chart, build_rev_pass_rate_chart, build_rev_defect_chart,
    build_wafer_map, build_wafer_profile_chart, build_recipe_score_chart,
    build_zone_cd_chart, build_zone_depth_chart, build_zone_spread_chart,
    build_zone_pass_rate_chart, build_zone_defect_chart,
    build_cd_comparison_chart, build_score_comparison_chart,
)
from login_page import render_login_page, render_sidebar_logout

st.set_page_config(page_title="Etch AI Decision Support System", page_icon="🧪", layout="wide")
inject_custom_css()

# Wafer Map에서 한 번에 보여줄 지표들 (표시용 라벨 -> Site_Level_Raw 실제 컬럼명)
WAFER_MAP_METRICS = {
    "Top CD": "Top_CD_nm", "Mid CD": "Mid_CD_nm", "Bottom CD": "Bottom_CD_nm", "Depth": "Depth_nm",
}
TARGET_MODE_LABEL = "목표 품질 → 레시피 변경점 추천"
DIRECT_MODE_LABEL = "레시피 조건 직접 입력 → 품질 평가"


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
# 0. 공정 선택 (isolation / trench) — 화면 전체에 영향
# ==============================================================================
def create_process_selector():
    options = list(PROCESS_STAGE_DEFS.keys())
    short_labels = {"isolation": "Isolation", "trench": "Trench"}

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

    selector_col, _ = st.columns([1.55, 3.45], gap="large")
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
        parameters.extend(
            (gas_col.lower(), f"{gas_col.split('_')[1]} Flow", "sccm")
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

    # ---- 공정 기본 정보 ----
    st.markdown("<div class='section-title'>공정 기본 정보</div>", unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Process Type", PROCESS_LABELS[process])
    a2.metric("공정 Step 수", f"{len(stage_defs)}-Step")
    a3.metric("Step1 Layer", stage_defs[0]["label"])
    a4.metric("마지막 Step Layer", stage_defs[-1]["label"])

    # ---- 장비 선택 (Chamber는 자동 대표값 사용 — 멘토 피드백: 모든 Chamber 동일 조건으로 가정) ----
    st.markdown("<div class='section-title'>장비 선택</div>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        equipment = st.selectbox("Equipment Model", sorted(wafer_df["Equipment_Model"].unique()), key=f"eq_{process}")
    chamber_choices = sorted(
        wafer_df.loc[wafer_df["Equipment_Model"] == equipment, "Chamber_ID"].unique()
    ) or sorted(wafer_df["Chamber_ID"].unique())
    chamber = chamber_choices[0]
    recipe_options = ordered_recipe_versions(recipe_df)
    with b2:
        recipe = st.selectbox("Recipe (참고용 베이스라인)", recipe_options, key=f"recipe_{process}")
    st.caption(f"Chamber는 모든 Chamber가 동일하다는 가정으로 대표값(`{chamber}`)을 자동 사용합니다.")

    stage_table = get_recipe_stage_table(recipe_df, recipe, stage_defs)
    if not stage_table.empty:
        st.markdown(f"**Recipe '{recipe}' 실제 Stage 조건 (참고용)**")
        st.dataframe(stage_table, use_container_width=True, hide_index=True)

    # ---- 목표 품질 설정 (참고값 → 목표값 형식 — 멘토 피드백) ----
    st.markdown("<div class='section-title'>목표 품질 설정</div>", unsafe_allow_html=True)
    st.caption(f"'{recipe}' Recipe의 실측 평균이 참고값입니다. 라벨에 표시된 참고값 → 원하는 목표값만 조정하세요.")
    defaults = get_default_targets(wafer_df, recipe_df, recipe)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_top_cd = st.number_input(
            f"Top CD (nm): {defaults['target_top_cd']:g} → 목표", value=defaults["target_top_cd"], key=f"ttop_{process}_{recipe}",
        )
    with c2:
        target_mid_cd = st.number_input(
            f"Mid CD (nm): {defaults['target_mid_cd']:g} → 목표", value=defaults["target_mid_cd"], key=f"tmid_{process}_{recipe}",
        )
    with c3:
        target_bottom_cd = st.number_input(
            f"Bottom CD (nm): {defaults['target_bottom_cd']:g} → 목표", value=defaults["target_bottom_cd"], key=f"tbot_{process}_{recipe}",
        )
    with c4:
        target_depth = st.number_input(
            f"Depth (nm): {defaults['target_depth']:g} → 목표", value=defaults["target_depth"], key=f"tdep_{process}_{recipe}",
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        max_cd_uniformity = st.number_input("Maximum CD Uniformity (%)", value=defaults["max_cd_uniformity"], step=0.5, key=f"mcdu_{process}")
    with c6:
        max_depth_uniformity = st.number_input("Maximum Depth Uniformity (%)", value=defaults["max_depth_uniformity"], step=0.5, key=f"mdu_{process}")
    with c7:
        min_pass_rate = st.number_input("Minimum Pass Rate (%)", value=defaults["min_pass_rate"], step=1.0, key=f"mpr_{process}")
    with c8:
        max_defect_count = st.number_input("Maximum Defect Count (건)", value=defaults["max_defect_count"], step=1.0, key=f"mdc_{process}")

    targets = {
        "target_top_cd": target_top_cd, "target_mid_cd": target_mid_cd,
        "target_bottom_cd": target_bottom_cd, "target_depth": target_depth,
        "max_cd_uniformity": max_cd_uniformity, "max_depth_uniformity": max_depth_uniformity,
        "min_pass_rate": min_pass_rate, "max_defect_count": max_defect_count,
    }

    stage_defaults = stage_inputs_from_recipe(recipe_df, recipe, stage_defs)
    baseline_inputs = {"equipment": equipment, "chamber": chamber, "recipe": recipe, **stage_defaults}

    # ---- 시뮬레이터 모드 (멘토 피드백: 기능 두 가지를 명확히 분리) ----
    st.markdown("<div class='section-title'>시뮬레이터 모드</div>", unsafe_allow_html=True)
    st.markdown('<span class="simulator-mode-anchor"></span>', unsafe_allow_html=True)
    mode_label = st.radio(
        "무엇을 하고 싶으신가요?",
        [TARGET_MODE_LABEL, DIRECT_MODE_LABEL],
        horizontal=True, key=f"mode_{process}",
    )

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
            gas_label = gas_col_name.split("_")[1]
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
    """Process Dashboard의 세 필터를 해당 공정 기본값으로 되돌린다."""
    for key in (f"pd_equipment_{process}", f"pd_chamber_{process}", f"pd_recipe_{process}"):
        st.session_state.pop(key, None)


def create_process_dashboard_selectors(workbook: dict, process: str):
    st.markdown("### Process Dashboard 조건 선택")
    wafer_df = workbook["Wafer_Summary"]
    recipe_df = workbook["Recipe_Master"]

    col1, col2, col3 = st.columns(3)
    with col1:
        equipment = st.selectbox("Equipment", sorted(wafer_df["Equipment_Model"].unique()), key=f"pd_equipment_{process}")
    df1 = wafer_df[wafer_df["Equipment_Model"] == equipment]

    with col2:
        chamber_choices = sorted(df1["Chamber_ID"].unique()) or sorted(wafer_df["Chamber_ID"].unique())
        chamber = st.selectbox("Chamber", chamber_choices, key=f"pd_chamber_{process}")
    df2 = df1[df1["Chamber_ID"] == chamber]

    with col3:
        recipe_choices = ordered_recipe_versions(recipe_df, df2["Recipe_Version"].unique()) or ordered_recipe_versions(recipe_df)
        recipe = st.selectbox("Recipe", recipe_choices, key=f"pd_recipe_{process}")

    filtered_wafer = df2[df2["Recipe_Version"] == recipe]

    site_df = workbook["Site_Level_Raw"]
    filtered_site = site_df[
        (site_df["Equipment_Model"] == equipment)
        & (site_df["Chamber_ID"] == chamber)
        & (site_df["Recipe_Version"] == recipe)
    ]

    context_col, reset_col = st.columns([5, 1], gap="small")
    with context_col:
        st.markdown(
            f"""
            <div class="dashboard-context-strip">
                <span>ACTIVE VIEW</span>
                <strong>{escape(PROCESS_LABELS[process])} · {escape(str(equipment))} · {escape(str(chamber))} · {escape(str(recipe))}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with reset_col:
        st.button(
            "↺ 필터 초기화",
            key=f"reset_pd_filters_{process}",
            on_click=reset_process_dashboard_filters,
            args=(process,),
            use_container_width=True,
        )

    return filtered_wafer, filtered_site, df2


def show_process_summary(filtered_wafer: pd.DataFrame):
    render_dashboard_section_title("Process Summary", "summary")
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

    row1 = st.columns(5)
    row1_items = [
        ("Equipment", filtered_wafer["Equipment_Model"].iloc[0]),
        ("Chamber", filtered_wafer["Chamber_ID"].iloc[0]),
        ("Recipe", filtered_wafer["Recipe_Version"].iloc[0]),
        ("Total Wafer 수", f"{total_wafers}"),
        ("평균 Pass Rate", f"{avg_pass_rate:.1f}%"),
    ]
    for col, (label, value) in zip(row1, row1_items):
        with col:
            render_summary_card(label, value)

    row2 = st.columns(4)
    row2_items = [
        ("평균 CD", f"{avg_cd:.1f} nm"),
        ("평균 Depth", f"{avg_depth:.1f} nm"),
        ("평균 Uniformity(CV%)", f"{avg_uniformity:.2f}%"),
        (f"Particle 발생 건수 (Defect>{PARTICLE_DEFECT_THRESHOLD})", f"{particle_count} 건"),
    ]
    for col, (label, value) in zip(row2, row2_items):
        with col:
            render_summary_card(label, value)


# ==============================================================================
# 2-3. Rev별 스코어링 순위 (멘토 피드백: 현재 어떤 레시피가 가장 좋은지 바로 알 수 있게)
# ==============================================================================
def show_recipe_scoreboard(equipment_chamber_wafer: pd.DataFrame, targets: dict):
    render_dashboard_section_title("Rev별 스코어링 순위", "score", "blue")
    st.caption("모델 예측이 아니라 실제 측정된 Wafer 결과를 Recipe(Rev)별로 집계한 종합 품질 점수입니다.")
    if equipment_chamber_wafer.empty:
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")
        return pd.DataFrame()

    scoreboard = score_recipe_versions(equipment_chamber_wafer, targets)
    if scoreboard.empty:
        st.info("Rev 스코어를 계산할 데이터가 없습니다.")
        return scoreboard

    table_cols = ["Recipe", "종합 점수", "Wafer 수", "Top CD", "Mid CD", "Bottom CD", "Depth", "Overall Spec Pass Rate"]
    st.dataframe(scoreboard[table_cols].round(1), use_container_width=True, hide_index=True)
    st.plotly_chart(build_recipe_score_chart(scoreboard), use_container_width=True)
    return scoreboard


def show_rev_quality_trends(
    equipment_chamber_wafer: pd.DataFrame,
    scoreboard: pd.DataFrame,
    recipe_df: pd.DataFrame,
):
    """현재 Equipment/Chamber의 전체 Recipe를 Rev 순서로 집계해 비교한다."""
    render_dashboard_section_title("Rev별 품질 변화", "quality")
    st.caption(
        "선택한 Equipment/Chamber의 실제 Wafer 결과를 Rev별로 집계했습니다. "
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
        st.plotly_chart(build_rev_cd_trend_chart(rev_scoreboard), use_container_width=True)
    with row1[1]:
        st.plotly_chart(build_rev_depth_trend_chart(rev_scoreboard), use_container_width=True)

    row2 = st.columns(2, gap="large")
    with row2[0]:
        st.plotly_chart(build_rev_uniformity_trend_chart(rev_scoreboard), use_container_width=True)
    with row2[1]:
        st.plotly_chart(build_rev_pass_rate_chart(rev_scoreboard), use_container_width=True)

    st.plotly_chart(build_rev_defect_chart(rev_particle_df), use_container_width=True)


# ==============================================================================
# 3. 품질 결과 시각화 (Wafer 단위 추이) — 각 그래프는 독립 카드
# ==============================================================================
def show_quality_visualization(filtered_wafer: pd.DataFrame):
    render_dashboard_section_title("품질 결과 시각화", "quality")
    if filtered_wafer.empty:
        return

    wafer_df = filtered_wafer.copy()
    if "Eval_Timestamp" in wafer_df.columns:
        wafer_df = wafer_df.sort_values("Eval_Timestamp")
    wafer_df["Wafer_Label"] = wafer_df["Lot_ID"].astype(str) + " / " + wafer_df["Wafer_ID"].astype(str)

    row1 = st.columns(2)
    with row1[0]:
        st.plotly_chart(build_cd_trend_chart(wafer_df), use_container_width=True)
    with row1[1]:
        st.plotly_chart(build_depth_trend_chart(wafer_df), use_container_width=True)

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
# 4. Wafer Map — Top/Mid/Bottom CD와 Depth를 한 번에 표시 (멘토 피드백 반영)
# ==============================================================================
def show_wafer_map(filtered_site: pd.DataFrame):
    render_dashboard_section_title("Wafer Map (Top/Mid/Bottom CD · Depth 한눈에 비교)", "map", "blue")
    if filtered_site.empty:
        st.info("선택한 조건에 해당하는 Site 데이터가 없습니다.")
        return

    cols = st.columns(4)
    for col, (label, metric_col) in zip(cols, WAFER_MAP_METRICS.items()):
        with col:
            st.plotly_chart(build_wafer_map(filtered_site, metric_col, label), use_container_width=True)
    st.caption("Zone: Center(중심) → Mid → Edge → Extreme Edge(바깥쪽) · 선택 조건에 해당하는 모든 Wafer의 같은 Site 위치를 평균해 표시")

    st.markdown("**Wafer 단면 Profile**")
    st.caption("x축 = Point 번호. Edge → Center → Edge 순서라 위 2D Wafer Map보다 정확한 수치 비교가 쉽습니다.")
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
    render_dashboard_section_title("Zone 분석", "zone")
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


def show_new_process_preview():
    """신규 공정 품질 프리뷰 (레이어 조합 기반, 편법) — 정식 AI 예측이 아니라
    입력한 레이어(물질) 순서를 기존 4개 공정의 Stage들과 대조해 참고값을 보여준다."""
    st.markdown("<div class='section-title'>신규 공정 품질 프리뷰</div>", unsafe_allow_html=True)
    st.caption(
        "아직 AI 모델이 없는 새 공정을, **Etch 하려는 레이어(물질)를 순서대로** 입력해서 미리 감을 잡는 "
        "기능입니다. 한 줄에 레이어 하나씩 입력하세요 — 예: 하드마스크로 SiO2를 먼저 Etch하고, "
        "그다음 PolySi를 Etch하는 공정이면 첫 줄 SiO2, 둘째 줄 PolySi. "
        "**AI 예측이 아니며 점수화도 하지 않습니다.**"
    )
    st.info("사용 가능한 물질: SiO2, SiON, SOC, Si, PolySi, Al")

    layer_text = st.text_area(
        "신규 공정의 레이어를 위에서부터 순서대로, 한 줄에 하나씩 입력하세요",
        placeholder="SiO2\nPolySi",
        height=100,
        key="new_process_layer_text",
    )
    if not layer_text.strip():
        return
    layer_materials = [line for line in layer_text.splitlines() if line.strip()]

    result = find_layer_references(layer_materials)
    layer_matches = result["layer_matches"]
    overall = result["overall_reference"]

    st.markdown(f"### 1) 레이어별 참고 Recipe 조건 ({len(layer_matches)}개 레이어)")
    for i, layer in enumerate(layer_matches, start=1):
        with st.expander(f"레이어 {i}: {layer['material']}", expanded=True):
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

    st.markdown("### 2) 종합 참고 품질")
    if overall is None:
        st.warning("입력한 레이어 구성과 겹치는 기존 공정을 찾지 못해 종합 참고 품질을 계산할 수 없습니다.")
        return

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


# ==============================================================================
# 메인 실행부
# ==============================================================================
def main():
    if not st.session_state.authenticated:
        render_login_page()
        return

    create_process_selector()
    recent_quality_slot = create_sidebar()
    process = st.session_state.process
    stage_defs = PROCESS_STAGE_DEFS[process]

    tab1, tab2, tab3 = st.tabs(["공정 예측 · 평가 · 추천", "Process Dashboard", "신규 공정 프리뷰"])

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
                st.markdown("---")
                st.markdown(f"<div class='section-title'>'{recipe}' 기준 현재 예측 품질</div>", unsafe_allow_html=True)
                show_prediction(baseline_result)
                if not baseline_result.get("_error"):
                    st.markdown("---")
                    show_target_diagnosis(baseline_result, targets)
                    st.markdown("---")
                    suggestion = st.session_state.get("target_mode_suggestion")
                    show_parameter_recommendations(suggestion, targets, stage_defs)
                    st.markdown("---")
                    show_combined_recipe_section(
                        inputs,
                        baseline_result,
                        suggestion,
                        targets,
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

    # ---- Tab 2: Process Dashboard (조건 선택 → AI 분석(맨 위) → Summary → 시각화 → Wafer Map → Zone 분석) ----
    with tab2:
        workbook = get_active_workbook()
        filtered_wafer, filtered_site, equipment_chamber_wafer = create_process_dashboard_selectors(workbook, process)

        st.markdown("---")
        zone_summary_preview = get_zone_summary(filtered_site) if not filtered_site.empty else None
        show_dashboard_ai_analysis(filtered_wafer, zone_summary_preview)
        st.markdown("---")
        show_process_summary(filtered_wafer)
        dashboard_targets = get_default_targets(workbook["Wafer_Summary"], workbook["Recipe_Master"])
        scoreboard = show_recipe_scoreboard(equipment_chamber_wafer, dashboard_targets)
        st.markdown("---")
        show_rev_quality_trends(equipment_chamber_wafer, scoreboard, workbook["Recipe_Master"])
        st.markdown("---")
        show_quality_visualization(filtered_wafer)
        show_wafer_map(filtered_site)
        show_zone_analysis(filtered_site)

    # ---- Tab 3: 신규 공정 프리뷰 (유사도 기반, 편법 — 정식 예측 아님) ----
    with tab3:
        show_new_process_preview()

    render_recent_quality_sidebar(recent_quality_slot)


if __name__ == "__main__":
    main()
