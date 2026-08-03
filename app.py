"""
Etch AI Decision Support System
반도체 Etch 공정 품질 예측 및 의사결정 지원 시스템

※ 아직 머신러닝(Random Forest / XGBoost) 미구현 단계.
   predict()는 업로드된 실측 이력(Wafer_Summary)에서 선택한 Recipe의 평균값을
   조회하는 "실데이터 기반 더미" 함수이며, 나중에 model.predict() 내부만
   교체하면 실제 학습 모델과 쉽게 연결할 수 있는 구조로 작성했다.

   업로드 워크북은 3개 시트 구조를 전제로 한다: Recipe_Master / Wafer_Summary / Site_Level_Raw

실행: streamlit run app.py
"""

import pandas as pd
import streamlit as st

from model import (
    predict, compute_composite_score, evaluate_against_target,
    recommend_best_recipe, build_stage_diff_table, generate_dashboard_analysis,
)
from data_utils import (
    generate_dummy_workbook, is_valid_workbook, load_required_sheets,
    get_zone_summary, get_recipe_stage_table, stage_inputs_from_recipe,
    ordered_recipe_versions, get_default_targets, STAGE_DEFS, REQUIRED_SHEETS, PARTICLE_DEFECT_THRESHOLD,
)
from style import (
    inject_custom_css, render_metric_card, render_summary_card, render_status_badge,
    render_score_hero, render_subscore_card, render_pill_card,
    get_pass_rate_status, get_cd_uniformity_status, get_depth_uniformity_status, get_particle_status,
)
from charts import (
    build_cd_bar_chart, build_gauge_chart, build_variation_gauge,
    build_cd_trend_chart, build_depth_trend_chart,
    build_cd_uniformity_trend_chart, build_depth_uniformity_trend_chart,
    build_pass_rate_trend_chart, build_particle_chart,
    build_wafer_map,
    build_zone_cd_chart, build_zone_depth_chart, build_zone_spread_chart,
    build_zone_pass_rate_chart, build_zone_defect_chart,
    build_cd_comparison_chart, build_score_comparison_chart,
)

st.set_page_config(page_title="Etch AI Decision Support System", page_icon="🧪", layout="wide")
inject_custom_css()

# Wafer Map 지표 선택지 (표시용 라벨 -> Site_Level_Raw 실제 컬럼명)
WAFER_MAP_METRICS = {
    "Top CD": "Top_CD_nm", "Mid CD": "Mid_CD_nm", "Bottom CD": "Bottom_CD_nm", "Depth": "Depth_nm",
}


# ==============================================================================
# 세션 상태 초기화
# ==============================================================================
def init_session_state():
    defaults = {
        "uploaded_filename": None,
        "sheet_names": [],
        "workbook": None,  # {"Recipe_Master":df, "Wafer_Summary":df, "Site_Level_Raw":df}
        "prediction_result": None,
        "prediction_inputs": None,
        "prediction_targets": None,
        "prediction_evaluation": None,
        "prediction_recommendation": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def get_active_workbook() -> dict:
    """Process Dashboard/예측에 사용할 워크북을 반환.
    업로드된 워크북이 유효(3-시트 구조 일치)하면 그것을, 아니면 더미 워크북을 사용한다."""
    if is_valid_workbook(st.session_state.workbook):
        return st.session_state.workbook
    return generate_dummy_workbook()


def using_real_data() -> bool:
    return is_valid_workbook(st.session_state.workbook)


# ==============================================================================
# 1. 사이드바 — Excel 업로드 / 시트 목록 / 데이터 불러오기
# ==============================================================================
def create_sidebar():
    with st.sidebar:
        st.markdown("## 📁 데이터 업로드")
        uploaded_file = st.file_uploader("Excel 파일 업로드 (.xlsx)", type=["xlsx"])

        if uploaded_file is not None:
            st.session_state.uploaded_filename = uploaded_file.name
            st.success(f"업로드됨: {uploaded_file.name}")

            try:
                excel_file = pd.ExcelFile(uploaded_file)
                st.session_state.sheet_names = excel_file.sheet_names

                st.markdown("**시트 목록**")
                for sheet in st.session_state.sheet_names:
                    mark = "✅" if sheet in REQUIRED_SHEETS else "•"
                    st.markdown(f"- {mark} {sheet}")

                if st.button("📥 데이터 불러오기", use_container_width=True):
                    sheets = load_required_sheets(excel_file)
                    if is_valid_workbook(sheets):
                        st.session_state.workbook = sheets
                        n_wafer = len(sheets["Wafer_Summary"])
                        n_site = len(sheets["Site_Level_Raw"])
                        st.success(f"로드 완료: Wafer {n_wafer}장 / Site {n_site}행")
                    else:
                        missing = [s for s in REQUIRED_SHEETS if s not in sheets]
                        st.error(
                            f"필요한 시트({', '.join(REQUIRED_SHEETS)}) 또는 컬럼이 맞지 않습니다. "
                            f"누락된 시트: {missing if missing else '없음(컬럼명 확인 필요)'}"
                        )
            except Exception as e:
                st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        else:
            st.info("Excel 파일을 업로드하지 않으면 더미 데이터로 데모가 진행됩니다.")

        st.markdown("---")
        if using_real_data():
            st.success("✅ 실제 업로드 데이터 사용 중")
        else:
            st.warning("⚠️ 더미 데이터 사용 중 (파일을 업로드하고 '데이터 불러오기'를 눌러주세요)")
        st.caption("필요 시트: Recipe_Master / Wafer_Summary / Site_Level_Raw")
        st.caption("Etch AI Decision Support System v0.3 · predict()는 실측 이력 기반 더미 로직")


# ==============================================================================
# 2. 공정 조건 입력 패널 (A. 기본정보 / B. 장비 / C. 목표 품질 / D. 현재 Recipe)
# ==============================================================================
def create_input_panel():
    workbook = get_active_workbook()
    wafer_df = workbook["Wafer_Summary"]
    recipe_df = workbook["Recipe_Master"]

    # ---- 공정 기본 정보 (현재는 데이터가 STI/Trench Etch 하나뿐이라 고정 표시) ----
    st.markdown("<div class='section-title'>공정 기본 정보</div>", unsafe_allow_html=True)
    st.caption("현재 데이터셋은 아래 1종류뿐이라 고정 표시됩니다. Recipe/Layer 종류가 늘어나면 Dropdown으로 전환됩니다.")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Process Type", "Trench Etch")
    a2.metric("공정 Step 수", f"{len(STAGE_DEFS)}-Step")
    a3.metric("Step1~3 Layer", "SiON/SOC/SiO2")
    a4.metric("Step4 Layer", "PolySi(Si Main)")

    # ---- 장비 및 Chamber 선택 (Chamber는 전체 목록에서 자유롭게 선택 가능) ----
    st.markdown("<div class='section-title'>장비 및 Chamber 선택</div>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        equipment = st.selectbox("Equipment Model", sorted(wafer_df["Equipment_Model"].unique()))
    with b2:
        chamber = st.selectbox("Chamber ID", sorted(wafer_df["Chamber_ID"].unique()))
    recipe_options = ordered_recipe_versions(recipe_df)
    with b3:
        recipe = st.selectbox("Recipe (참고용 베이스라인)", recipe_options)

    stage_table = get_recipe_stage_table(recipe_df, recipe)
    if not stage_table.empty:
        with st.expander(f"📋 Recipe '{recipe}' 실제 Stage 조건 (참고용)"):
            st.dataframe(stage_table, use_container_width=True, hide_index=True)

    # ---- 목표 품질 설정 (AI가 평가·추천할 때 사용할 기준값, 매번 사용자 입력) ----
    st.markdown("<div class='section-title'>목표 품질 설정</div>", unsafe_allow_html=True)
    st.caption("데이터셋에는 없는 값으로, 평가·추천의 기준이 됩니다. 기본값은 가장 성숙한 Recipe의 실측 평균입니다.")
    defaults = get_default_targets(wafer_df, recipe_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_top_cd = st.number_input("Target Top CD (nm)", value=defaults["target_top_cd"])
    with c2:
        target_mid_cd = st.number_input("Target Mid CD (nm)", value=defaults["target_mid_cd"])
    with c3:
        target_bottom_cd = st.number_input("Target Bottom CD (nm)", value=defaults["target_bottom_cd"])
    with c4:
        target_depth = st.number_input("Target Depth (nm)", value=defaults["target_depth"])

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        max_cd_uniformity = st.number_input("Maximum CD Uniformity (%)", value=defaults["max_cd_uniformity"], step=0.5)
    with c6:
        max_depth_uniformity = st.number_input("Maximum Depth Uniformity (%)", value=defaults["max_depth_uniformity"], step=0.5)
    with c7:
        min_pass_rate = st.number_input("Minimum Pass Rate (%)", value=defaults["min_pass_rate"], step=1.0)
    with c8:
        max_defect_count = st.number_input("Maximum Defect Count (건)", value=defaults["max_defect_count"], step=1.0)

    targets = {
        "target_top_cd": target_top_cd, "target_mid_cd": target_mid_cd,
        "target_bottom_cd": target_bottom_cd, "target_depth": target_depth,
        "max_cd_uniformity": max_cd_uniformity, "max_depth_uniformity": max_depth_uniformity,
        "min_pass_rate": min_pass_rate, "max_defect_count": max_defect_count,
    }

    # ---- 현재 공정 조건 입력 (Stage별 실제 조건, Recipe 선택 시 그 Recipe의 실측값으로 초기화) ----
    st.markdown("<div class='section-title'>현재 공정 조건 입력</div>", unsafe_allow_html=True)
    st.caption("Stage(S1~S4)별 Time/RF Bias/Pressure는 선택한 Recipe 대비 What-if 예측에 반영됩니다. Gas Flow는 입력창만 우선 제공됩니다.")

    stage_defaults = stage_inputs_from_recipe(recipe_df, recipe)
    inputs = {"equipment": equipment, "chamber": chamber, "recipe": recipe}
    match = recipe_df[recipe_df["Recipe_Version"] == recipe]
    recipe_row = match.iloc[0] if not match.empty else None

    for stage in STAGE_DEFS:
        key = stage["key"].lower()
        with st.expander(f"{stage['label']}", expanded=False):
            d1, d2, d3 = st.columns(3)
            with d1:
                inputs[f"{key}_time"] = st.number_input(
                    "Etch Time (s)", value=stage_defaults.get(f"{key}_time", 0.0),
                    key=f"{key}_time_{recipe}",
                )
            with d2:
                inputs[f"{key}_rf_bias"] = st.number_input(
                    "RF Bias (W)", value=stage_defaults.get(f"{key}_rf_bias", 0.0),
                    key=f"{key}_rf_bias_{recipe}",
                )
            with d3:
                inputs[f"{key}_pressure"] = st.number_input(
                    "Pressure (mT)", value=stage_defaults.get(f"{key}_pressure", 0.0),
                    key=f"{key}_pressure_{recipe}",
                )

            # Gas Flow — 멘토 가이드(O2 Flow가 핵심 영향 변수)를 반영해 입력창으로 노출.
            # 아직 predict()의 What-if 보정식에는 연결하지 않았고(추후 반영 예정), 입력/저장만 우선 지원한다.
            gas_cols_ui = st.columns(len(stage["gas_cols"]))
            for gas_col_widget, gas_col_name in zip(gas_cols_ui, stage["gas_cols"]):
                gas_label = gas_col_name.split("_")[1]
                default_val = float(recipe_row[gas_col_name]) if recipe_row is not None and gas_col_name in recipe_row.index else 0.0
                input_key = gas_col_name.lower()
                with gas_col_widget:
                    inputs[input_key] = st.number_input(
                        f"{gas_label} Flow (sccm)", value=default_val,
                        key=f"{input_key}_{recipe}",
                    )
            st.caption("Gas Flow는 입력만 반영된 상태이며, 예측 보정 로직 연결은 추후 진행됩니다.")

    submitted = st.button("🔮 예측 · 평가 · 추천 실행", type="primary", use_container_width=True)

    return inputs, targets, submitted


# ==============================================================================
# Output A. 현재 Recipe 품질 예측
# ==============================================================================
def show_prediction(result: dict):
    st.markdown("<div class='section-title'>현재 Recipe 품질 예측</div>", unsafe_allow_html=True)
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
    st.caption("※ Uniformity는 변동계수(CV%) 기준으로 값이 낮을수록 좋습니다. Particle 발생 확률은 과거 이력 중 Defect>3건 Wafer 비율입니다.")

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
def show_recommendation(current_inputs: dict, recommendation: dict, recipe_df: pd.DataFrame):
    st.markdown("<div class='section-title'>최적 Recipe 추천</div>", unsafe_allow_html=True)

    if recommendation is None:
        st.info("추천할 다른 Recipe 이력이 없습니다.")
        return

    st.markdown(f"현재 조건(Equipment/Chamber)에서 알려진 Recipe 중 **{recommendation['recipe']}**의 종합 품질 점수가 가장 높게 예상됩니다.")

    diff_rows = build_stage_diff_table(current_inputs, recommendation["inputs"], recipe_df)
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
# 2-1 / 2-2. Process Dashboard — 조건 선택 + Summary
# ==============================================================================
def create_process_dashboard_selectors(workbook: dict):
    st.markdown("### 🧭 Process Dashboard 조건 선택")
    wafer_df = workbook["Wafer_Summary"]
    recipe_df = workbook["Recipe_Master"]

    col1, col2, col3 = st.columns(3)
    with col1:
        equipment = st.selectbox("Equipment", sorted(wafer_df["Equipment_Model"].unique()), key="pd_equipment")
    df1 = wafer_df[wafer_df["Equipment_Model"] == equipment]

    with col2:
        chamber_choices = sorted(df1["Chamber_ID"].unique()) or sorted(wafer_df["Chamber_ID"].unique())
        chamber = st.selectbox("Chamber", chamber_choices, key="pd_chamber")
    df2 = df1[df1["Chamber_ID"] == chamber]

    with col3:
        recipe_choices = ordered_recipe_versions(recipe_df, df2["Recipe_Version"].unique()) or ordered_recipe_versions(recipe_df)
        recipe = st.selectbox("Recipe", recipe_choices, key="pd_recipe")

    filtered_wafer = df2[df2["Recipe_Version"] == recipe]

    site_df = workbook["Site_Level_Raw"]
    filtered_site = site_df[
        (site_df["Equipment_Model"] == equipment)
        & (site_df["Chamber_ID"] == chamber)
        & (site_df["Recipe_Version"] == recipe)
    ]

    return filtered_wafer, filtered_site


def show_process_summary(filtered_wafer: pd.DataFrame):
    st.markdown("### 🗂️ Process Summary")
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
# 3. 품질 결과 시각화 (Wafer 단위 추이) — 각 그래프는 독립 카드
# ==============================================================================
def show_quality_visualization(filtered_wafer: pd.DataFrame):
    st.markdown("### 📊 품질 결과 시각화")
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
# 4. Wafer Map
# ==============================================================================
def show_wafer_map(filtered_site: pd.DataFrame):
    st.markdown("### 🎯 Wafer Map")
    if filtered_site.empty:
        st.info("선택한 조건에 해당하는 Site 데이터가 없습니다.")
        return

    label = st.selectbox("표시할 지표 선택", list(WAFER_MAP_METRICS.keys()), key="wafer_map_metric")
    st.plotly_chart(build_wafer_map(filtered_site, WAFER_MAP_METRICS[label], label), use_container_width=True)
    st.caption("Zone: Center(중심) → Mid → Edge → Extreme Edge(바깥쪽) · 선택 조건에 해당하는 모든 Wafer의 같은 Site 위치를 평균해 표시")


# ==============================================================================
# 5. Zone 분석
# ==============================================================================
def show_zone_analysis(filtered_site: pd.DataFrame):
    st.markdown("### 🧩 Zone 분석")
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
# 6. AI 분석 (Process Dashboard, Rule Base)
# ==============================================================================
def show_dashboard_ai_analysis(filtered_wafer: pd.DataFrame, zone_summary: pd.DataFrame):
    st.markdown("### 🤖 AI 분석")
    messages = generate_dashboard_analysis(filtered_wafer, zone_summary)
    html = "<div class='analysis-card'>" + "".join(f"<div>• {m}</div>" for m in messages) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ==============================================================================
# 메인 실행부
# ==============================================================================
def main():
    create_sidebar()

    st.markdown("<h1 class='main-title'>🧪 Etch AI Decision Support System</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>반도체 Etch 공정 품질 예측 및 의사결정 지원 시스템</p>",
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["🔮 공정 예측 · 평가 · 추천", "📊 Process Dashboard"])

    # ---- Tab 1: 입력(A~D) → 예측/평가/추천/비교(Output A~D) ----
    with tab1:
        workbook = get_active_workbook()
        inputs, targets, submitted = create_input_panel()

        if submitted:
            result = predict(inputs, workbook["Wafer_Summary"], workbook["Recipe_Master"])
            evaluation = evaluate_against_target(result, targets)
            recommendation = recommend_best_recipe(
                inputs["recipe"], inputs, targets, workbook["Wafer_Summary"], workbook["Recipe_Master"]
            )

            st.session_state.prediction_result = result
            st.session_state.prediction_inputs = inputs
            st.session_state.prediction_targets = targets
            st.session_state.prediction_evaluation = evaluation
            st.session_state.prediction_recommendation = recommendation

        if st.session_state.prediction_result is not None:
            st.markdown("---")
            show_prediction(st.session_state.prediction_result)
            st.markdown("---")
            show_target_evaluation(st.session_state.prediction_evaluation)
            st.markdown("---")
            show_recommendation(
                st.session_state.prediction_inputs, st.session_state.prediction_recommendation, workbook["Recipe_Master"]
            )
            st.markdown("---")
            show_comparison(
                st.session_state.prediction_result,
                st.session_state.prediction_evaluation["score"],
                st.session_state.prediction_recommendation,
            )
        else:
            st.info("공정 조건과 목표 품질을 입력하고 '예측 · 평가 · 추천 실행' 버튼을 눌러주세요.")

    # ---- Tab 2: Process Dashboard (조건 선택 → Summary → 시각화 → Wafer Map → Zone 분석 → AI 분석) ----
    with tab2:
        workbook = get_active_workbook()
        filtered_wafer, filtered_site = create_process_dashboard_selectors(workbook)

        st.markdown("---")
        show_process_summary(filtered_wafer)
        show_quality_visualization(filtered_wafer)
        show_wafer_map(filtered_site)
        zone_summary = show_zone_analysis(filtered_site)
        show_dashboard_ai_analysis(filtered_wafer, zone_summary)


if __name__ == "__main__":
    main()
