"""
Etch AI Decision Support System - 데이터 유틸리티 모듈

실제 업로드 대상 워크북(isolation / trench)은 3개 시트로 구성된다.

- Recipe_Master  : Recipe_Version별 Stage(S1~S2 또는 S1~S4) 공정 조건 (Time/Gas/RF Bias/Pressure)
- Wafer_Summary  : Wafer 1장당 1행 (Top/Mid/Bottom CD, Depth, Uniformity%, Pass Rate, Defect)
- Site_Level_Raw : Wafer 1장당 25개 측정 Site (Zone/Radius_frac/Angle_deg 포함, Wafer Map용)

Process Dashboard와 predict()는 이 3-시트 구조를 전제로 동작한다.
isolation은 Depth 원본 단위가 Angstrom(A)이라 로딩 시 nm으로 환산한 컬럼을 추가해,
이후 로직(charts.py 포함)은 항상 *_nm 컬럼만 보면 되도록 통일한다.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# 워크북 구조 정의
# ----------------------------------------------------------------------------
REQUIRED_SHEETS = ["Recipe_Master", "Wafer_Summary", "Site_Level_Raw"]

# Wafer Map / Zone 분석에서 사용하는 Zone 순서 및 대표 반경(중심=0 ~ 바깥쪽=1에 가까움)
ZONE_ORDER = ["Center", "Mid", "Edge", "Extreme Edge"]
ZONE_RADIUS = {"Center": 0.0, "Mid": 0.55, "Edge": 0.85, "Extreme Edge": 0.97}

# Total_Defect_Count(Wafer 1장 = Site 25개 합산)는 실제 데이터 기준 대다수가 1건 이상이라
# ">0"을 "Particle 발생" 기준으로 쓰면 거의 모든 Wafer가 발생으로 잡혀 의미가 없어진다.
# 전체 분포(중앙값 근처)를 참고해 "3건 초과"를 발생 기준으로 사용한다 (근사치, 조정 가능).
PARTICLE_DEFECT_THRESHOLD = 3

_DUMMY_RECIPES = ["Base", "Rev1", "Rev2", "Rev3", "Rev4", "Rev5"]
_DUMMY_EQUIPMENTS = ["EQP-A", "EQP-B"]
_DUMMY_CHAMBERS = ["CH-A", "CH-B"]

# ----------------------------------------------------------------------------
# 공정별 Stage 정의
# ----------------------------------------------------------------------------
TRENCH_STAGE_DEFS = [
    {"key": "S1", "label": "S1 (SiON Strip)", "time_col": "S1_Time_s",
     "gas_cols": ["S1_CF4_sccm", "S1_CHF3_sccm"], "bias_col": "S1_RF_Bias_W", "pressure_col": "S1_Pressure_mT"},
    {"key": "S2", "label": "S2 (SOC Open)", "time_col": "S2_Time_s",
     "gas_cols": ["S2_O2_sccm", "S2_N2_sccm"], "bias_col": "S2_RF_Bias_W", "pressure_col": "S2_Pressure_mT"},
    {"key": "S3", "label": "S3 (SiO2 HM)", "time_col": "S3_Time_s",
     "gas_cols": ["S3_CF4_sccm", "S3_C4F8_sccm"], "bias_col": "S3_RF_Bias_W", "pressure_col": "S3_Pressure_mT"},
    {"key": "S4", "label": "S4 (Si Main)", "time_col": "S4_Time_s",
     "gas_cols": ["S4_HBr_sccm", "S4_Cl2_sccm"], "bias_col": "S4_RF_Bias_W", "pressure_col": "S4_Pressure_mT"},
]

ISOLATION_STAGE_DEFS = [
    {"key": "S1", "label": "S1 (SiO2 Main Etch)", "time_col": "S1_Time_s",
     "gas_cols": ["S1_CHF3_sccm", "S1_C4F8_sccm", "S1_O2_sccm"], "bias_col": "S1_RF_Bias_W", "pressure_col": "S1_Pressure_mT"},
    {"key": "S2", "label": "S2 (PolySi Etch)", "time_col": "S2_Time_s",
     "gas_cols": ["S2_HBr_sccm", "S2_Cl2_sccm", "S2_O2_sccm"], "bias_col": "S2_RF_Bias_W", "pressure_col": "S2_Pressure_mT"},
]

GATE_STAGE_DEFS = [
    {"key": "S1", "label": "S1 (PolySi Gate Etch, stop on Gate Oxide)", "time_col": "Time_s",
     "gas_cols": ["HBr_sccm", "Cl2_sccm"], "bias_col": "RF_Bias_W", "pressure_col": "Pressure_mT"},
]

METAL_STAGE_DEFS = [
    {"key": "S1", "label": "S1 (Al Metal Line Etch, stop on TiN Barrier)", "time_col": "Time_s",
     "gas_cols": ["Cl2_sccm", "BCl3_sccm"], "bias_col": "RF_Bias_W", "pressure_col": "Pressure_mT"},
]

PROCESS_ETCH_TARGET_MATERIAL = {
    "isolation": "SiO2 / PolySi (STI)",
    "trench": "Si (Trench Main), HM: SiO2/SiON",
    "gate": "PolySi (Gate)",
    "metal": "Al (Metal Line)",
}
# 신규 공정 프리뷰(유사도 기반, 편법)에서 "이 물질과 겹치는 기존 공정"을 찾을 때 쓰는 키워드 집합.
# 정식 예측이 아니라 참고용 실측 조회이므로 엄격한 물질 DB 대신 단순 키워드 매칭으로 충분하다.
PROCESS_MATERIAL_KEYWORDS = {
    "isolation": {"sio2", "polysi", "si", "sti"},
    "trench": {"sion", "soc", "sio2", "si", "trench"},
    "gate": {"polysi", "si", "gate"},
    "metal": {"al", "aluminum", "tin", "metal"},
}

PROCESS_STAGE_DEFS = {
    "isolation": ISOLATION_STAGE_DEFS, "trench": TRENCH_STAGE_DEFS,
    "gate": GATE_STAGE_DEFS, "metal": METAL_STAGE_DEFS,
}
PROCESS_LABELS = {
    "isolation": "Isolation (STI) Etch", "trench": "Trench Etch",
    "gate": "Gate PolySi Etch", "metal": "Metal (Al) Etch",
}

# 하위 호환용 기본값(과거 코드가 STAGE_DEFS를 직접 참조하던 부분 대비)
STAGE_DEFS = TRENCH_STAGE_DEFS

# isolation은 Depth 원본이 Angstrom. 로딩 시 nm 환산 컬럼을 만들어 이후 로직은 전부 nm만 보게 한다.
_ANGSTROM_PER_NM = 10.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_FILES = {
    "isolation": os.path.join(BASE_DIR, "data", "isolation_dataset.xlsx"),
    "trench": os.path.join(BASE_DIR, "data", "trench_dataset.xlsx"),
    "gate": os.path.join(BASE_DIR, "data", "gate_dataset.xlsx"),
    "metal": os.path.join(BASE_DIR, "data", "metal_dataset.xlsx"),
}


def build_required_columns(process_key: str) -> dict:
    """공정(Stage 수가 다름)에 맞춰 업로드 검증용 필수 컬럼을 동적으로 구성."""
    stage_defs = PROCESS_STAGE_DEFS[process_key]
    first, last = stage_defs[0], stage_defs[-1]
    return {
        "Recipe_Master": [
            "Recipe_Version",
            first["time_col"], first["bias_col"], first["pressure_col"],
            last["time_col"], last["bias_col"], last["pressure_col"],
        ],
        "Wafer_Summary": [
            "Recipe_Version", "Wafer_ID", "Lot_ID", "Equipment_Model", "Chamber_ID",
            "Top_CD_Mean_nm", "Mid_CD_Mean_nm", "Bottom_CD_Mean_nm",
            "Top_CD_Uniformity_pct", "Mid_CD_Uniformity_pct", "Bottom_CD_Uniformity_pct",
            "Total_Defect_Count", "Overall_Spec_Pass_Rate_pct",
        ],
        "Site_Level_Raw": [
            "Recipe_Version", "Wafer_ID", "Lot_ID", "Equipment_Model", "Chamber_ID",
            "Zone", "Radius_frac", "Angle_deg",
            "Top_CD_nm", "Mid_CD_nm", "Bottom_CD_nm",
            "Top_CD_Spec_Pass", "Mid_CD_Spec_Pass", "Bottom_CD_Spec_Pass", "Depth_Spec_Pass",
            "Defect_Count",
        ],
    }


def is_valid_workbook(sheets: dict, process_key: str = "trench") -> bool:
    """업로드된 시트 dict가 선택한 공정이 요구하는 3-시트 구조를 갖췄는지 확인.
    Depth 컬럼은 공정마다 단위가 달라(_A vs _nm) 별도로 확인한다."""
    if not sheets:
        return False
    required = build_required_columns(process_key)
    for sheet_name, required_cols in required.items():
        df = sheets.get(sheet_name)
        if df is None or not all(col in df.columns for col in required_cols):
            return False

    wafer_df = sheets.get("Wafer_Summary")
    site_df = sheets.get("Site_Level_Raw")
    has_depth_wafer = any(c in wafer_df.columns for c in ("Depth_Mean_nm", "Depth_Mean_A"))
    has_depth_site = any(c in site_df.columns for c in ("Depth_nm", "Depth_A"))
    return bool(has_depth_wafer and has_depth_site)


def _normalize_depth_units(sheets: dict) -> dict:
    """Angstrom 단위 Depth 컬럼(isolation)을 nm 환산 컬럼으로 보강.
    이미 nm 컬럼이 있으면(trench) 그대로 둔다."""
    wafer_df = sheets.get("Wafer_Summary")
    site_df = sheets.get("Site_Level_Raw")

    if wafer_df is not None and "Depth_Mean_nm" not in wafer_df.columns and "Depth_Mean_A" in wafer_df.columns:
        wafer_df = wafer_df.copy()
        wafer_df["Depth_Mean_nm"] = wafer_df["Depth_Mean_A"] / _ANGSTROM_PER_NM
        if "Depth_Std_A" in wafer_df.columns:
            wafer_df["Depth_Std_nm"] = wafer_df["Depth_Std_A"] / _ANGSTROM_PER_NM
        sheets = {**sheets, "Wafer_Summary": wafer_df}

    if site_df is not None and "Depth_nm" not in site_df.columns and "Depth_A" in site_df.columns:
        site_df = site_df.copy()
        site_df["Depth_nm"] = site_df["Depth_A"] / _ANGSTROM_PER_NM
        sheets = {**sheets, "Site_Level_Raw": site_df}

    return sheets


def load_required_sheets(excel_file: pd.ExcelFile) -> dict:
    """엑셀 파일에서 REQUIRED_SHEETS에 해당하는 시트만 읽어 dict로 반환 (없으면 빈 dict)"""
    sheets = {}
    for name in REQUIRED_SHEETS:
        if name in excel_file.sheet_names:
            sheets[name] = excel_file.parse(name)
    return _normalize_depth_units(sheets)


@st.cache_data(show_spinner="기본 제공 데이터를 불러오는 중입니다...")
def load_bundled_workbook(process_key: str) -> dict:
    """레포에 함께 배포된 실측 데이터셋(data/*.xlsx)을 기본값으로 불러온다.
    업로드를 하지 않아도 배포된 대시보드가 바로 동작하도록 하기 위함."""
    path = DEFAULT_DATA_FILES[process_key]
    if not os.path.exists(path):
        return generate_dummy_workbook()
    xls = pd.ExcelFile(path)
    sheets = {}
    for name in REQUIRED_SHEETS:
        if name in xls.sheet_names:
            sheets[name] = xls.parse(name)
    return _normalize_depth_units(sheets)


# ----------------------------------------------------------------------------
# 더미 워크북 생성 (배포 데이터도 없고 업로드도 없을 때의 최종 폴백)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_dummy_workbook(wafers_per_combo: int = 3, seed: int = 42) -> dict:
    """실제 워크북과 유사한 3-시트 스키마의 더미 데이터 생성 (trench 4-Stage 형태 기준)"""
    rng = np.random.default_rng(seed)
    _SITE_TEMPLATE = (
        [("C1", "Center", 0.0, None)]
        + [
            (sid, "Mid", 0.55, angle)
            for sid, angle in [("Mid-Right", 0.0), ("Mid-Top", 90.0), ("Mid-Left", 180.0), ("Mid-Bottom", 270.0)]
        ]
        + [(f"Edge-{i:02d}", "Edge", 0.85, (i - 1) * 36.0) for i in range(1, 11)]
        + [(f"ExtEdge-{i:02d}", "Extreme Edge", 0.97, 18.0 + (i - 1) * 36.0) for i in range(1, 11)]
    )

    recipe_rows = []
    for i, recipe in enumerate(_DUMMY_RECIPES):
        progress = i / (len(_DUMMY_RECIPES) - 1)
        recipe_rows.append({
            "Recipe_Version": recipe,
            "S1_Time_s": 8.0, "S1_CF4_sccm": 30, "S1_CHF3_sccm": 20.0,
            "S1_RF_Bias_W": 100, "S1_Pressure_mT": 15,
            "S2_Time_s": 20.0, "S2_O2_sccm": 40, "S2_N2_sccm": 20,
            "S2_RF_Bias_W": 90, "S2_Pressure_mT": 20,
            "S3_Time_s": 15, "S3_CF4_sccm": 25, "S3_C4F8_sccm": 8.0,
            "S3_RF_Bias_W": 110, "S3_Pressure_mT": 18,
            "S4_Time_s": round(70 + progress * 8, 1),
            "S4_HBr_sccm": 140, "S4_Cl2_sccm": 90,
            "S4_RF_Bias_W": round(160 + progress * 40),
            "S4_Pressure_mT": round(35 - progress * 7, 1),
            "Change_Notes": f"{recipe} 더미 레시피",
        })
    recipe_master = pd.DataFrame(recipe_rows)

    wafer_rows, site_rows = [], []
    wafer_seq = 1
    for i, recipe in enumerate(_DUMMY_RECIPES):
        progress = i / (len(_DUMMY_RECIPES) - 1)
        base_top, base_mid = 260.0, 260.0
        base_bottom = 265.0 - progress * 30.0
        base_depth = 1050.0 + progress * 150.0
        base_pass = 20.0 + progress * 78.0
        base_cd_var = 3.5 - progress * 2.3
        base_depth_var = 30.0 - progress * 27.0

        for equipment in _DUMMY_EQUIPMENTS:
            for chamber in _DUMMY_CHAMBERS:
                for w in range(wafers_per_combo):
                    wafer_id = f"W{wafer_seq:04d}"
                    lot_id = f"LOT-{wafer_seq:04d}"
                    wafer_seq += 1

                    top_cd = base_top + rng.normal(0, 3)
                    mid_cd = base_mid + rng.normal(0, 3)
                    bottom_cd = base_bottom + rng.normal(0, 3)
                    depth = base_depth + rng.normal(0, 15)
                    cd_var = max(0.3, base_cd_var + rng.normal(0, 0.3))
                    depth_var = max(0.3, base_depth_var + rng.normal(0, 2))
                    pass_rate = float(np.clip(base_pass + rng.normal(0, 5), 1, 100))
                    defect_count = int(max(0, rng.poisson(max(0.3, 12 * (1 - progress)))))

                    wafer_rows.append({
                        "Recipe_Version": recipe, "Version_No": i, "Lot_ID": lot_id, "Wafer_ID": wafer_id,
                        "Eval_Timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(days=wafer_seq),
                        "Equipment_Model": equipment, "Chamber_ID": chamber,
                        "Top_CD_Mean_nm": round(top_cd, 2), "Top_CD_Std_nm": round(abs(rng.normal(2, 0.5)), 2),
                        "Top_CD_Uniformity_pct": round(cd_var, 2),
                        "Mid_CD_Mean_nm": round(mid_cd, 2), "Mid_CD_Std_nm": round(abs(rng.normal(2, 0.5)), 2),
                        "Mid_CD_Uniformity_pct": round(cd_var * 0.9, 2),
                        "Bottom_CD_Mean_nm": round(bottom_cd, 2), "Bottom_CD_Std_nm": round(abs(rng.normal(2, 0.5)), 2),
                        "Bottom_CD_Uniformity_pct": round(cd_var * 1.1, 2),
                        "Depth_Mean_nm": round(depth, 1), "Depth_Std_nm": round(abs(rng.normal(10, 3)), 2),
                        "Depth_Uniformity_pct": round(depth_var, 2),
                        "Total_Defect_Count": defect_count,
                        "Avg_Defect_Severity": round(float(rng.uniform(0, 5)), 2) if defect_count else 0.0,
                        "Overall_Spec_Pass_Rate_pct": round(pass_rate, 1),
                    })

                    for site_id, zone, radius, angle in _SITE_TEMPLATE:
                        edge_drift = radius * rng.uniform(-1.4, -0.3)
                        s_top = top_cd + edge_drift + rng.normal(0, 1.5)
                        s_mid = mid_cd + edge_drift * 0.9 + rng.normal(0, 1.5)
                        s_bottom = bottom_cd + edge_drift * 0.7 + rng.normal(0, 1.8)
                        s_depth = depth + edge_drift * 6 + rng.normal(0, 8)
                        site_defect = int(rng.random() < (defect_count / 25.0))

                        site_rows.append({
                            "Recipe_Version": recipe, "Version_No": i, "Lot_ID": lot_id, "Wafer_ID": wafer_id,
                            "Equipment_Model": equipment, "Chamber_ID": chamber,
                            "Site_ID": site_id, "Zone": zone, "Radius_frac": radius, "Angle_deg": angle,
                            "Top_CD_nm": round(s_top, 2), "Mid_CD_nm": round(s_mid, 2),
                            "Bottom_CD_nm": round(s_bottom, 2), "Depth_nm": round(s_depth, 1),
                            "Top_CD_Spec_Pass": bool(rng.random() < pass_rate / 100),
                            "Mid_CD_Spec_Pass": bool(rng.random() < pass_rate / 100),
                            "Bottom_CD_Spec_Pass": bool(rng.random() < pass_rate / 100),
                            "Depth_Spec_Pass": bool(rng.random() < pass_rate / 100),
                            "Defect_Count": site_defect,
                        })

    return {
        "Recipe_Master": recipe_master,
        "Wafer_Summary": pd.DataFrame(wafer_rows),
        "Site_Level_Raw": pd.DataFrame(site_rows),
    }


# ----------------------------------------------------------------------------
# 집계 헬퍼
# ----------------------------------------------------------------------------
def get_zone_summary(site_df: pd.DataFrame) -> pd.DataFrame:
    """Site_Level_Raw(필터링된)를 Zone별로 집계 (Zone 분석, Wafer Map 캡션에 사용)

    CD Spread / Depth Spread: 실제 데이터에는 Zone 단위 Uniformity% 컬럼이 없어서,
    같은 Zone 안의 여러 Site(각도별 측정점)가 Wafer 한 장 안에서 얼마나 흩어져 있는지
    (Wafer별 Site간 표준편차의 평균)를 Zone별 균일도 대체 지표로 사용한다.
    Center Zone은 Site가 1개뿐이라 표준편차를 계산할 수 없어 NaN(그래프에서 생략)이 된다."""
    if site_df.empty:
        return pd.DataFrame(columns=[
            "Zone", "Top CD", "Mid CD", "Bottom CD", "Depth",
            "Pass Rate", "Defect Rate", "CD Spread", "Depth Spread",
        ])

    zone_order = [z for z in ZONE_ORDER if z in site_df["Zone"].unique()]

    def _pass_rate(d):
        all_pass = d["Top_CD_Spec_Pass"] & d["Mid_CD_Spec_Pass"] & d["Bottom_CD_Spec_Pass"] & d["Depth_Spec_Pass"]
        return all_pass.mean() * 100

    grouped = site_df.groupby("Zone")
    per_wafer_zone = site_df.groupby(["Zone", "Wafer_ID"])
    cd_spread = per_wafer_zone[["Top_CD_nm", "Mid_CD_nm", "Bottom_CD_nm"]].std().mean(axis=1).groupby("Zone").mean()
    depth_spread = per_wafer_zone["Depth_nm"].std().groupby("Zone").mean()

    summary = pd.DataFrame({
        "Top CD": grouped["Top_CD_nm"].mean(),
        "Mid CD": grouped["Mid_CD_nm"].mean(),
        "Bottom CD": grouped["Bottom_CD_nm"].mean(),
        "Depth": grouped["Depth_nm"].mean(),
        "Pass Rate": grouped.apply(_pass_rate),
        "Defect Rate": grouped["Defect_Count"].apply(lambda s: (s > 0).mean() * 100),
        "CD Spread": cd_spread,
        "Depth Spread": depth_spread,
    }).reindex(zone_order).reset_index()

    return summary


def get_recipe_stage_table(recipe_master_df: pd.DataFrame, recipe_version: str, stage_defs=None) -> pd.DataFrame:
    """선택한 Recipe_Version의 Stage별 조건을 보기 좋은 표 형태로 변환"""
    stage_defs = stage_defs or STAGE_DEFS
    match = recipe_master_df[recipe_master_df["Recipe_Version"] == recipe_version]
    if match.empty:
        return pd.DataFrame()
    row = match.iloc[0]

    records = []
    for stage in stage_defs:
        gas_str = " / ".join(f"{c.split('_')[1]} {row[c]:g}sccm" for c in stage["gas_cols"] if c in row.index)
        records.append({
            "Stage": stage["label"],
            "Time (s)": row.get(stage["time_col"], None),
            "Gas": gas_str,
            "RF Bias (W)": row.get(stage["bias_col"], None),
            "Pressure (mT)": row.get(stage["pressure_col"], None),
        })
    return pd.DataFrame(records)


def stage_inputs_from_recipe(recipe_master_df: pd.DataFrame, recipe_version: str, stage_defs=None) -> dict:
    """선택한 Recipe의 실제 Stage별 Time/RF Bias/Pressure를 dict로 반환
    (입력 패널 기본값 채우기 + 후보 Recipe를 '그 레시피 그대로' 평가할 때 사용)"""
    stage_defs = stage_defs or STAGE_DEFS
    match = recipe_master_df[recipe_master_df["Recipe_Version"] == recipe_version]
    if match.empty:
        return {}
    row = match.iloc[0]
    result = {}
    for stage in stage_defs:
        key = stage["key"].lower()
        result[f"{key}_time"] = float(row[stage["time_col"]])
        result[f"{key}_rf_bias"] = float(row[stage["bias_col"]])
        result[f"{key}_pressure"] = float(row[stage["pressure_col"]])
        for gas_col in stage["gas_cols"]:
            if gas_col in row.index:
                result[gas_col.lower()] = float(row[gas_col])
    return result


def ordered_recipe_versions(recipe_df: pd.DataFrame, subset_values=None) -> list:
    """Recipe_Master에 등장하는 순서(Base, Rev1, Rev2...)를 그대로 유지해서 반환.
    문자열 sorted()를 쓰면 'Rev10'이 'Rev2'보다 앞에 오는 등 순서가 깨진다."""
    order = list(recipe_df["Recipe_Version"])
    if subset_values is not None:
        allowed = set(subset_values)
        order = [r for r in order if r in allowed]
    seen, result = set(), []
    for r in order:
        if r not in seen:
            seen.add(r)
            result.append(r)
    return result


def get_default_targets(wafer_df: pd.DataFrame, recipe_df: pd.DataFrame, recipe: str | None = None) -> dict:
    """목표 품질(Input C) 기본값. Target CD/Depth는 선택된 Recipe(참고용 베이스라인)의 실측 평균을 사용해
    "참고값 → 목표값" 형태로 보여줄 수 있게 한다. recipe를 안 넘기면 가장 마지막(최신/성숙) Recipe를 쓴다.
    Uniformity/Pass Rate/Defect Count 기준은 통상적인 근사 목표치를 기본값으로 둔다 (모두 사용자 조정 가능)."""
    reference_recipe = recipe if recipe is not None else recipe_df["Recipe_Version"].iloc[-1]
    subset = wafer_df[wafer_df["Recipe_Version"] == reference_recipe]
    if subset.empty:
        subset = wafer_df
    return {
        "target_top_cd": round(float(subset["Top_CD_Mean_nm"].mean()), 1),
        "target_mid_cd": round(float(subset["Mid_CD_Mean_nm"].mean()), 1),
        "target_bottom_cd": round(float(subset["Bottom_CD_Mean_nm"].mean()), 1),
        "target_depth": round(float(subset["Depth_Mean_nm"].mean()), 1),
        "max_cd_uniformity": 3.0,
        "max_depth_uniformity": 3.0,
        "min_pass_rate": 95.0,
        "max_defect_count": 3.0,
    }
