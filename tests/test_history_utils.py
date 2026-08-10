"""
history_utils.py(Parameter History 로직) 테스트.

실제 배포 데이터(data/*.xlsx)를 읽지 않고, Recipe_Master/Wafer_Summary와 동일한 컬럼
구조를 가진 최소 fixture로 검증한다 — 파싱/분류/정렬/필터/집계 로직이 스키마만 맞으면
공정과 무관하게 동작해야 하기 때문이다.
"""

import math

import numpy as np
import pandas as pd
import pytest

import history_utils as hu


@pytest.fixture
def recipe_df() -> pd.DataFrame:
    rows = [
        {
            "Recipe_Version": "Base", "Changed_Params_This_Rev": "(no change)",
            "Change_Notes": "초기 레시피",
            "S1_Time_s": 10, "S1_RF_Bias_W": 100, "S1_Pressure_mT": 0, "S1_O2_sccm": 40,
            "S2_Time_s": 20, "S2_RF_Bias_W": 90, "S2_Pressure_mT": 20, "S2_O2_sccm": 30,
        },
        {
            "Recipe_Version": "Rev1", "Changed_Params_This_Rev": "S1_RF_Bias_W, S1_Pressure_mT",
            "Change_Notes": "S1 RF Bias/Pressure 조정(1차): Top CD 마스크침식 제어.",
            "S1_Time_s": 10, "S1_RF_Bias_W": 118, "S1_Pressure_mT": 12, "S1_O2_sccm": 40,
            "S2_Time_s": 20, "S2_RF_Bias_W": 90, "S2_Pressure_mT": 20, "S2_O2_sccm": 30,
        },
        {
            "Recipe_Version": "Rev2", "Changed_Params_This_Rev": "S2_O2_sccm",
            "Change_Notes": float("nan"),  # 결측 Change_Notes
            "S1_Time_s": 10, "S1_RF_Bias_W": 118, "S1_Pressure_mT": 12, "S1_O2_sccm": 40,
            "S2_Time_s": 20, "S2_RF_Bias_W": 90, "S2_Pressure_mT": 20, "S2_O2_sccm": 45,
        },
        {
            "Recipe_Version": "Rev3", "Changed_Params_This_Rev": "S1_Time_s, S2_Time_s",
            "Change_Notes": "S1/S2 Time 확정(최종): 수렴 완료.",
            "S1_Time_s": 12, "S1_RF_Bias_W": 118, "S1_Pressure_mT": 12, "S1_O2_sccm": 40,
            "S2_Time_s": 25, "S2_RF_Bias_W": 90, "S2_Pressure_mT": 20, "S2_O2_sccm": 45,
        },
    ]
    return pd.DataFrame(rows)


def _wafer_row(revision, equipment, chamber, wafer_id, pass_rate, **overrides):
    row = {
        "Recipe_Version": revision, "Equipment_Model": equipment, "Chamber_ID": chamber,
        "Wafer_ID": wafer_id, "Lot_ID": f"LOT-{wafer_id}", "Eval_Timestamp": "2025-01-07 07:00",
        "Top_CD_Mean_nm": 260.0, "Mid_CD_Mean_nm": 260.0, "Bottom_CD_Mean_nm": 250.0,
        "Top_CD_Uniformity_pct": 2.0, "Mid_CD_Uniformity_pct": 2.0, "Bottom_CD_Uniformity_pct": 2.0,
        "Depth_Mean_nm": 1000.0, "Depth_Std_nm": 10.0, "Depth_Uniformity_pct": 2.0,
        "Total_Defect_Count": 1, "Overall_Spec_Pass_Rate_pct": pass_rate,
    }
    row.update(overrides)
    return row


@pytest.fixture
def wafer_df() -> pd.DataFrame:
    rows = [
        _wafer_row("Rev1", "EQP-A", "CH-A", "W0001", 90, Top_CD_Mean_nm=260.0),
        _wafer_row("Rev1", "EQP-A", "CH-A", "W0002", 92, Top_CD_Mean_nm=262.0),
        _wafer_row("Rev1", "EQP-B", "CH-B", "W0003", 85, Top_CD_Mean_nm=258.0),
        _wafer_row("Rev2", "EQP-A", "CH-A", "W0004", 88),
        _wafer_row("Rev2", "EQP-B", "CH-B", "W0005", 91),
        # Rev3은 EQP-B에서만 평가됨 -> EQP-A로 필터링하면 결과에서 제외되어야 함
        _wafer_row("Rev3", "EQP-B", "CH-C", "W0006", 95),
        _wafer_row("Rev3", "EQP-B", "CH-D", "W0007", 97),
    ]
    return pd.DataFrame(rows)


# 1. 쉼표로 연결된 복수 Changed Params 파싱
def test_parse_changed_params_multiple():
    assert hu.parse_changed_params("S1_RF_Bias_W, S1_Pressure_mT") == ["S1_RF_Bias_W", "S1_Pressure_mT"]
    assert hu.parse_changed_params("S1_Time_s,S2_Time_s") == ["S1_Time_s", "S2_Time_s"]


def test_parse_changed_params_sentinel_and_missing():
    assert hu.parse_changed_params("(no change)") == []
    assert hu.parse_changed_params("(no change / confirmation run)") == []
    assert hu.parse_changed_params(None) == []
    assert hu.parse_changed_params(float("nan")) == []


# 2. Time 분류
def test_classify_time():
    assert hu.classify_param_type("S1_Time_s") == "Time"
    assert hu.classify_param_type("S4_Time_s") == "Time"


# 3. Pressure 분류
def test_classify_pressure():
    assert hu.classify_param_type("S2_Pressure_mT") == "Pressure"


# 4. RF Bias 분류
def test_classify_rf_bias():
    assert hu.classify_param_type("S3_RF_Bias_W") == "RF Bias"


# 5. Gas Flow / Ratio 분류
def test_classify_gas_flow():
    assert hu.classify_param_type("S1_CHF3_sccm") == "Gas Flow / Ratio"
    assert hu.classify_param_type("S4_HBr_sccm") == "Gas Flow / Ratio"


def test_classify_unknown_returns_none():
    assert hu.classify_param_type("Recipe_Version") is None
    assert hu.classify_param_type("") is None


# 6. 한 Revision이 여러 필터에 포함되는 경우
def test_revision_included_in_multiple_filters(recipe_df):
    rf_bias_revs = [e["revision"] for e in hu.filter_revisions_by_param_type(recipe_df, "RF Bias")]
    pressure_revs = [e["revision"] for e in hu.filter_revisions_by_param_type(recipe_df, "Pressure")]
    assert "Rev1" in rf_bias_revs
    assert "Rev1" in pressure_revs


# 7. Revision 숫자 정렬 (문자열 정렬이 아니라 등장 순서를 그대로 유지해야 함)
def test_revision_order_not_lexicographic():
    df = pd.DataFrame({
        "Recipe_Version": ["Base", "Rev1", "Rev2", "Rev10"],
        "Changed_Params_This_Rev": ["(no change)"] * 4,
        "Change_Notes": [""] * 4,
    })
    ordered = hu.ordered_revisions_with_index(df)
    assert ordered == ["Base", "Rev1", "Rev2", "Rev10"]
    assert ordered.index("Rev2") < ordered.index("Rev10")


# 8. 바로 이전에 존재하는 Revision 조회
def test_previous_revision_row(recipe_df):
    prev = hu.previous_revision_row(recipe_df, "Rev1")
    assert prev["Recipe_Version"] == "Base"

    prev2 = hu.previous_revision_row(recipe_df, "Rev3")
    assert prev2["Recipe_Version"] == "Rev2"


def test_previous_revision_row_missing_for_first_or_unknown(recipe_df):
    assert hu.previous_revision_row(recipe_df, "Base") is None
    assert hu.previous_revision_row(recipe_df, "Rev999") is None


# 9. Equipment Model 필터
def test_filter_wafer_by_equipment(wafer_df):
    filtered = hu.filter_wafer_by_equipment(wafer_df, "EQP-A")
    assert set(filtered["Equipment_Model"].unique()) == {"EQP-A"}
    assert len(filtered) == 3  # Rev1 x2 + Rev2 x1

    all_df = hu.filter_wafer_by_equipment(wafer_df, "전체")
    assert len(all_df) == len(wafer_df)


def test_equipment_filter_excludes_revision_without_records(recipe_df, wafer_df):
    results = hu.build_history_results(recipe_df, wafer_df, "trench", "Time", "EQP-A")
    # Rev3(Time 변경)은 EQP-A 평가 기록이 없으므로 제외되어야 함
    assert results["revision_count"] == 0


# 10. Revision별 품질 평균 및 Wafer Count
def test_aggregate_revision_quality(wafer_df):
    quality = hu.aggregate_revision_quality(wafer_df, "Rev1", "trench")
    assert quality["wafer_count"] == 3
    assert quality["top_cd_mean"] == pytest.approx((260.0 + 262.0 + 258.0) / 3)
    assert quality["pass_rate_mean"] == pytest.approx((90 + 92 + 85) / 3)


def test_aggregate_revision_quality_with_equipment_filter(wafer_df):
    eqp_a_only = hu.filter_wafer_by_equipment(wafer_df, "EQP-A")
    quality = hu.aggregate_revision_quality(eqp_a_only, "Rev1", "trench")
    assert quality["wafer_count"] == 2
    assert quality["pass_rate_mean"] == pytest.approx((90 + 92) / 2)


# 11. Change Notes 결측 처리
def test_missing_change_notes_does_not_crash():
    assert hu.extract_change_stage_tag(float("nan")) is None
    assert hu.extract_change_stage_tag(None) is None
    assert hu.extract_change_stage_tag("") is None


def test_build_history_results_survives_missing_change_notes(recipe_df, wafer_df):
    results = hu.build_history_results(recipe_df, wafer_df, "trench", "Gas Flow / Ratio", "전체")
    rev2 = next(r for r in results["revisions"] if r["revision"] == "Rev2")
    assert rev2["stage_tag"] is None
    assert math.isnan(rev2["change_notes"]) if isinstance(rev2["change_notes"], float) else True


# 12. 이전 값이 0인 경우
def test_previous_value_zero_gives_na_pct_change(recipe_df):
    recipe_row = recipe_df[recipe_df["Recipe_Version"] == "Rev1"].iloc[0]
    prev_row = recipe_df[recipe_df["Recipe_Version"] == "Base"].iloc[0]  # S1_Pressure_mT = 0
    detail = hu.build_param_change_detail(recipe_row, prev_row, ["S1_RF_Bias_W", "S1_Pressure_mT"])
    pressure_row = next(r for r in detail if r["param_col"] == "S1_Pressure_mT")
    assert pressure_row["prev_value"] == 0
    assert pressure_row["abs_delta"] == pytest.approx(12.0)
    assert pressure_row["pct_change"] is None  # 0으로 나눌 수 없으니 N/A


# 13. Isolation과 Trench의 서로 다른 Depth 컬럼 처리 (config 기반, 하드코딩 아님)
def test_depth_column_comes_from_process_config(monkeypatch, wafer_df):
    custom_wafer_df = wafer_df.rename(columns={"Depth_Mean_nm": "Depth_Mean_custom_unit"})

    def fake_config():
        return {
            "trench": {"depth_mean_col": "Depth_Mean_custom_unit", "depth_std_col": "Depth_Std_nm"},
            "isolation": {"depth_mean_col": "Depth_Mean_nm", "depth_std_col": "Depth_Std_nm"},
        }

    monkeypatch.setattr(hu, "load_process_config", fake_config)
    quality = hu.aggregate_revision_quality(custom_wafer_df, "Rev1", "trench")
    assert quality["depth_mean"] == pytest.approx(1000.0)


def test_depth_missing_column_is_none_not_crash(wafer_df, monkeypatch):
    def fake_config():
        return {"trench": {"depth_mean_col": "Depth_Mean_does_not_exist"}, "isolation": {}}

    monkeypatch.setattr(hu, "load_process_config", fake_config)
    quality = hu.aggregate_revision_quality(wafer_df, "Rev1", "trench")
    assert quality["depth_mean"] is None


# 14. 검색 결과 없음
def test_no_search_results(recipe_df, wafer_df):
    empty_recipe = pd.DataFrame(columns=recipe_df.columns)
    results = hu.build_history_results(empty_recipe, wafer_df, "trench", "전체", "전체")
    assert results["revision_count"] == 0
    assert results["revisions"] == []
    assert results["avg_pass_rate"] is None


def test_no_search_results_equipment_mismatch(recipe_df, wafer_df):
    results = hu.build_history_results(recipe_df, wafer_df, "trench", "전체", "EQP-NONEXISTENT")
    assert results["revision_count"] == 0


# 그 밖의 방어 로직
def test_build_param_change_detail_handles_missing_columns_gracefully():
    recipe_row = pd.Series({"Recipe_Version": "Rev1", "S1_RF_Bias_W": 118})
    prev_row = pd.Series({"Recipe_Version": "Base", "S1_RF_Bias_W": 100})
    detail = hu.build_param_change_detail(recipe_row, prev_row, ["S1_RF_Bias_W", "S1_Pressure_mT"])
    missing = next(r for r in detail if r["param_col"] == "S1_Pressure_mT")
    assert missing["prev_value"] is None
    assert missing["new_value"] is None
    assert missing["abs_delta"] is None


def test_build_param_change_detail_no_previous_revision():
    recipe_row = pd.Series({"Recipe_Version": "Base", "S1_RF_Bias_W": 100})
    detail = hu.build_param_change_detail(recipe_row, None, ["S1_RF_Bias_W"])
    assert detail[0]["prev_value"] is None
    assert detail[0]["abs_delta"] is None


def test_aggregate_revision_quality_no_wafer_data_returns_none(wafer_df):
    assert hu.aggregate_revision_quality(wafer_df, "Rev999", "trench") is None
    assert hu.aggregate_revision_quality(pd.DataFrame(), "Rev1", "trench") is None


# 평가 기간은 시간 없이 날짜만 표시한다
def test_eval_period_is_date_only(wafer_df):
    quality = hu.aggregate_revision_quality(wafer_df, "Rev1", "trench")
    assert quality["eval_period"] == "2025-01-07"  # 모두 같은 날짜라 단일 날짜만


# 세부 파라미터 드롭다운: 이력에 실제 등장한 컬럼만, 공정별 Stage 라벨
def test_get_detail_param_options_only_actually_changed_columns():
    df = pd.DataFrame([
        {"Recipe_Version": "Rev1", "Changed_Params_This_Rev": "S1_RF_Bias_W"},
        {"Recipe_Version": "Rev2", "Changed_Params_This_Rev": "S2_RF_Bias_W"},
    ])
    # S4_RF_Bias_W 같은 컬럼은 df에 아예 없으니 당연히 옵션에도 없어야 함
    options = hu.get_detail_param_options(df, "RF Bias")
    assert options == ["S1_RF_Bias_W", "S2_RF_Bias_W"]
    assert hu.get_detail_param_options(df, "전체") == []


def test_format_detail_param_label_differs_by_process():
    trench_label = hu.format_detail_param_label("S4_RF_Bias_W", "trench")
    isolation_label = hu.format_detail_param_label("S2_RF_Bias_W", "isolation")
    assert "S4" in trench_label and "Si Main" in trench_label
    assert "S2" in isolation_label and "PolySi" in isolation_label


def test_format_short_step_label():
    assert hu.format_short_step_label("S1_Time_s") == "Step 1 · Time"
    assert hu.format_short_step_label("S4_RF_Bias_W") == "Step 4 · RF Bias"


# 카드용 card_rows: detail_param을 골라도 같은 유형의 다른 Step은 함께 나오고,
# 고른 항목만 selected=True로 맨 앞에 온다. 다른 유형은 card_rows에서 제외된다.
def test_card_rows_group_by_type_and_flag_selected():
    recipe = pd.DataFrame([
        {
            "Recipe_Version": "Base", "Changed_Params_This_Rev": "(no change)", "Change_Notes": "",
            "S1_Time_s": 9.0, "S2_Time_s": 38.0, "S1_Pressure_mT": 12,
        },
        {
            "Recipe_Version": "Rev15", "Changed_Params_This_Rev": "S1_Time_s, S2_Time_s, S1_Pressure_mT",
            "Change_Notes": "최종 양산 후보",
            "S1_Time_s": 10.0, "S2_Time_s": 40.0, "S1_Pressure_mT": 10,
        },
    ])
    wafer = pd.DataFrame([_wafer_row("Rev15", "EQP-A", "CH-A", "W1", 99.7)])

    results = hu.build_history_results(recipe, wafer, "trench", "Time", "전체", detail_param="S2_Time_s")
    rev15 = results["revisions"][0]
    card_cols = [r["param_col"] for r in rev15["card_rows"]]
    assert card_cols == ["S2_Time_s", "S1_Time_s"]  # 선택한 항목이 맨 앞
    assert rev15["card_rows"][0]["selected"] is True
    assert rev15["card_rows"][1]["selected"] is False
    # S1_Pressure_mT(다른 유형)는 card_rows에 없어야 한다 — 상세 보기의 detail_rows에만 있음
    assert "S1_Pressure_mT" not in card_cols
    assert "S1_Pressure_mT" in [r["param_col"] for r in rev15["detail_rows"]]


def test_card_rows_without_detail_param_none_selected():
    recipe = pd.DataFrame([
        {
            "Recipe_Version": "Base", "Changed_Params_This_Rev": "(no change)", "Change_Notes": "",
            "S1_Time_s": 9.0, "S2_Time_s": 38.0,
        },
        {
            "Recipe_Version": "Rev1", "Changed_Params_This_Rev": "S1_Time_s, S2_Time_s", "Change_Notes": "",
            "S1_Time_s": 10.0, "S2_Time_s": 40.0,
        },
    ])
    wafer = pd.DataFrame([_wafer_row("Rev1", "EQP-A", "CH-A", "W1", 99.0)])
    results = hu.build_history_results(recipe, wafer, "trench", "Time", "전체")
    rev1 = results["revisions"][0]
    assert all(r["selected"] is False for r in rev1["card_rows"])


# 직전 Revision 대비 품질 변화 판정 — Change_Notes가 아니라 실측값 기준
def test_compare_revision_quality_classification():
    targets = {"target_top_cd": 250.0}
    prev_quality = {
        "top_cd_mean": 253.0,  # 오차 +3.0
        "pass_rate_mean": 90.0,
        "defect_count_mean": 5.0,
        "uniformity_mean": 3.0,
        "mid_cd_mean": None, "bottom_cd_mean": None, "depth_mean": None,
    }
    curr_quality = {
        "top_cd_mean": 250.5,  # 오차 +0.5 (절대오차 3.0 -> 0.5, 개선)
        "pass_rate_mean": 96.0,  # +6 -> 개선
        "defect_count_mean": 4.9,  # -0.1 -> 임계값(0.1) 이내라 유지
        "uniformity_mean": 4.5,  # +1.5 -> 악화(값 증가는 Uniformity에선 나쁨) -> 확인 필요
        "mid_cd_mean": None, "bottom_cd_mean": None, "depth_mean": None,
    }
    rows = hu.compare_revision_quality(curr_quality, prev_quality, targets)
    by_key = {r["key"]: r for r in rows}
    assert by_key["top_cd"]["classification"] == "개선"
    assert by_key["pass_rate"]["classification"] == "개선"
    assert by_key["defect_count"]["classification"] == "유지"
    assert by_key["uniformity"]["classification"] == "확인 필요"
    # Target이 없는 지표(Pass Rate 등)는 오차 필드가 계산되지 않는다
    assert by_key["pass_rate"]["prev_error"] is None


def test_compare_revision_quality_none_when_missing_prev():
    assert hu.compare_revision_quality({"top_cd_mean": 250.0}, None) is None
    assert hu.compare_revision_quality(None, {"top_cd_mean": 250.0}) is None


def test_build_history_results_attaches_comparison_and_kpi_target_diff():
    recipe = pd.DataFrame([
        {
            "Recipe_Version": "Base", "Changed_Params_This_Rev": "(no change)", "Change_Notes": "",
            "S1_Time_s": 9.0,
        },
        {
            "Recipe_Version": "Rev1", "Changed_Params_This_Rev": "S1_Time_s", "Change_Notes": "",
            "S1_Time_s": 10.0,
        },
    ])
    wafer = pd.DataFrame([
        _wafer_row("Base", "EQP-A", "CH-A", "W0", 90.0, Top_CD_Mean_nm=253.0),
        _wafer_row("Rev1", "EQP-A", "CH-A", "W1", 96.0, Top_CD_Mean_nm=250.5),
    ])
    targets = {"target_top_cd": 250.0}
    results = hu.build_history_results(recipe, wafer, "trench", "Time", "전체", targets=targets)
    rev1 = results["revisions"][0]
    assert rev1["prev_revision"] == "Base"
    assert rev1["comparison"] is not None
    assert rev1["kpi_target_diff"]["top_cd"]["delta"] == pytest.approx(0.5)


def test_build_history_results_comparison_none_without_prev_quality():
    # Rev1의 이전 Revision(Base)이 이 장비에서 평가된 적이 없으면 비교 불가 -> None이어야 함
    recipe = pd.DataFrame([
        {"Recipe_Version": "Base", "Changed_Params_This_Rev": "(no change)", "Change_Notes": "", "S1_Time_s": 9.0},
        {"Recipe_Version": "Rev1", "Changed_Params_This_Rev": "S1_Time_s", "Change_Notes": "", "S1_Time_s": 10.0},
    ])
    wafer = pd.DataFrame([_wafer_row("Rev1", "EQP-A", "CH-A", "W1", 96.0)])
    results = hu.build_history_results(recipe, wafer, "trench", "Time", "전체")
    rev1 = results["revisions"][0]
    assert rev1["prev_quality"] is None
    assert rev1["comparison"] is None
