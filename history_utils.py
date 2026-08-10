"""
Etch AI Decision Support System - Parameter History(Recipe Change History) 모듈

Recipe_Master.Changed_Params_This_Rev / Change_Notes와 Wafer_Summary 실측치를 조합해
"과거에 특정 종류의 파라미터를 바꾼 Revision과 그때 품질이 어땠는지"를 조회하는 데 필요한
순수 로직만 모아 둔다. Streamlit 렌더링(app.py)과 분리해 테스트하기 쉽게 유지한다.

검색 대상 Revision은 반드시 Changed_Params_This_Rev(실제 컬럼명)로 결정하고,
Change_Notes(자연어)는 설명/상태 태그 보조용으로만 쓴다 — 자연어 키워드 검색으로
Revision을 찾지 않는다.
"""

from __future__ import annotations

import functools
import os

import pandas as pd
import yaml

from model import format_parameter_label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESS_CONFIG_PATH = os.path.join(BASE_DIR, "config", "process_config.yaml")

# UI에 노출하는 필터 값(순서 고정) -> 판정에 쓰는 실제 컬럼 접미사
PARAM_TYPE_LABELS = ["Time", "Pressure", "RF Bias", "Gas Flow / Ratio"]
_PARAM_TYPE_SUFFIXES = [
    ("Time", "_Time_s"),
    ("Pressure", "_Pressure_mT"),
    ("RF Bias", "_RF_Bias_W"),
    ("Gas Flow / Ratio", "_sccm"),
]

# Change_Notes에서 보조적으로 추출하는 변경 단계 태그 (우선순위 순 — 더 "완결된" 상태를 우선 표시)
STAGE_TAG_PRIORITY = ["양산 후보", "최종", "확정", "완료", "1차"]


# ------------------------------------------------------------------------------
# 공정별 설정 (config/process_config.yaml)
# ------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def load_process_config(path: str | None = None) -> dict:
    """isolation/trench별 Depth 컬럼명 등을 YAML에서 읽어온다. 파일이 없거나 깨져 있어도
    Depth_Mean_nm/Depth_Std_nm 기본값으로 안전하게 동작한다 (data_utils가 항상 이 컬럼을 보강함)."""
    target = path or PROCESS_CONFIG_PATH
    fallback = {
        "isolation": {"depth_mean_col": "Depth_Mean_nm", "depth_std_col": "Depth_Std_nm", "depth_unit": "nm"},
        "trench": {"depth_mean_col": "Depth_Mean_nm", "depth_std_col": "Depth_Std_nm", "depth_unit": "nm"},
    }
    try:
        with open(target, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        for process_key, defaults in fallback.items():
            loaded.setdefault(process_key, {})
            for field, default_value in defaults.items():
                loaded[process_key].setdefault(field, default_value)
        return loaded
    except (OSError, yaml.YAMLError):
        return fallback


# ------------------------------------------------------------------------------
# Changed_Params_This_Rev 파싱 / 분류
# ------------------------------------------------------------------------------
def parse_changed_params(raw_value) -> list:
    """Changed_Params_This_Rev 원문을 실제 컬럼명 리스트로 분리한다.

    쉼표로 분리 후 앞뒤 공백을 제거하고, "(no change)" / "(no change / confirmation run)"
    같은 괄호 표기 문구(실제 컬럼명이 아님)와 빈 값은 제외한다. 결측(NaN)이면 빈 리스트."""
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return []
    if not isinstance(raw_value, str):
        return []
    tokens = [t.strip() for t in raw_value.split(",")]
    return [t for t in tokens if t and not t.startswith("(")]


def classify_param_type(param_col: str) -> str | None:
    """실제 컬럼명(S1_RF_Bias_W 등)을 4가지 변경 파라미터 유형 중 하나로 분류.
    분리된 컬럼명 접미사를 정확히 검사하며, 매칭되는 유형이 없으면 None."""
    if not isinstance(param_col, str):
        return None
    for label, suffix in _PARAM_TYPE_SUFFIXES:
        if param_col.endswith(suffix):
            return label
    return None


def classify_changed_params(changed_params: list) -> dict:
    """파싱된 컬럼명 리스트를 유형별로 묶는다. {"Time": [...], "Pressure": [...], ...}
    (유형이 4개보다 적어도 되고, 분류 실패한 컬럼은 조용히 무시한다.)"""
    grouped: dict = {}
    for col in changed_params:
        param_type = classify_param_type(col)
        if param_type is None:
            continue
        grouped.setdefault(param_type, []).append(col)
    return grouped


# ------------------------------------------------------------------------------
# Change_Notes -> 상태 태그 (보조용, 실패해도 검색 결과에 영향 없음)
# ------------------------------------------------------------------------------
def extract_change_stage_tag(change_notes) -> str | None:
    """Change_Notes 문장에서 1차/최종/확정/완료/양산 후보 태그를 보조적으로 추출한다.
    결측이거나 어떤 키워드도 없으면 None. 여러 태그가 동시에 있으면 STAGE_TAG_PRIORITY
    순서로 가장 "완결된" 상태 하나만 고른다."""
    try:
        if not isinstance(change_notes, str) or not change_notes.strip():
            return None
        for tag in STAGE_TAG_PRIORITY:
            if tag in change_notes:
                return tag
    except Exception:
        return None
    return None


# ------------------------------------------------------------------------------
# Revision 정렬 / 이전 값 계산
# ------------------------------------------------------------------------------
def ordered_revisions_with_index(recipe_df: pd.DataFrame) -> list:
    """Recipe_Master에 등장하는 순서(Base, Rev1, Rev2 ...) 그대로, 중복 없이 반환.
    data_utils.ordered_recipe_versions와 동일한 규칙(문자열 정렬 금지)을 이 모듈 안에서도
    독립적으로 보장하기 위해 별도로 둔다."""
    seen, result = set(), []
    for r in recipe_df["Recipe_Version"]:
        if r not in seen:
            seen.add(r)
            result.append(r)
    return result


def previous_revision_row(recipe_df: pd.DataFrame, revision: str):
    """정렬된 Recipe_Master에서 주어진 Revision '바로 앞에 실제로 존재하는' Revision의 행을 반환.
    Rev 번호에서 1을 빼서 찾지 않는다. 첫 Revision이거나 못 찾으면 None."""
    ordered = ordered_revisions_with_index(recipe_df)
    if revision not in ordered:
        return None
    idx = ordered.index(revision)
    if idx == 0:
        return None
    prev_revision = ordered[idx - 1]
    match = recipe_df[recipe_df["Recipe_Version"] == prev_revision]
    if match.empty:
        return None
    return match.iloc[0]


# ------------------------------------------------------------------------------
# Revision별 변경 이력 (Changed_Params_This_Rev 기준 — Change_Notes는 설명용)
# ------------------------------------------------------------------------------
def revision_change_entries(recipe_df: pd.DataFrame) -> list:
    """Recipe_Master의 각 Revision을 실제 변경 이력 기준으로 정리한다.
    Base/(no change)/confirmation run처럼 실제로 바뀐 파라미터가 없는 Revision은 제외한다."""
    entries = []
    for _, row in recipe_df.iterrows():
        revision = row.get("Recipe_Version")
        changed_params = parse_changed_params(row.get("Changed_Params_This_Rev"))
        if not changed_params:
            continue
        entries.append({
            "revision": revision,
            "changed_params": changed_params,
            "changed_by_type": classify_changed_params(changed_params),
            "change_notes": row.get("Change_Notes"),
            "stage_tag": extract_change_stage_tag(row.get("Change_Notes")),
        })
    return entries


def filter_revisions_by_param_type(recipe_df: pd.DataFrame, param_type: str | None) -> list:
    """변경 이력 중 선택한 파라미터 유형이 포함된 Revision만 반환.
    param_type이 None/"전체"면 실제 변경이 있었던 모든 Revision을 반환.
    한 Revision에서 여러 유형이 동시에 바뀌었으면 관련된 모든 필터 결과에 포함된다."""
    entries = revision_change_entries(recipe_df)
    if not param_type or param_type == "전체":
        return entries
    return [e for e in entries if param_type in e["changed_by_type"]]


# ------------------------------------------------------------------------------
# 변경 상세 (이전 값 -> 변경 값 -> 변화량/변화율)
# ------------------------------------------------------------------------------
def build_param_change_detail(recipe_row, prev_row, changed_params: list) -> list:
    """한 Revision에서 실제로 바뀐 각 파라미터에 대해 이전 값/변경 값/변화량/변화율을 계산.
    이전 Revision이 없거나, 이전/현재 값이 없거나 숫자가 아니면 해당 항목만 N/A로 표시하고
    나머지 항목 계산에는 영향을 주지 않는다."""
    rows = []
    for col in changed_params:
        try:
            new_value = recipe_row[col] if (recipe_row is not None and col in recipe_row.index) else None
        except Exception:
            new_value = None
        prev_value = None
        if prev_row is not None:
            try:
                prev_value = prev_row[col] if col in prev_row.index else None
            except Exception:
                prev_value = None

        abs_delta = None
        pct_change = None
        try:
            if prev_value is not None and new_value is not None and pd.notna(prev_value) and pd.notna(new_value):
                prev_num = float(prev_value)
                new_num = float(new_value)
                abs_delta = new_num - prev_num
                pct_change = (abs_delta / prev_num * 100) if prev_num != 0 else None
        except (TypeError, ValueError):
            abs_delta = None
            pct_change = None

        rows.append({
            "param_col": col,
            "label": format_parameter_label(col),
            "stage": col.split("_")[0],
            "param_type": classify_param_type(col),
            "prev_value": prev_value if prev_value is None or pd.notna(prev_value) else None,
            "new_value": new_value if new_value is None or pd.notna(new_value) else None,
            "abs_delta": abs_delta,
            "pct_change": pct_change,
        })
    return rows


# ------------------------------------------------------------------------------
# Equipment Model 필터
# ------------------------------------------------------------------------------
def get_equipment_models(wafer_df: pd.DataFrame) -> list:
    """Wafer_Summary에 존재하는 Equipment Model 목록(정렬)을 반환. 컬럼이 없거나 비어 있으면 빈 리스트."""
    if wafer_df is None or wafer_df.empty or "Equipment_Model" not in wafer_df.columns:
        return []
    return sorted(v for v in wafer_df["Equipment_Model"].dropna().unique())


def filter_wafer_by_equipment(wafer_df: pd.DataFrame, equipment: str | None) -> pd.DataFrame:
    """equipment가 None/"전체"면 원본 그대로, 아니면 해당 Equipment_Model의 Wafer만 반환."""
    if wafer_df is None:
        return pd.DataFrame()
    if not equipment or equipment == "전체" or "Equipment_Model" not in wafer_df.columns:
        return wafer_df
    return wafer_df[wafer_df["Equipment_Model"] == equipment]


# ------------------------------------------------------------------------------
# Revision별 품질 집계
# ------------------------------------------------------------------------------
def aggregate_revision_quality(wafer_df: pd.DataFrame, revision: str, process: str) -> dict | None:
    """선택한 Equipment 조건까지 반영된 Wafer_Summary에서 해당 Revision의 품질을 집계.
    평가 기록이 없으면 None (호출부에서 해당 Revision을 결과에서 제외하는 데 사용)."""
    if wafer_df is None or wafer_df.empty or "Recipe_Version" not in wafer_df.columns:
        return None
    subset = wafer_df[wafer_df["Recipe_Version"] == revision]
    if subset.empty:
        return None

    process_config = load_process_config()
    depth_mean_col = process_config.get(process, {}).get("depth_mean_col", "Depth_Mean_nm")

    def _safe_mean(col: str):
        if col not in subset.columns:
            return None
        series = pd.to_numeric(subset[col], errors="coerce")
        if series.dropna().empty:
            return None
        return float(series.mean())

    uniformity_cols = [
        c for c in [
            "Top_CD_Uniformity_pct", "Mid_CD_Uniformity_pct", "Bottom_CD_Uniformity_pct", "Depth_Uniformity_pct",
        ] if c in subset.columns
    ]
    uniformity_mean = None
    if uniformity_cols:
        series = pd.to_numeric(subset[uniformity_cols].stack(), errors="coerce")
        if not series.dropna().empty:
            uniformity_mean = float(series.mean())

    return {
        "wafer_count": int(subset["Wafer_ID"].nunique()) if "Wafer_ID" in subset.columns else int(len(subset)),
        "top_cd_mean": _safe_mean("Top_CD_Mean_nm"),
        "mid_cd_mean": _safe_mean("Mid_CD_Mean_nm"),
        "bottom_cd_mean": _safe_mean("Bottom_CD_Mean_nm"),
        "depth_mean": _safe_mean(depth_mean_col),
        "pass_rate_mean": _safe_mean("Overall_Spec_Pass_Rate_pct"),
        "defect_count_mean": _safe_mean("Total_Defect_Count"),
        "uniformity_mean": uniformity_mean,
        "equipment_models": sorted(subset["Equipment_Model"].dropna().unique()) if "Equipment_Model" in subset.columns else [],
        "chambers": sorted(subset["Chamber_ID"].dropna().unique()) if "Chamber_ID" in subset.columns else [],
        "eval_period": _eval_period(subset),
    }


def _eval_period(subset: pd.DataFrame) -> str | None:
    """평가 기간을 "가장 이른 시각 ~ 가장 늦은 시각" 문자열로 요약. Eval_Timestamp가 없거나
    파싱할 수 없으면 None (표시 쪽에서 '—' 처리)."""
    if "Eval_Timestamp" not in subset.columns:
        return None
    ts = pd.to_datetime(subset["Eval_Timestamp"], errors="coerce").dropna()
    if ts.empty:
        return None
    start, end = ts.min(), ts.max()
    fmt = "%Y-%m-%d %H:%M"
    if start == end:
        return start.strftime(fmt)
    return f"{start.strftime(fmt)} ~ {end.strftime(fmt)}"


# ------------------------------------------------------------------------------
# 화면에서 호출하는 최상위 조회 함수
# ------------------------------------------------------------------------------
def build_history_results(recipe_df: pd.DataFrame, wafer_df: pd.DataFrame, process: str,
                           param_type: str | None, equipment: str | None) -> dict:
    """필터(파라미터 유형/Equipment Model) 조건에 맞는 Revision 변경 이력 + 품질 집계를 만든다.
    - Revision 후보는 Changed_Params_This_Rev 기준(filter_revisions_by_param_type)으로만 결정한다.
    - Equipment를 특정 장비로 좁혔을 때 해당 장비 평가 기록이 없는 Revision은 결과에서 빠진다.
    - 이전 값은 Equipment 필터와 무관하게 Recipe_Master 원본 기준으로 계산한다(레시피 조건 자체는
      장비와 무관하게 하나이므로).
    """
    matched_entries = filter_revisions_by_param_type(recipe_df, param_type)
    equipment_wafer_df = filter_wafer_by_equipment(wafer_df, equipment)

    results = []
    for entry in matched_entries:
        revision = entry["revision"]
        quality = aggregate_revision_quality(equipment_wafer_df, revision, process)
        if quality is None:
            continue

        recipe_row = recipe_df[recipe_df["Recipe_Version"] == revision]
        recipe_row = recipe_row.iloc[0] if not recipe_row.empty else None
        prev_row = previous_revision_row(recipe_df, revision)
        detail_rows = build_param_change_detail(recipe_row, prev_row, entry["changed_params"])

        results.append({
            **entry,
            "quality": quality,
            "detail_rows": detail_rows,
        })

    total_wafer = sum(r["quality"]["wafer_count"] for r in results)
    pass_rates = [r["quality"]["pass_rate_mean"] for r in results if r["quality"]["pass_rate_mean"] is not None]
    matched_param_count = sum(
        len(r["changed_by_type"][param_type]) if param_type and param_type != "전체" else len(r["changed_params"])
        for r in results
    )

    return {
        "revisions": results,
        "revision_count": len(results),
        "total_wafer_count": total_wafer,
        "avg_pass_rate": (sum(pass_rates) / len(pass_rates)) if pass_rates else None,
        "matched_param_count": matched_param_count,
    }
