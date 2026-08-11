"""
Etch AI Decision Support System - 예측/평가/추천 모델 모듈

predict()는 이제 실측 이력 기반 더미가 아니라, 학습된 RandomForest/XGBoost 모델
(ml_engine/{isolation,trench,gate,metal}_core.py)을 호출해 실제 예측을 수행한다.
공정(isolation/trench/gate/metal)에 따라 파라미터 개수·Depth 단위가 다르므로 어댑터에서 흡수한다.

compute_composite_score() / evaluate_against_target() / recommend_best_recipe()는
predict()가 반환하는 dict 형태만 유지되면 그대로 재사용된다 (모델 교체와 무관).
"""

import pandas as pd

from ml_engine import isolation_core, trench_core, gate_core, metal_core
from ml_engine.scoring import compute_composite_score, _proximity_score, _uniformity_score
from data_utils import PARTICLE_DEFECT_THRESHOLD, PROCESS_STAGE_DEFS, PROCESS_LABELS, stage_inputs_from_recipe

_CORES = {"isolation": isolation_core, "trench": trench_core, "gate": gate_core, "metal": metal_core}
_MODEL_DIRS = {
    "isolation": "ml_engine/isolation_models",
    "trench": "ml_engine/trench_models",
    "gate": "ml_engine/gate_models",
    "metal": "ml_engine/metal_models",
}
# 각 공정 엔진이 내부적으로 Depth를 어떤 컬럼명/단위로 예측하는지 (isolation은 학습 데이터 원본이 Angstrom)
_DEPTH_TARGET = {"isolation": "Depth_A", "trench": "Depth_nm", "gate": "Depth_nm", "metal": "Depth_nm"}
_DEPTH_TO_NM = {"isolation": 0.1, "trench": 1.0, "gate": 1.0, "metal": 1.0}


# ==============================================================================
# 0. 입력 dict(Stage별 UI 값) -> 모델 파라미터 dict 변환
# ==============================================================================
def _column_app_key_map(stage_defs: list) -> dict:
    """STAGE_DEFS(time_col/bias_col/pressure_col/gas_cols)를 기준으로 실제 모델 컬럼명 ->
    화면 입력 dict key 매핑을 만든다. Stage 접두사(S1_ 등)가 있든 없든(1-Stage 공정) 항상 정확하다."""
    mapping = {}
    for stage in stage_defs:
        key = stage["key"].lower()
        mapping[stage["time_col"]] = f"{key}_time"
        mapping[stage["bias_col"]] = f"{key}_rf_bias"
        mapping[stage["pressure_col"]] = f"{key}_pressure"
        for gas_col in stage["gas_cols"]:
            mapping[gas_col] = gas_col.lower()
    return mapping


def _app_key_for_param(param_col: str, stage_defs: list) -> str:
    """PARAMETER_COLUMNS의 실제 컬럼명을 화면 입력 dict의 key로 변환 (stage_defs 기반)."""
    return _column_app_key_map(stage_defs)[param_col]


def _recipe_from_inputs(inputs: dict, param_columns: list, stage_defs: list) -> dict:
    key_map = _column_app_key_map(stage_defs)
    return {col: float(inputs.get(key_map[col], 0.0)) for col in param_columns}


def format_parameter_label(param_col: str, stage_defs: list) -> str:
    """모델 파라미터 컬럼명을 Stage·항목·단위가 보이는 UI 라벨로 바꾼다."""
    for stage in stage_defs:
        stage_key = stage["key"]
        if param_col == stage["time_col"]:
            return f"{stage_key} Etch Time [s]"
        if param_col == stage["bias_col"]:
            return f"{stage_key} RF Bias [W]"
        if param_col == stage["pressure_col"]:
            return f"{stage_key} Pressure [mT]"
        if param_col in stage["gas_cols"]:
            gas = param_col
            prefix = f"{stage_key}_"
            if gas.startswith(prefix):
                gas = gas[len(prefix):]
            if gas.endswith("_sccm"):
                gas = gas[: -len("_sccm")]
            return f"{stage_key} {gas} Flow [sccm]"
    return param_col


# ==============================================================================
# 1. predict() - Recipe 조건 -> 예상 품질 결과 (실제 ML 모델 호출)
# ==============================================================================
def predict(inputs: dict, wafer_summary_df=None, recipe_master_df=None, process: str = "trench",
            allow_out_of_range: bool = False) -> dict:
    """
    Stage별 Time/RF Bias/Pressure/Gas Flow가 담긴 inputs dict를 실제 학습된 모델에 넣어
    Recipe 예상 품질을 반환한다. Equipment/Chamber는 학습 데이터에서 실제 관측된 조합만 허용된다.
    파라미터가 학습 range를 벗어나면 기본적으로 예측을 막는다 (allow_out_of_range=True면 경고만 남기고 실행).
    """
    core = _CORES[process]
    model_dir = _MODEL_DIRS[process]
    equipment = inputs.get("equipment")
    chamber = inputs.get("chamber")

    recipe = _recipe_from_inputs(inputs, core.PARAMETER_COLUMNS, PROCESS_STAGE_DEFS[process])

    try:
        raw = core.predict_wafer(recipe, equipment, chamber, model_dir=model_dir, allow_out_of_range=allow_out_of_range)
    except core.OutOfRangeError as exc:
        return {
            "Top CD": None, "Mid CD": None, "Bottom CD": None, "Depth": None,
            "CD Uniformity": None, "Depth Uniformity": None, "Overall Spec Pass Rate": None,
            "Particle": False, "Defect Count": None, "Particle Probability": None,
            "_source_wafer_count": 0, "_error": str(exc), "_out_of_range": True,
        }
    except ValueError as exc:
        # 학습 데이터에 없는 Equipment/Chamber 조합 등 - 화면에 그대로 노출할 수 있도록 반환
        return {
            "Top CD": None, "Mid CD": None, "Bottom CD": None, "Depth": None,
            "CD Uniformity": None, "Depth Uniformity": None, "Overall Spec Pass Rate": None,
            "Particle": False, "Defect Count": None, "Particle Probability": None,
            "_source_wafer_count": 0, "_error": str(exc),
        }

    wm = raw["wafer_metrics"]
    depth_target = _DEPTH_TARGET[process]
    depth_factor = _DEPTH_TO_NM[process]
    cd_uniformity = (
        wm["Top_CD_nm"]["uniformity_pct"] + wm["Mid_CD_nm"]["uniformity_pct"] + wm["Bottom_CD_nm"]["uniformity_pct"]
    ) / 3.0

    return {
        "Top CD": round(float(wm["Top_CD_nm"]["mean"]), 1),
        "Mid CD": round(float(wm["Mid_CD_nm"]["mean"]), 1),
        "Bottom CD": round(float(wm["Bottom_CD_nm"]["mean"]), 1),
        "Depth": round(float(wm[depth_target]["mean"]) * depth_factor, 1),
        "CD Uniformity": round(float(cd_uniformity), 2),
        "Depth Uniformity": round(float(wm[depth_target]["uniformity_pct"]), 2),
        "Overall Spec Pass Rate": round(float(raw["predicted_overall_spec_pass_rate_pct"]), 1),
        "Particle": bool(raw["predicted_any_particle_probability_pct_independence_approx"] >= 50),
        "Defect Count": round(float(raw["predicted_total_defect_count"]), 1),
        "Particle Probability": round(float(raw["predicted_any_particle_probability_pct_independence_approx"]), 1),
        "_source_wafer_count": 25,
        "_quality_index_v3_pct": round(float(raw["predicted_quality_index_pct"]), 1),
        "_worst_zone": raw["worst_zone"],
        "_warnings": raw["warnings"],
    }


# ==============================================================================
# 2. 종합 품질 점수 (Pass Rate 60% + Uniformity 25% + 목표 근접도 15%)
# 실제 공식은 ml_engine/scoring.py에 있음 (core 엔진의 파라미터 추천도 같은 공식을 써야 해서 공유 모듈로 분리).
# ==============================================================================


# ==============================================================================
# 3. 목표 대비 평가 (Output B)
# ==============================================================================
def score_recipe_versions(wafer_df: pd.DataFrame, targets: dict) -> pd.DataFrame:
    """Wafer_Summary 실측치를 Recipe_Version별로 집계해 종합 품질 점수로 순위를 매긴다.

    모델 예측이 아니라 실제 측정된 Wafer 결과 기반이라, "지금 이 Rev들 중 뭐가 제일 좋았나"를
    바로 보여줄 수 있다 (대시보드의 Rev별 스코어링 표/차트에 사용).
    """
    rows = []
    for recipe_version, group in wafer_df.groupby("Recipe_Version"):
        result = {
            "Top CD": float(group["Top_CD_Mean_nm"].mean()),
            "Mid CD": float(group["Mid_CD_Mean_nm"].mean()),
            "Bottom CD": float(group["Bottom_CD_Mean_nm"].mean()),
            "Depth": float(group["Depth_Mean_nm"].mean()),
            "CD Uniformity": float(
                group[["Top_CD_Uniformity_pct", "Mid_CD_Uniformity_pct", "Bottom_CD_Uniformity_pct"]].mean().mean()
            ),
            "Depth Uniformity": float(group["Depth_Uniformity_pct"].mean()),
            "Overall Spec Pass Rate": float(group["Overall_Spec_Pass_Rate_pct"].mean()),
        }
        score = compute_composite_score(result, targets)
        rows.append({
            "Recipe": recipe_version,
            "종합 점수": score["total"],
            "Wafer 수": int(group["Wafer_ID"].nunique()) if "Wafer_ID" in group.columns else len(group),
            **result,
        })
    scoreboard = pd.DataFrame(rows)
    if scoreboard.empty:
        return scoreboard
    return scoreboard.sort_values("종합 점수", ascending=False).reset_index(drop=True)


def evaluate_against_target(result: dict, targets: dict) -> dict:
    errors = {
        "Top CD": round(result["Top CD"] - targets["target_top_cd"], 1),
        "Mid CD": round(result["Mid CD"] - targets["target_mid_cd"], 1),
        "Bottom CD": round(result["Bottom CD"] - targets["target_bottom_cd"], 1),
        "Depth": round(result["Depth"] - targets["target_depth"], 1),
    }
    satisfied = {
        "CD Uniformity": result["CD Uniformity"] <= targets["max_cd_uniformity"],
        "Depth Uniformity": result["Depth Uniformity"] <= targets["max_depth_uniformity"],
        "Pass Rate": result["Overall Spec Pass Rate"] >= targets["min_pass_rate"],
        "Defect Count": result["Defect Count"] <= targets["max_defect_count"],
    }
    score = compute_composite_score(result, targets)
    issues = generate_priority_issues(result, targets, errors, satisfied)

    return {"errors": errors, "satisfied": satisfied, "score": score, "issues": issues}


def generate_priority_issues(result: dict, targets: dict, errors: dict, satisfied: dict) -> list:
    """미달 항목을 심각도(목표 대비 편차 비율) 순으로 정렬해 우선순위 문장 생성"""
    candidates = []

    for name, error in errors.items():
        target_key = {"Top CD": "target_top_cd", "Mid CD": "target_mid_cd",
                      "Bottom CD": "target_bottom_cd", "Depth": "target_depth"}[name]
        target_value = targets[target_key]
        severity = abs(error) / abs(target_value) if target_value else (1.0 if error != 0 else 0.0)
        if severity > 0.01:  # 1% 이상 벗어난 경우만 이슈로 취급
            direction = "큼" if error > 0 else "부족"
            unit = "nm"
            candidates.append((severity, f"{name}가 목표보다 {abs(error):.1f}{unit} {direction}"))

    if not satisfied["CD Uniformity"]:
        gap = result["CD Uniformity"] - targets["max_cd_uniformity"]
        severity = gap / targets["max_cd_uniformity"] if targets["max_cd_uniformity"] else 0
        candidates.append((severity, f"CD Uniformity가 기준({targets['max_cd_uniformity']}%)보다 {gap:.2f}%p 초과"))

    if not satisfied["Depth Uniformity"]:
        gap = result["Depth Uniformity"] - targets["max_depth_uniformity"]
        severity = gap / targets["max_depth_uniformity"] if targets["max_depth_uniformity"] else 0
        candidates.append((severity, f"Depth Uniformity가 기준({targets['max_depth_uniformity']}%)보다 {gap:.2f}%p 초과"))

    if not satisfied["Pass Rate"]:
        gap = targets["min_pass_rate"] - result["Overall Spec Pass Rate"]
        severity = gap / targets["min_pass_rate"] if targets["min_pass_rate"] else 0
        candidates.append((severity, f"Pass Rate가 목표보다 {gap:.1f}%p 낮음"))

    if not satisfied["Defect Count"]:
        gap = result["Defect Count"] - targets["max_defect_count"]
        severity = gap / targets["max_defect_count"] if targets["max_defect_count"] else 0
        candidates.append((severity, f"평균 Defect Count가 기준({targets['max_defect_count']}건)보다 {gap:.1f}건 초과"))

    candidates.sort(key=lambda c: c[0], reverse=True)
    return [msg for _, msg in candidates[:5]]


# ==============================================================================
# 4. 최적 Recipe 추천 (Output C) + 비교 (Output D)
# ==============================================================================
def recommend_best_recipe(current_recipe: str, inputs: dict, targets: dict,
                           wafer_summary_df, recipe_master_df, process: str = "trench") -> dict:
    """현재 조건(Equipment/Chamber)에서 알려진 Recipe들을 각각 '그 Recipe 그대로' 실제 모델로 평가해
    종합 품질 점수가 가장 높은 Recipe를 추천한다."""
    if recipe_master_df is None or recipe_master_df.empty:
        return None

    stage_defs = PROCESS_STAGE_DEFS[process]
    candidates = []
    for recipe in recipe_master_df["Recipe_Version"]:
        if recipe == current_recipe:
            continue
        cand_inputs = {"equipment": inputs.get("equipment"), "chamber": inputs.get("chamber"), "recipe": recipe}
        cand_inputs.update(stage_inputs_from_recipe(recipe_master_df, recipe, stage_defs))

        result = predict(cand_inputs, wafer_summary_df, recipe_master_df, process=process)
        if result.get("_error"):
            continue
        score = compute_composite_score(result, targets)
        candidates.append({"recipe": recipe, "inputs": cand_inputs, "result": result, "score": score})

    if not candidates:
        return None

    candidates.sort(key=lambda c: c["score"]["total"], reverse=True)
    return candidates[0]


def generate_dashboard_analysis(filtered_wafer_df, zone_summary) -> list:
    """선택한 Equipment/Chamber/Recipe 조건에 대한 Rule-Base 분석 문장 생성 (Process Dashboard 탭)"""
    if filtered_wafer_df is None or filtered_wafer_df.empty:
        return ["선택한 조건에 해당하는 데이터가 없습니다."]

    summary_messages = []
    detail_messages = []

    total_wafers = len(filtered_wafer_df)
    defect_wafers = int((filtered_wafer_df["Total_Defect_Count"] > PARTICLE_DEFECT_THRESHOLD).sum())
    if defect_wafers > 0:
        defect_wafer_pct = defect_wafers / total_wafers * 100
        summary_messages.append(
            f"전체 Wafer {total_wafers}장 중 {defect_wafers}장({defect_wafer_pct:.1f}%)에서 "
            f"Defect가 허용 기준 {PARTICLE_DEFECT_THRESHOLD}건을 초과했습니다."
        )
    else:
        summary_messages.append(f"선택한 조건에서 Defect가 허용 기준 {PARTICLE_DEFECT_THRESHOLD}건을 초과한 Wafer는 없었습니다.")

    avg_pass_rate = filtered_wafer_df["Overall_Spec_Pass_Rate_pct"].mean()
    if avg_pass_rate >= 95:
        summary_messages.append("현재 Recipe는 안정적인 공정으로 판단됩니다.")
    elif avg_pass_rate >= 90:
        summary_messages.append("현재 Recipe는 대체로 안정적이나 일부 개선 여지가 있습니다.")
    else:
        summary_messages.append(f"현재 Recipe는 평균 Pass Rate {avg_pass_rate:.1f}%로 공정 안정성이 낮아 조건 재검토가 필요합니다.")

    if zone_summary is not None and not zone_summary.empty and "Zone" in zone_summary.columns:
        zones = zone_summary.set_index("Zone")

        if "Center" in zones.index and "Extreme Edge" in zones.index:
            bottom_gap = zones.loc["Center", "Bottom CD"] - zones.loc["Extreme Edge", "Bottom CD"]
            if bottom_gap > 10:
                detail_messages.append(
                    f"Extreme Edge Zone의 Bottom CD가 Center 대비 {bottom_gap:.1f}nm 낮아 Edge 테이퍼링이 심화되는 경향이 있습니다."
                )

        if "Pass Rate" in zones.columns and not zones["Pass Rate"].empty:
            worst_zone = zones["Pass Rate"].idxmin()
            worst_value = zones.loc[worst_zone, "Pass Rate"]
            if worst_value < 90:
                detail_messages.append(f"{worst_zone} Zone의 Pass Rate({worst_value:.1f}%)가 가장 낮게 나타납니다.")

        if "Defect Rate" in zones.columns and not zones["Defect Rate"].empty:
            worst_defect_zone = zones["Defect Rate"].idxmax()
            worst_defect_value = zones.loc[worst_defect_zone, "Defect Rate"]
            if worst_defect_value > 0:
                worst_defect_count = int(zones.loc[worst_defect_zone, "Defect Count"])
                detail_messages.append(
                    f"{worst_defect_zone} Zone의 Defect 발생 비율이 {worst_defect_value:.1f}%로 가장 높으며, "
                    f"총 {worst_defect_count}건의 Defect가 집계되었습니다."
                )

    return summary_messages + detail_messages


def recommend_parameter_adjustments(inputs: dict, targets: dict, process: str = "trench", top_n: int = 5,
                                     allow_out_of_range: bool = False) -> dict:
    """현재 화면에 입력된 (기존 Recipe가 아닌) Custom 값을 출발점으로,
    파라미터를 하나씩 바꿔가며(One-Factor-at-a-Time) targets(목표 품질) 기준 종합 점수가
    가장 좋아지는 방향을 찾는다. Recipe 단위 추천(recommend_best_recipe)과 달리 파라미터 단위 제안이다."""
    core = _CORES[process]
    model_dir = _MODEL_DIRS[process]
    equipment = inputs.get("equipment")
    chamber = inputs.get("chamber")
    recipe = _recipe_from_inputs(inputs, core.PARAMETER_COLUMNS, PROCESS_STAGE_DEFS[process])
    try:
        return core.recommend_parameter_changes(
            recipe, equipment, chamber, targets, model_dir=model_dir, top_n=top_n,
            allow_out_of_range=allow_out_of_range,
        )
    except ValueError as exc:
        return {"error": str(exc)}


def generate_recommendation_reason(suggestion: dict, candidate: dict, targets: dict) -> str:
    """기존 점수 함수로 예측 전후를 비교해 추천 이유를 짧게 설명한다.

    추천 순위나 품질 공식은 변경하지 않고, 엔진이 이미 계산한 후보별 예측값을 설명에만 사용한다.
    """
    cd_signals = [
        (
            "Top CD",
            _proximity_score(candidate["predicted_top_cd"], targets["target_top_cd"])
            - _proximity_score(suggestion["baseline_top_cd"], targets["target_top_cd"]),
        ),
        (
            "Mid CD",
            _proximity_score(candidate["predicted_mid_cd"], targets["target_mid_cd"])
            - _proximity_score(suggestion["baseline_mid_cd"], targets["target_mid_cd"]),
        ),
        (
            "Bottom CD",
            _proximity_score(candidate["predicted_bottom_cd"], targets["target_bottom_cd"])
            - _proximity_score(suggestion["baseline_bottom_cd"], targets["target_bottom_cd"]),
        ),
    ]
    depth_gain = (
        _proximity_score(candidate["predicted_depth"], targets["target_depth"])
        - _proximity_score(suggestion["baseline_depth"], targets["target_depth"])
    )
    cd_uniformity_gain = (
        _uniformity_score(candidate["predicted_cd_uniformity_pct"], targets["max_cd_uniformity"])
        - _uniformity_score(suggestion["baseline_cd_uniformity_pct"], targets["max_cd_uniformity"])
    )
    depth_uniformity_gain = (
        _uniformity_score(candidate["predicted_depth_uniformity_pct"], targets["max_depth_uniformity"])
        - _uniformity_score(suggestion["baseline_depth_uniformity_pct"], targets["max_depth_uniformity"])
    )
    pass_rate_gain = candidate["predicted_overall_spec_pass_rate_change_pct_point"]
    spec_probability_gain = candidate["predicted_mean_spec_pass_probability_change_pct_point"]
    defect_count_gain = -candidate["predicted_total_defect_count_change"]
    defect_severity_gain = -candidate["predicted_mean_defect_severity_change"]

    score_epsilon = 0.5
    defect_epsilon = 0.05
    best_cd_name, best_cd_gain = max(cd_signals, key=lambda signal: signal[1])
    close_cd_names = [
        name for name, gain in cd_signals
        if gain > score_epsilon and gain >= best_cd_gain * 0.85
    ]

    primary_signals = []
    if best_cd_gain > score_epsilon:
        cd_label = "목표 CD 오차 감소" if len(close_cd_names) >= 2 else f"{best_cd_name} 목표 오차 감소"
        primary_signals.append((best_cd_gain, cd_label))
    if depth_gain > score_epsilon:
        primary_signals.append((depth_gain, "Depth 목표 오차 감소"))
    if cd_uniformity_gain > score_epsilon and depth_uniformity_gain > score_epsilon:
        primary_signals.append((max(cd_uniformity_gain, depth_uniformity_gain), "Uniformity 개선"))
    elif cd_uniformity_gain > score_epsilon:
        primary_signals.append((cd_uniformity_gain, "CD Uniformity 개선"))
    elif depth_uniformity_gain > score_epsilon:
        primary_signals.append((depth_uniformity_gain, "Depth Uniformity 개선"))
    if pass_rate_gain > score_epsilon:
        primary_signals.append((pass_rate_gain, "Pass Rate 개선"))
    if spec_probability_gain > score_epsilon:
        primary_signals.append((spec_probability_gain, "Spec Pass 예측확률 상승"))

    secondary_signals = []
    if defect_count_gain > defect_epsilon:
        secondary_signals.append((defect_count_gain, "Defect Count 감소"))
    if defect_severity_gain > defect_epsilon:
        secondary_signals.append((defect_severity_gain, "Defect Severity 감소"))

    if not primary_signals:
        secondary_signals.sort(key=lambda signal: signal[0], reverse=True)
        return secondary_signals[0][1] if secondary_signals else "목표점수 소폭 개선"

    primary_signals.sort(key=lambda signal: signal[0], reverse=True)
    reasons = [primary_signals[0][1]]
    remaining = sorted(primary_signals[1:] + secondary_signals, key=lambda signal: signal[0], reverse=True)
    for _, label in remaining:
        if label not in reasons:
            reasons.append(label)
            break
    return " · ".join(reasons)


def apply_recommended_changes(baseline_inputs: dict, recommendations: list, stage_defs: list) -> tuple[dict, set]:
    """기준 입력값에 추천 후보들을 함께 적용해 조합 Recipe 입력값을 만든다."""
    applied = dict(baseline_inputs)
    changed = set()
    for recommendation in recommendations:
        parameter = recommendation["parameter"]
        applied[_app_key_for_param(parameter, stage_defs)] = recommendation["proposed"]
        changed.add(parameter)
    return applied, changed


def build_combined_recipe_table(current_inputs: dict, applied_inputs: dict, stage_defs: list) -> list:
    """변경 여부와 관계없이 조합 Recipe의 전체 Stage 파라미터를 표 데이터로 만든다."""
    rows = []
    for stage in stage_defs:
        raw_columns = [stage["time_col"], stage["bias_col"], stage["pressure_col"], *stage["gas_cols"]]
        for raw_column in raw_columns:
            app_key = _app_key_for_param(raw_column, stage_defs)
            current_value = current_inputs.get(app_key)
            applied_value = applied_inputs.get(app_key)
            if current_value is None or applied_value is None:
                continue
            rows.append({
                "Stage": stage["key"],
                "Parameter": format_parameter_label(raw_column, stage_defs),
                "기준 Recipe": float(current_value),
                "추천 적용값": float(applied_value),
            })
    return rows


def build_stage_diff_table(current_inputs: dict, recommended_inputs: dict, recipe_master_df, stage_defs=None) -> list:
    """현재 입력 조건 대비 추천 Recipe의 Stage별 파라미터 변경점만 표로 정리"""
    from data_utils import STAGE_DEFS as _DEFAULT_STAGE_DEFS
    stage_defs = stage_defs or _DEFAULT_STAGE_DEFS
    field_labels = {"time": "Time", "rf_bias": "RF Bias", "pressure": "Pressure"}
    rows = []
    for stage in stage_defs:
        key = stage["key"].lower()
        for field, unit in [("time", "s"), ("rf_bias", "W"), ("pressure", "mT")]:
            cur = current_inputs.get(f"{key}_{field}")
            rec = recommended_inputs.get(f"{key}_{field}")
            if cur is None or rec is None:
                continue
            if abs(cur - rec) < 1e-6:
                continue
            rows.append({
                "Parameter": f"{stage['key']} {field_labels[field]}",
                "현재값": f"{cur:g}{unit}",
                "추천값": f"{rec:g}{unit}",
                "변경": f"{'+' if rec - cur > 0 else ''}{rec - cur:g}{unit}",
            })
    return rows


# ==============================================================================
# 5. 신규 공정 품질 프리뷰 (유사도 기반, 편법) — 정식 예측이 아니라 실측 참고값 조회
# ==============================================================================
def find_layer_references(layer_materials: list[str]) -> dict:
    """아직 학습된 모델이 없는 신규 공정을, 순서대로 입력한 레이어(물질) 목록으로 정의하면
    두 가지 참고자료를 편법으로 만들어 보여준다.

    1) 레이어별 참고 Recipe 조건 — 요청한 물질과 같은 Stage를 4개 기존 공정 전체에서 찾아,
       그 Stage가 실제 관측된 파라미터 범위(Time/RF Bias/Pressure/Gas, 16개 Recipe 기준 min~max)를 보여줌.
    2) 종합 참고 품질 — 요청한 레이어 구성과 Stage 물질 구성이 가장 많이 겹치는 기존 공정 하나를
       골라, 그 공정의 실측 평균 최종 CD/Depth/Uniformity/Pass Rate를 보여줌.

    AI로 새 공정 전체를 예측하는 게 아니다. 레이어 단위 관측 범위와, 가장 비슷한 기존 공정의
    실측 평균만 보여주는 참고용 편법 프리뷰다 (정식 모델링은 그 공정만의 데이터·검증이 필요).
    """
    from data_utils import load_bundled_workbook

    query_materials = [m.strip().lower() for m in layer_materials if m.strip()]
    if not query_materials:
        return {"layer_matches": [], "overall_reference": None}

    recipe_cache: dict[str, "pd.DataFrame"] = {}

    def _recipe(process: str):
        if process not in recipe_cache:
            recipe_cache[process] = load_bundled_workbook(process)["Recipe_Master"]
        return recipe_cache[process]

    layer_matches = []
    for material in query_materials:
        candidates = []
        for process, stage_defs in PROCESS_STAGE_DEFS.items():
            recipe_df = _recipe(process)
            for stage in stage_defs:
                if stage.get("material", "").strip().lower() != material:
                    continue
                param_ranges = {}
                for label, col in [
                    ("Time (s)", stage["time_col"]),
                    ("RF Bias (W)", stage["bias_col"]),
                    ("Pressure (mT)", stage["pressure_col"]),
                ]:
                    param_ranges[label] = (round(float(recipe_df[col].min()), 2), round(float(recipe_df[col].max()), 2))
                gas_ranges = {}
                for gas_col in stage["gas_cols"]:
                    prefix = f"{stage['key']}_"
                    gas_name = gas_col[len(prefix):] if gas_col.startswith(prefix) else gas_col
                    gas_name = gas_name[: -len("_sccm")] if gas_name.endswith("_sccm") else gas_name
                    gas_ranges[f"{gas_name} (sccm)"] = (
                        round(float(recipe_df[gas_col].min()), 2), round(float(recipe_df[gas_col].max()), 2)
                    )
                candidates.append({
                    "process": process,
                    "process_label": PROCESS_LABELS[process],
                    "stage_label": stage["label"],
                    "param_ranges": param_ranges,
                    "gas_ranges": gas_ranges,
                })
        layer_matches.append({"material": material, "matches": candidates})

    overall_reference = None
    best_overlap = 0
    for process, stage_defs in PROCESS_STAGE_DEFS.items():
        # 매칭은 소문자 기준으로 하되, 화면 표시용으로는 원래 대소문자(PROCESS_STAGE_DEFS 표기)를 남긴다.
        material_display_by_lower = {}
        for s in stage_defs:
            material = s.get("material", "").strip()
            if material:
                material_display_by_lower.setdefault(material.lower(), material)
        overlap = len(set(query_materials) & material_display_by_lower.keys())
        if overlap > best_overlap:
            best_overlap = overlap
            wafer_df = load_bundled_workbook(process)["Wafer_Summary"]
            overall_reference = {
                "process": process,
                "process_label": PROCESS_LABELS[process],
                "process_materials": sorted(material_display_by_lower.values()),
                "overlap_count": overlap,
                "requested_count": len(query_materials),
                "wafer_count": int(wafer_df["Wafer_ID"].nunique()),
                "avg_top_cd_nm": round(float(wafer_df["Top_CD_Mean_nm"].mean()), 1),
                "avg_mid_cd_nm": round(float(wafer_df["Mid_CD_Mean_nm"].mean()), 1),
                "avg_bottom_cd_nm": round(float(wafer_df["Bottom_CD_Mean_nm"].mean()), 1),
                "avg_depth_nm": round(float(wafer_df["Depth_Mean_nm"].mean()), 1),
                "avg_uniformity_pct": round(float(
                    wafer_df[["Top_CD_Uniformity_pct", "Mid_CD_Uniformity_pct", "Bottom_CD_Uniformity_pct"]]
                    .mean().mean()
                ), 2),
                "avg_pass_rate_pct": round(float(wafer_df["Overall_Spec_Pass_Rate_pct"].mean()), 1),
            }

    return {"layer_matches": layer_matches, "overall_reference": overall_reference}
