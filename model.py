"""
Etch AI Decision Support System - 예측/평가/추천 모델 모듈

predict()는 이제 실측 이력 기반 더미가 아니라, 학습된 RandomForest/XGBoost 모델
(ml_engine/isolation_core.py, ml_engine/trench_core.py)을 호출해 실제 예측을 수행한다.
공정(isolation/trench)에 따라 파라미터 개수·Depth 단위가 다르므로 어댑터에서 흡수한다.

compute_composite_score() / evaluate_against_target() / recommend_best_recipe()는
predict()가 반환하는 dict 형태만 유지되면 그대로 재사용된다 (모델 교체와 무관).
"""

from ml_engine import isolation_core, trench_core
from data_utils import PARTICLE_DEFECT_THRESHOLD, PROCESS_STAGE_DEFS, stage_inputs_from_recipe

_CORES = {"isolation": isolation_core, "trench": trench_core}
_MODEL_DIRS = {
    "isolation": "ml_engine/isolation_models",
    "trench": "ml_engine/trench_models",
}
# 각 공정 엔진이 내부적으로 Depth를 어떤 컬럼명/단위로 예측하는지 (isolation은 학습 데이터 원본이 Angstrom)
_DEPTH_TARGET = {"isolation": "Depth_A", "trench": "Depth_nm"}
_DEPTH_TO_NM = {"isolation": 0.1, "trench": 1.0}


# ==============================================================================
# 0. 입력 dict(Stage별 UI 값) -> 모델 파라미터 dict 변환
# ==============================================================================
def _app_key_for_param(param_col: str) -> str:
    """PARAMETER_COLUMNS의 실제 컬럼명(예: S1_RF_Bias_W)을 화면 입력 dict의 key(s1_rf_bias)로 변환."""
    stage = param_col.split("_")[0].lower()
    if param_col.endswith("_Time_s"):
        return f"{stage}_time"
    if param_col.endswith("_RF_Bias_W"):
        return f"{stage}_rf_bias"
    if param_col.endswith("_Pressure_mT"):
        return f"{stage}_pressure"
    return param_col.lower()  # Gas Flow 컬럼은 그대로 소문자 매칭 (예: S1_CHF3_sccm -> s1_chf3_sccm)


def _recipe_from_inputs(inputs: dict, param_columns: list) -> dict:
    return {col: float(inputs.get(_app_key_for_param(col), 0.0)) for col in param_columns}


# ==============================================================================
# 1. predict() - Recipe 조건 -> 예상 품질 결과 (실제 ML 모델 호출)
# ==============================================================================
def predict(inputs: dict, wafer_summary_df=None, recipe_master_df=None, process: str = "trench") -> dict:
    """
    Stage별 Time/RF Bias/Pressure/Gas Flow가 담긴 inputs dict를 실제 학습된 모델에 넣어
    Recipe 예상 품질을 반환한다. Equipment/Chamber는 학습 데이터에서 실제 관측된 조합만 허용된다.
    """
    core = _CORES[process]
    model_dir = _MODEL_DIRS[process]
    equipment = inputs.get("equipment")
    chamber = inputs.get("chamber")

    recipe = _recipe_from_inputs(inputs, core.PARAMETER_COLUMNS)

    try:
        raw = core.predict_wafer(recipe, equipment, chamber, model_dir=model_dir)
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
# ==============================================================================
def _clip(value, lo=0.0, hi=100.0):
    return max(lo, min(hi, value))


def _uniformity_score(value: float, max_allowed: float) -> float:
    """0이면 100점, max_allowed 이상이면 0점 (선형)"""
    if max_allowed <= 0:
        return 0.0
    return _clip(100 * (1 - value / max_allowed))


def _proximity_score(predicted: float, target: float, tolerance_pct: float = 0.05) -> float:
    """target과 오차 0이면 100점, 오차가 target의 tolerance_pct 이상이면 0점 (선형)"""
    if target == 0:
        return 100.0 if predicted == 0 else 0.0
    tolerance = abs(target) * tolerance_pct
    error = abs(predicted - target)
    return _clip(100 * (1 - error / tolerance))


def compute_composite_score(result: dict, targets: dict) -> dict:
    """종합 품질 점수 = Pass Rate 60% + Uniformity 25% + 목표 근접도 15%"""
    pass_rate_score = _clip(result["Overall Spec Pass Rate"])

    cd_u_score = _uniformity_score(result["CD Uniformity"], targets["max_cd_uniformity"])
    depth_u_score = _uniformity_score(result["Depth Uniformity"], targets["max_depth_uniformity"])
    uniformity_score = (cd_u_score + depth_u_score) / 2

    proximity_scores = [
        _proximity_score(result["Top CD"], targets["target_top_cd"]),
        _proximity_score(result["Mid CD"], targets["target_mid_cd"]),
        _proximity_score(result["Bottom CD"], targets["target_bottom_cd"]),
        _proximity_score(result["Depth"], targets["target_depth"]),
    ]
    target_proximity_score = sum(proximity_scores) / len(proximity_scores)

    total = pass_rate_score * 0.60 + uniformity_score * 0.25 + target_proximity_score * 0.15

    return {
        "total": round(total, 1),
        "pass_rate_score": round(pass_rate_score, 1),
        "uniformity_score": round(uniformity_score, 1),
        "target_proximity_score": round(target_proximity_score, 1),
    }


# ==============================================================================
# 3. 목표 대비 평가 (Output B)
# ==============================================================================
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
        severity = abs(error) / abs(target_value) if target_value else 0
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

    messages = []

    if zone_summary is not None and not zone_summary.empty and "Zone" in zone_summary.columns:
        zones = zone_summary.set_index("Zone")

        if "Center" in zones.index and "Extreme Edge" in zones.index:
            bottom_gap = zones.loc["Center", "Bottom CD"] - zones.loc["Extreme Edge", "Bottom CD"]
            if bottom_gap > 10:
                messages.append(
                    f"Extreme Edge Zone의 Bottom CD가 Center 대비 {bottom_gap:.1f}nm 낮아 Edge 테이퍼링이 심화되는 경향이 있습니다."
                )

        if "Pass Rate" in zones.columns and not zones["Pass Rate"].empty:
            worst_zone = zones["Pass Rate"].idxmin()
            worst_value = zones.loc[worst_zone, "Pass Rate"]
            if worst_value < 90:
                messages.append(f"{worst_zone} Zone의 Pass Rate({worst_value:.1f}%)가 가장 낮게 나타납니다.")

        if "Defect Rate" in zones.columns and not zones["Defect Rate"].empty:
            worst_defect_zone = zones["Defect Rate"].idxmax()
            worst_defect_value = zones.loc[worst_defect_zone, "Defect Rate"]
            if worst_defect_value > 0:
                messages.append(f"{worst_defect_zone} Zone에서 Defect 발생 비율({worst_defect_value:.1f}%)이 가장 높습니다.")

    total_wafers = len(filtered_wafer_df)
    defect_wafers = int((filtered_wafer_df["Total_Defect_Count"] > PARTICLE_DEFECT_THRESHOLD).sum())
    if defect_wafers > 0:
        messages.append(
            f"전체 {total_wafers}장 중 {defect_wafers}장의 Wafer에서 Defect(Particle) {PARTICLE_DEFECT_THRESHOLD}건 초과가 발생했습니다."
        )
    else:
        messages.append(f"선택한 조건에서 Defect(Particle) {PARTICLE_DEFECT_THRESHOLD}건 초과 Wafer는 없었습니다.")

    avg_pass_rate = filtered_wafer_df["Overall_Spec_Pass_Rate_pct"].mean()
    if avg_pass_rate >= 95:
        messages.append("현재 Recipe는 안정적인 공정으로 판단됩니다.")
    elif avg_pass_rate >= 90:
        messages.append("현재 Recipe는 대체로 안정적이나 일부 개선 여지가 있습니다.")
    else:
        messages.append(f"현재 Recipe는 평균 Pass Rate {avg_pass_rate:.1f}%로 공정 안정성이 낮아 조건 재검토가 필요합니다.")

    return messages


def recommend_parameter_adjustments(inputs: dict, process: str = "trench", top_n: int = 5) -> dict:
    """현재 화면에 입력된 (기존 Recipe가 아닌) Custom 값을 출발점으로,
    파라미터를 하나씩 바꿔가며(One-Factor-at-a-Time) 관측된 값들 중 품질이 가장 좋아지는
    방향을 찾는다. Recipe 단위 추천(recommend_best_recipe)과 달리 파라미터 단위 제안이다."""
    core = _CORES[process]
    model_dir = _MODEL_DIRS[process]
    equipment = inputs.get("equipment")
    chamber = inputs.get("chamber")
    recipe = _recipe_from_inputs(inputs, core.PARAMETER_COLUMNS)
    try:
        return core.recommend_parameter_changes(recipe, equipment, chamber, model_dir=model_dir, top_n=top_n)
    except ValueError as exc:
        return {"error": str(exc)}


def build_stage_diff_table(current_inputs: dict, recommended_inputs: dict, recipe_master_df, stage_defs=None) -> list:
    """현재 입력 조건 대비 추천 Recipe의 Stage별 파라미터 변경점만 표로 정리"""
    from data_utils import STAGE_DEFS as _DEFAULT_STAGE_DEFS
    stage_defs = stage_defs or _DEFAULT_STAGE_DEFS
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
                "Parameter": f"{stage['key']} {field.replace('_', ' ').title()}",
                "현재값": f"{cur:g}{unit}",
                "추천값": f"{rec:g}{unit}",
                "변경": f"{'+' if rec - cur > 0 else ''}{rec - cur:g}{unit}",
            })
    return rows
