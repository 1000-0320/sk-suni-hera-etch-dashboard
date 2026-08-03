"""
Etch AI Decision Support System - 예측/평가/추천 모델 모듈

※ 아직 Random Forest / XGBoost 등 머신러닝 모델을 사용하지 않는다.
   predict()는 업로드된 Wafer_Summary(실측 이력)에서 선택한 Recipe의 평균값을
   "베이스라인"으로 조회하고, 사용자가 입력한 Stage(S1~S4)별 Time/RF Bias/Pressure가
   그 Recipe의 실제 값과 얼마나 다른지에 따라 소폭 보정하는 "실이력 기반 더미" 함수다.

   나중에 실제 모델로 교체할 때는 predict() 내부만 아래처럼 바꾸면 된다.

   import joblib
   _model = joblib.load("rf_model.pkl")

   def predict(inputs, wafer_summary_df=None, recipe_master_df=None):
       X = pd.DataFrame([inputs])
       y = _model.predict(X)[0]
       return {"Top CD": y[0], ...}

   compute_composite_score() / evaluate_against_target() / recommend_best_recipe()는
   predict()가 반환하는 dict 형태만 유지되면 모델 교체 후에도 그대로 재사용 가능하다.
"""

from data_utils import PARTICLE_DEFECT_THRESHOLD, STAGE_DEFS, stage_inputs_from_recipe


# ==============================================================================
# 1. predict() - Recipe 조건 -> 예상 품질 결과
# ==============================================================================
def _predict_formula_only(inputs: dict) -> dict:
    """실측 이력 데이터가 없을 때 쓰는 순수 공식 기반 폴백 (더미 로직)"""
    rf_bias = inputs.get("s4_rf_bias", 180)
    pressure = inputs.get("s4_pressure", 30)
    time = inputs.get("s4_time", 75)

    top_cd = 255.0 - (pressure - 30) * 0.02
    mid_cd = 253.0
    bottom_cd = 245.0 - (rf_bias - 180) * 0.05
    depth = 1150.0 + (time - 75) * 2

    cd_uniformity = max(0.5, 2.0)
    depth_uniformity = max(0.5, 8.0)
    pass_rate = min(100.0, max(0.0, 80.0 - abs(rf_bias - 180) * 0.1))

    return {
        "Top CD": round(top_cd, 1), "Mid CD": round(mid_cd, 1), "Bottom CD": round(bottom_cd, 1),
        "Depth": round(depth, 1),
        "CD Uniformity": round(cd_uniformity, 2), "Depth Uniformity": round(depth_uniformity, 2),
        "Overall Spec Pass Rate": round(pass_rate, 1),
        "Particle": False, "Defect Count": 0.0, "Particle Probability": 0.0,
        "_source_wafer_count": 0,
    }


# Stage별 조건이 어떤 출력 지표에 주로 영향을 주는지에 대한 단순화된 가정(더미).
# 실제 공정 물리를 정밀 모델링한 것이 아니라, ML 도입 전까지 쓰는 근사 민감도다.
#   S1(SiON Strip)  -> Top CD 형성에 주로 영향
#   S2(SOC Open)    -> Mid CD 형성에 주로 영향
#   S3(SiO2 HM)     -> Top/Mid CD 경계(테이퍼)에 영향
#   S4(Si Main)     -> Bottom CD / Depth에 주로 영향 (가장 큰 영향)
def _stage_delta(inputs: dict, recipe_row, stage_key: str, field: str, ref_col: str) -> float:
    input_key = f"{stage_key.lower()}_{field}"
    if input_key not in inputs or recipe_row is None:
        return 0.0
    return float(inputs[input_key]) - float(recipe_row[ref_col])


def predict(inputs: dict, wafer_summary_df=None, recipe_master_df=None) -> dict:
    """
    공정 조건(dict, Stage별 Time/RF Bias/Pressure 포함) + 실측 이력(Wafer_Summary)을 이용한 예측 함수.

    1) 선택한 Recipe(+Equipment/Chamber)의 과거 Wafer_Summary 평균을 베이스라인으로 조회
    2) 사용자가 입력한 Stage별 조건이 그 Recipe의 실제 값과 얼마나 다른지에 비례해
       베이스라인을 소폭 보정 (What-if 튜닝 효과)
    """
    recipe = inputs.get("recipe")

    if wafer_summary_df is None or wafer_summary_df.empty or not recipe:
        return _predict_formula_only(inputs)

    subset = wafer_summary_df[wafer_summary_df["Recipe_Version"] == recipe]
    narrowed = subset[
        (subset["Equipment_Model"] == inputs.get("equipment"))
        & (subset["Chamber_ID"] == inputs.get("chamber"))
    ]
    if len(narrowed) >= 3:
        subset = narrowed

    if subset.empty:
        return _predict_formula_only(inputs)

    baseline_top = subset["Top_CD_Mean_nm"].mean()
    baseline_mid = subset["Mid_CD_Mean_nm"].mean()
    baseline_bottom = subset["Bottom_CD_Mean_nm"].mean()
    baseline_depth = subset["Depth_Mean_nm"].mean()
    baseline_cd_uniformity = subset[
        ["Top_CD_Uniformity_pct", "Mid_CD_Uniformity_pct", "Bottom_CD_Uniformity_pct"]
    ].mean().mean()
    baseline_depth_uniformity = subset["Depth_Uniformity_pct"].mean()
    baseline_pass_rate = subset["Overall_Spec_Pass_Rate_pct"].mean()
    baseline_defect_count = subset["Total_Defect_Count"].mean()
    particle_ratio = (subset["Total_Defect_Count"] > PARTICLE_DEFECT_THRESHOLD).mean()

    recipe_row = None
    if recipe_master_df is not None and not recipe_master_df.empty:
        match = recipe_master_df[recipe_master_df["Recipe_Version"] == recipe]
        if not match.empty:
            recipe_row = match.iloc[0]

    d_s1_bias = _stage_delta(inputs, recipe_row, "S1", "rf_bias", "S1_RF_Bias_W")
    d_s1_pressure = _stage_delta(inputs, recipe_row, "S1", "pressure", "S1_Pressure_mT")
    d_s2_pressure = _stage_delta(inputs, recipe_row, "S2", "pressure", "S2_Pressure_mT")
    d_s3_bias = _stage_delta(inputs, recipe_row, "S3", "rf_bias", "S3_RF_Bias_W")
    d_s4_bias = _stage_delta(inputs, recipe_row, "S4", "rf_bias", "S4_RF_Bias_W")
    d_s4_pressure = _stage_delta(inputs, recipe_row, "S4", "pressure", "S4_Pressure_mT")
    d_s4_time = _stage_delta(inputs, recipe_row, "S4", "time", "S4_Time_s")

    top_cd = baseline_top - d_s1_pressure * 0.05 + d_s1_bias * 0.01
    mid_cd = baseline_mid - d_s2_pressure * 0.04 - d_s3_bias * 0.01
    bottom_cd = baseline_bottom - d_s4_bias * 0.08
    depth = baseline_depth + d_s4_time * 1.5 - d_s4_pressure * 0.8

    total_abs_delta = abs(d_s1_bias) + abs(d_s1_pressure) + abs(d_s2_pressure) + abs(d_s3_bias) + abs(d_s4_bias)
    cd_uniformity = max(0.3, baseline_cd_uniformity + total_abs_delta * 0.003)
    depth_uniformity = max(0.3, baseline_depth_uniformity + abs(d_s4_time) * 0.02 + abs(d_s4_pressure) * 0.03)
    pass_rate = min(100.0, max(0.0, baseline_pass_rate - abs(d_s4_bias) * 0.02 - abs(d_s4_pressure) * 0.05))
    particle = particle_ratio >= 0.5

    return {
        "Top CD": round(float(top_cd), 1),
        "Mid CD": round(float(mid_cd), 1),
        "Bottom CD": round(float(bottom_cd), 1),
        "Depth": round(float(depth), 1),
        "CD Uniformity": round(float(cd_uniformity), 2),
        "Depth Uniformity": round(float(depth_uniformity), 2),
        "Overall Spec Pass Rate": round(float(pass_rate), 1),
        "Particle": bool(particle),
        "Defect Count": round(float(baseline_defect_count), 1),
        "Particle Probability": round(float(particle_ratio * 100), 1),
        "_source_wafer_count": int(len(subset)),
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
                           wafer_summary_df, recipe_master_df) -> dict:
    """현재 조건(Equipment/Chamber)에서 알려진 Recipe들을 각각 '그 Recipe 그대로' 평가해
    종합 품질 점수가 가장 높은 Recipe를 추천한다 (아직 ML 최적화가 아닌, 기존 Recipe 중 탐색)."""
    if recipe_master_df is None or recipe_master_df.empty:
        return None

    candidates = []
    for recipe in recipe_master_df["Recipe_Version"]:
        if recipe == current_recipe:
            continue
        cand_inputs = {"equipment": inputs.get("equipment"), "chamber": inputs.get("chamber"), "recipe": recipe}
        cand_inputs.update(stage_inputs_from_recipe(recipe_master_df, recipe))  # 보정 없이 그 Recipe 고유값으로 평가

        result = predict(cand_inputs, wafer_summary_df, recipe_master_df)
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


def build_stage_diff_table(current_inputs: dict, recommended_inputs: dict, recipe_master_df) -> list:
    """현재 입력 조건 대비 추천 Recipe의 Stage별 파라미터 변경점만 표로 정리"""
    rows = []
    for stage in STAGE_DEFS:
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
