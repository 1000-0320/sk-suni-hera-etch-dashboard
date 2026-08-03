from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PARAMETER_COLUMNS = [
    "S1_Time_s",
    "S1_CF4_sccm",
    "S1_CHF3_sccm",
    "S1_RF_Bias_W",
    "S1_Pressure_mT",
    "S2_Time_s",
    "S2_O2_sccm",
    "S2_N2_sccm",
    "S2_RF_Bias_W",
    "S2_Pressure_mT",
    "S3_Time_s",
    "S3_CF4_sccm",
    "S3_C4F8_sccm",
    "S3_RF_Bias_W",
    "S3_Pressure_mT",
    "S4_Time_s",
    "S4_HBr_sccm",
    "S4_Cl2_sccm",
    "S4_RF_Bias_W",
    "S4_Pressure_mT",
]

REGRESSION_TARGETS = ["Top_CD_nm", "Mid_CD_nm", "Bottom_CD_nm", "Depth_nm"]
SPEC_PASS_TARGETS = [
    "Top_CD_Spec_Pass",
    "Mid_CD_Spec_Pass",
    "Bottom_CD_Spec_Pass",
    "Depth_Spec_Pass",
]
DEFECT_REGRESSION_TARGETS = ["Defect_Count", "Defect_Severity_Score"]
SPEC_TARGET_TO_MEASUREMENT = {
    "Top_CD_Spec_Pass": "Top_CD_nm",
    "Mid_CD_Spec_Pass": "Mid_CD_nm",
    "Bottom_CD_Spec_Pass": "Bottom_CD_nm",
    "Depth_Spec_Pass": "Depth_nm",
}


def _resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    if model_dir is None:
        return Path(__file__).resolve().parent / "models"
    return Path(model_dir).resolve()


@lru_cache(maxsize=8)
def load_metadata(model_dir: str | Path | None = None) -> dict[str, Any]:
    directory = _resolve_model_dir(model_dir)
    return json.loads((directory / "metadata.json").read_text(encoding="utf-8"))


def valid_tool_pairs(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only equipment/chamber pairs observed in the training wafers."""
    support = metadata.get("tool_support", {})
    pairs = support.get("valid_pairs", [])
    if not pairs:
        raise ValueError(
            "학습 데이터의 Equipment/Chamber 조합 정보가 없습니다. metadata를 다시 생성해야 합니다."
        )
    return [dict(row) for row in pairs]


def validate_tool_pair(
    equipment_model: str, chamber_id: str, metadata: dict[str, Any]
) -> None:
    observed = {
        (str(row["equipment_model"]), str(row["chamber_id"]))
        for row in valid_tool_pairs(metadata)
    }
    if (equipment_model, chamber_id) not in observed:
        allowed = ", ".join(
            f"{equipment}/{chamber}" for equipment, chamber in sorted(observed)
        )
        raise ValueError(
            f"학습 데이터에 없는 Equipment/Chamber 조합입니다: "
            f"{equipment_model}/{chamber_id}. 사용 가능한 조합: {allowed}"
        )


def recipe_tool_support_count(
    recipe_version: str,
    equipment_model: str,
    chamber_id: str,
    metadata: dict[str, Any],
) -> int:
    counts = metadata.get("tool_support", {}).get("recipe_pair_wafer_counts", {})
    return int(
        counts.get(recipe_version, {})
        .get(equipment_model, {})
        .get(chamber_id, 0)
    )


@lru_cache(maxsize=32)
def _load_model(model_path: str) -> Any:
    return joblib.load(model_path)


def parameter_closure_score(
    recipe: dict[str, float],
    metadata: dict[str, Any],
    *,
    clip: bool = True,
) -> dict[str, Any]:
    base = metadata["base_recipe"]
    optimal = metadata["optimal_recipe"]
    components: dict[str, float] = {}
    for parameter in PARAMETER_COLUMNS:
        denominator = abs(float(optimal[parameter]) - float(base[parameter]))
        if denominator <= 1e-12:
            component = 100.0 if abs(float(recipe[parameter]) - float(optimal[parameter])) <= 1e-12 else 0.0
        else:
            component = 100.0 * (
                1.0 - abs(float(recipe[parameter]) - float(optimal[parameter])) / denominator
            )
        components[parameter] = float(np.clip(component, 0.0, 100.0) if clip else component)
    raw_score = float(np.mean(list(components.values())))
    return {"score_pct": round(raw_score, 1), "components_pct": components, "clipped": clip}


def _site_input_frame(
    recipe: dict[str, float],
    equipment_model: str,
    chamber_id: str,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    frame = pd.DataFrame(metadata["site_template"])
    for parameter in PARAMETER_COLUMNS:
        frame[parameter] = float(recipe[parameter])
    frame["Equipment_Model"] = equipment_model
    frame["Chamber_ID"] = chamber_id
    angle_radians = np.deg2rad(frame["Angle_deg"].fillna(0.0).astype(float))
    frame["Angle_sin"] = np.sin(angle_radians)
    frame["Angle_cos"] = np.cos(angle_radians)
    return frame[metadata["feature_columns"]]


def _empirical_spec_pass(values: np.ndarray, limits: dict[str, Any]) -> np.ndarray:
    mask = np.ones(values.shape[0], dtype=bool)
    lower = limits.get("lower")
    upper = limits.get("upper")
    if lower is not None:
        mask &= values >= float(lower)
    if upper is not None:
        mask &= values <= float(upper)
    return mask


def _lower_is_better_score(value: float, base: float, optimal: float) -> float:
    """Return a clipped Base-to-Rev15 directional score for a lower-is-better metric."""
    denominator = float(base) - float(optimal)
    if denominator <= 1e-12:
        return 100.0 if float(value) <= float(optimal) else 0.0
    score = 100.0 * (float(base) - float(value)) / denominator
    return float(np.clip(score, 0.0, 100.0))


def _quality_component_scores(
    spec_pass_rate_pct: float,
    defect_count: float,
    mean_defect_severity: float,
    reference: dict[str, float],
    weights: dict[str, float],
) -> dict[str, float]:
    count_score = _lower_is_better_score(
        defect_count,
        reference["base_total_defect_count"],
        reference["optimal_total_defect_count"],
    )
    severity_score = _lower_is_better_score(
        mean_defect_severity,
        reference["base_mean_defect_severity"],
        reference["optimal_mean_defect_severity"],
    )
    core_quality = (
        float(weights["spec_pass"]) * float(spec_pass_rate_pct)
        + float(weights["defect_count"]) * count_score
        + float(weights["defect_severity"]) * severity_score
    )
    return {
        "spec_pass_score_pct": float(np.clip(spec_pass_rate_pct, 0.0, 100.0)),
        "defect_count_score_pct": count_score,
        "defect_severity_score_pct": severity_score,
        "core_quality_pct": float(np.clip(core_quality, 0.0, 100.0)),
    }


def predict_wafer(
    recipe: dict[str, float],
    equipment_model: str,
    chamber_id: str,
    *,
    process: str = "Trench Etch",
    layer: str = "Dataset scope (unspecified layer)",
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    directory = _resolve_model_dir(model_dir)
    metadata = load_metadata(str(directory))
    validate_tool_pair(equipment_model, chamber_id, metadata)
    frame = _site_input_frame(recipe, equipment_model, chamber_id, metadata)

    site_predictions: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, float]] = {}
    empirical_spec_flags: list[np.ndarray] = []
    for target in REGRESSION_TARGETS:
        selection = metadata["selected_models"][target]
        model = _load_model(str(directory / selection["file"]))
        predicted = np.asarray(model.predict(frame), dtype=float)
        site_predictions[target] = predicted
        target_limits = metadata["empirical_spec_limits"][target]
        passed = _empirical_spec_pass(predicted, target_limits)
        empirical_spec_flags.append(passed)
        mean_value = float(np.mean(predicted))
        std_value = float(np.std(predicted, ddof=1)) if len(predicted) > 1 else 0.0
        summaries[target] = {
            "mean": mean_value,
            "std": std_value,
            "uniformity_pct": 100.0 * std_value / abs(mean_value) if abs(mean_value) > 1e-12 else math.nan,
            "predicted_empirical_spec_pass_rate_pct": 100.0 * float(np.mean(passed)),
        }

    quality_v2 = metadata.get("quality_v2")
    if quality_v2 is None:
        raise ValueError("Quality-v2 models are missing. Run train_quality_v2.py before using this simulator version.")

    direct_spec_probabilities: dict[str, np.ndarray] = {}
    direct_spec_flags: dict[str, np.ndarray] = {}
    for target in SPEC_PASS_TARGETS:
        selection = metadata["selected_models"][target]
        model = _load_model(str(directory / selection["file"]))
        probability = np.asarray(model.predict_proba(frame)[:, 1], dtype=float)
        passed = probability >= float(selection.get("threshold", 0.5))
        direct_spec_probabilities[target] = probability
        direct_spec_flags[target] = passed
        measurement = SPEC_TARGET_TO_MEASUREMENT[target]
        summaries[measurement]["predicted_spec_pass_rate_pct"] = 100.0 * float(np.mean(passed))
        summaries[measurement]["mean_spec_pass_probability_pct"] = 100.0 * float(np.mean(probability))

    defect_site_predictions: dict[str, np.ndarray] = {}
    for target in DEFECT_REGRESSION_TARGETS:
        selection = metadata["selected_models"][target]
        model = _load_model(str(directory / selection["file"]))
        predicted = np.asarray(model.predict(frame), dtype=float)
        if target == "Defect_Count":
            predicted = np.clip(predicted, 0.0, None)
        else:
            predicted = np.clip(predicted, 0.0, 10.0)
        defect_site_predictions[target] = predicted

    particle_selection = metadata["selected_models"]["Particle_Defect"]
    particle_model = _load_model(str(directory / particle_selection["file"]))
    particle_probability = np.asarray(particle_model.predict_proba(frame)[:, 1], dtype=float)
    mean_particle_probability = float(np.mean(particle_probability))
    expected_defect_sites = float(np.sum(particle_probability))
    wafer_any_particle_probability = float(1.0 - np.prod(1.0 - np.clip(particle_probability, 0.0, 1.0)))

    empirical_spec_pass_rate = 100.0 * float(np.mean(np.concatenate(empirical_spec_flags)))
    overall_hard_spec_pass_rate = 100.0 * float(
        np.mean(np.concatenate([direct_spec_flags[target] for target in SPEC_PASS_TARGETS]))
    )
    overall_mean_spec_probability = 100.0 * float(
        np.mean(
            np.concatenate(
                [direct_spec_probabilities[target] for target in SPEC_PASS_TARGETS]
            )
        )
    )
    particle_free_site_rate = 100.0 * (1.0 - mean_particle_probability)
    quality_index_v1 = 0.70 * empirical_spec_pass_rate + 0.30 * particle_free_site_rate

    predicted_total_defect_count = float(np.sum(defect_site_predictions["Defect_Count"]))
    predicted_mean_defect_severity = float(np.mean(defect_site_predictions["Defect_Severity_Score"]))
    weights = quality_v2["weights"]
    references = quality_v2["defect_score_reference"]
    wafer_components = _quality_component_scores(
        overall_mean_spec_probability,
        predicted_total_defect_count,
        predicted_mean_defect_severity,
        references["overall"],
        weights,
    )
    wafer_components_hard = _quality_component_scores(
        overall_hard_spec_pass_rate,
        predicted_total_defect_count,
        predicted_mean_defect_severity,
        references["overall"],
        weights,
    )

    template_frame = pd.DataFrame(metadata["site_template"])
    zone_metrics: list[dict[str, Any]] = []
    for zone in sorted(map(str, template_frame["Zone"].unique())):
        zone_mask = template_frame["Zone"].astype(str).eq(zone).to_numpy()
        zone_hard_spec_pass_rate = 100.0 * float(
            np.mean(
                np.concatenate(
                    [direct_spec_flags[target][zone_mask] for target in SPEC_PASS_TARGETS]
                )
            )
        )
        zone_mean_spec_probability = 100.0 * float(
            np.mean(
                np.concatenate(
                    [
                        direct_spec_probabilities[target][zone_mask]
                        for target in SPEC_PASS_TARGETS
                    ]
                )
            )
        )
        zone_total_defect_count = float(np.sum(defect_site_predictions["Defect_Count"][zone_mask]))
        zone_mean_severity = float(np.mean(defect_site_predictions["Defect_Severity_Score"][zone_mask]))
        components = _quality_component_scores(
            zone_mean_spec_probability,
            zone_total_defect_count,
            zone_mean_severity,
            references["zones"][zone],
            weights,
        )
        hard_components = _quality_component_scores(
            zone_hard_spec_pass_rate,
            zone_total_defect_count,
            zone_mean_severity,
            references["zones"][zone],
            weights,
        )
        zone_metrics.append(
            {
                "zone": zone,
                "site_count": int(np.sum(zone_mask)),
                "predicted_mean_spec_pass_probability_pct": zone_mean_spec_probability,
                "predicted_spec_pass_rate_pct": zone_hard_spec_pass_rate,
                "predicted_total_defect_count": zone_total_defect_count,
                "predicted_mean_defect_severity": zone_mean_severity,
                "spec_pass_score_pct": components["spec_pass_score_pct"],
                "defect_count_score_pct": components["defect_count_score_pct"],
                "defect_severity_score_pct": components["defect_severity_score_pct"],
                "zone_quality_pct": components["core_quality_pct"],
                "zone_quality_v2_hard_pct": hard_components["core_quality_pct"],
                "zone_risk_pct": 100.0 - components["core_quality_pct"],
            }
        )
    worst_zone = min(zone_metrics, key=lambda row: row["zone_quality_pct"])
    worst_zone_hard = min(zone_metrics, key=lambda row: row["zone_quality_v2_hard_pct"])
    quality_index = (
        float(weights["wafer_core"]) * wafer_components["core_quality_pct"]
        + float(weights["worst_zone_guardrail"]) * float(worst_zone["zone_quality_pct"])
    )
    quality_index_v2_hard = (
        float(weights["wafer_core"]) * wafer_components_hard["core_quality_pct"]
        + float(weights["worst_zone_guardrail"])
        * float(worst_zone_hard["zone_quality_v2_hard_pct"])
    )
    closure = parameter_closure_score(recipe, metadata, clip=True)

    warnings = []
    if process != "Trench Etch" or layer != "Dataset scope (unspecified layer)":
        warnings.append("Process and Layer are metadata only because the source data has no Process/Layer columns.")
    for parameter in PARAMETER_COLUMNS:
        lower, upper = metadata["parameter_ranges"][parameter]
        value = float(recipe[parameter])
        if value < lower or value > upper:
            warnings.append(f"{parameter}={value:g} is outside the observed training range [{lower:g}, {upper:g}].")

    site_rows = []
    template = metadata["site_template"]
    for row_index, template_row in enumerate(template):
        row = {
            "Site_ID": template_row["Site_ID"],
            "Zone": template_row["Zone"],
            "Top_CD_nm": float(site_predictions["Top_CD_nm"][row_index]),
            "Mid_CD_nm": float(site_predictions["Mid_CD_nm"][row_index]),
            "Bottom_CD_nm": float(site_predictions["Bottom_CD_nm"][row_index]),
            "Depth_nm": float(site_predictions["Depth_nm"][row_index]),
            "Particle_Probability": float(particle_probability[row_index]),
            "Top_CD_Spec_Pass_Probability": float(direct_spec_probabilities["Top_CD_Spec_Pass"][row_index]),
            "Mid_CD_Spec_Pass_Probability": float(direct_spec_probabilities["Mid_CD_Spec_Pass"][row_index]),
            "Bottom_CD_Spec_Pass_Probability": float(direct_spec_probabilities["Bottom_CD_Spec_Pass"][row_index]),
            "Depth_Spec_Pass_Probability": float(direct_spec_probabilities["Depth_Spec_Pass"][row_index]),
            "Predicted_Defect_Count": float(defect_site_predictions["Defect_Count"][row_index]),
            "Predicted_Defect_Severity_Score": float(defect_site_predictions["Defect_Severity_Score"][row_index]),
        }
        site_rows.append(row)

    return {
        "context": {
            "process": process,
            "layer": layer,
            "equipment_model": equipment_model,
            "chamber_id": chamber_id,
        },
        "parameter_closure_to_rev15_pct": closure["score_pct"],
        "predicted_quality_index_pct": float(np.clip(quality_index, 0.0, 100.0)),
        "predicted_quality_index_v2_hard_pct": float(
            np.clip(quality_index_v2_hard, 0.0, 100.0)
        ),
        "predicted_quality_index_v1_pct": float(np.clip(quality_index_v1, 0.0, 100.0)),
        "predicted_core_quality_pct": wafer_components["core_quality_pct"],
        "predicted_mean_spec_pass_probability_pct": overall_mean_spec_probability,
        "predicted_overall_spec_pass_rate_pct": overall_hard_spec_pass_rate,
        "predicted_empirical_spec_pass_rate_pct": empirical_spec_pass_rate,
        "predicted_particle_free_site_rate_pct": particle_free_site_rate,
        "predicted_expected_defect_sites": expected_defect_sites,
        "predicted_total_defect_count": predicted_total_defect_count,
        "predicted_mean_defect_severity": predicted_mean_defect_severity,
        "defect_count_score_pct": wafer_components["defect_count_score_pct"],
        "defect_severity_score_pct": wafer_components["defect_severity_score_pct"],
        "worst_zone": worst_zone["zone"],
        "worst_zone_quality_pct": worst_zone["zone_quality_pct"],
        "zone_metrics": zone_metrics,
        "predicted_any_particle_probability_pct_independence_approx": round(100.0 * wafer_any_particle_probability, 1),
        "wafer_metrics": summaries,
        "site_predictions": site_rows,
        "warnings": warnings,
        "score_definition": {
            "parameter_closure": "Recovered original formula; average normalized distance from Base to Rev15 across 12 parameters, clipped to 0-100 for simulator inputs.",
            "predicted_quality_index": (
                "Relative quality score v3 = 85% wafer core + 15% worst-zone core. Each core = 60% mean "
                "CD/Depth spec-pass probability + 20% Base-to-Rev15 defect-count score + 20% Base-to-Rev15 "
                "severity score. This score is not a production pass probability."
            ),
            "quality_v1_reference": (
                "Legacy comparison only = 70% empirical CD/Depth pass + 30% predicted particle-free site rate."
            ),
            "defect_reference": (
                "Base and Rev15 observed averages are normalization anchors, not production acceptance limits."
            ),
        },
    }


def _predict_quality_summaries(
    recipes: list[dict[str, float]],
    equipment_model: str,
    chamber_id: str,
    *,
    model_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Vectorized quality-v2 prediction used for one-factor sensitivity screening.

    The public ``predict_wafer`` function returns the full 25-site payload.  A
    recommendation scan only needs the quality-v2 components, so evaluating all
    trial recipes in one model call is substantially faster and keeps the
    recommendation button responsive.
    """
    if not recipes:
        return []

    directory = _resolve_model_dir(model_dir)
    metadata = load_metadata(str(directory))
    validate_tool_pair(equipment_model, chamber_id, metadata)
    frames = [_site_input_frame(recipe, equipment_model, chamber_id, metadata) for recipe in recipes]
    frame = pd.concat(frames, ignore_index=True)
    site_count = len(metadata["site_template"])
    recipe_count = len(recipes)

    direct_spec_probabilities: dict[str, np.ndarray] = {}
    direct_spec_flags: dict[str, np.ndarray] = {}
    for target in SPEC_PASS_TARGETS:
        selection = metadata["selected_models"][target]
        model = _load_model(str(directory / selection["file"]))
        probability = np.asarray(model.predict_proba(frame)[:, 1], dtype=float)
        reshaped = probability.reshape(recipe_count, site_count)
        direct_spec_probabilities[target] = reshaped
        direct_spec_flags[target] = reshaped >= float(selection.get("threshold", 0.5))

    defect_predictions: dict[str, np.ndarray] = {}
    for target in DEFECT_REGRESSION_TARGETS:
        selection = metadata["selected_models"][target]
        model = _load_model(str(directory / selection["file"]))
        predicted = np.asarray(model.predict(frame), dtype=float).reshape(recipe_count, site_count)
        if target == "Defect_Count":
            predicted = np.clip(predicted, 0.0, None)
        else:
            predicted = np.clip(predicted, 0.0, 10.0)
        defect_predictions[target] = predicted

    quality_v2 = metadata["quality_v2"]
    weights = quality_v2["weights"]
    references = quality_v2["defect_score_reference"]
    template_frame = pd.DataFrame(metadata["site_template"])
    zone_masks = {
        zone: template_frame["Zone"].astype(str).eq(zone).to_numpy()
        for zone in sorted(map(str, template_frame["Zone"].unique()))
    }

    summaries: list[dict[str, Any]] = []
    for recipe_index, recipe in enumerate(recipes):
        overall_hard_spec_pass_rate = 100.0 * float(
            np.mean(
                np.concatenate(
                    [direct_spec_flags[target][recipe_index] for target in SPEC_PASS_TARGETS]
                )
            )
        )
        overall_mean_spec_probability = 100.0 * float(
            np.mean(
                np.concatenate(
                    [
                        direct_spec_probabilities[target][recipe_index]
                        for target in SPEC_PASS_TARGETS
                    ]
                )
            )
        )
        total_defect_count = float(np.sum(defect_predictions["Defect_Count"][recipe_index]))
        mean_defect_severity = float(np.mean(defect_predictions["Defect_Severity_Score"][recipe_index]))
        wafer_components = _quality_component_scores(
            overall_mean_spec_probability,
            total_defect_count,
            mean_defect_severity,
            references["overall"],
            weights,
        )

        zone_rows = []
        for zone, zone_mask in zone_masks.items():
            zone_mean_spec_probability = 100.0 * float(
                np.mean(
                    np.concatenate(
                        [
                            direct_spec_probabilities[target][recipe_index][zone_mask]
                            for target in SPEC_PASS_TARGETS
                        ]
                    )
                )
            )
            zone_total_defect_count = float(
                np.sum(defect_predictions["Defect_Count"][recipe_index][zone_mask])
            )
            zone_mean_severity = float(
                np.mean(defect_predictions["Defect_Severity_Score"][recipe_index][zone_mask])
            )
            zone_components = _quality_component_scores(
                zone_mean_spec_probability,
                zone_total_defect_count,
                zone_mean_severity,
                references["zones"][zone],
                weights,
            )
            zone_rows.append(
                {
                    "zone": zone,
                    "zone_quality_pct": zone_components["core_quality_pct"],
                }
            )

        worst_zone = min(zone_rows, key=lambda row: row["zone_quality_pct"])
        quality_index = (
            float(weights["wafer_core"]) * wafer_components["core_quality_pct"]
            + float(weights["worst_zone_guardrail"]) * float(worst_zone["zone_quality_pct"])
        )
        summaries.append(
            {
                "parameter_closure_to_rev15_pct": parameter_closure_score(
                    recipe, metadata, clip=True
                )["score_pct"],
                "predicted_quality_index_pct": float(np.clip(quality_index, 0.0, 100.0)),
                "predicted_core_quality_pct": wafer_components["core_quality_pct"],
                "predicted_mean_spec_pass_probability_pct": overall_mean_spec_probability,
                "predicted_overall_spec_pass_rate_pct": overall_hard_spec_pass_rate,
                "predicted_total_defect_count": total_defect_count,
                "predicted_mean_defect_severity": mean_defect_severity,
                "worst_zone": worst_zone["zone"],
                "worst_zone_quality_pct": worst_zone["zone_quality_pct"],
            }
        )
    return summaries


def _observed_parameter_values(
    metadata: dict[str, Any], parameter: str, current: float
) -> list[float]:
    """Return the actual values seen across Base--Rev15 plus the current input."""
    values = {float(current)}
    values.update(float(recipe[parameter]) for recipe in metadata["recipe_versions"].values())
    lower, upper = map(float, metadata["parameter_ranges"][parameter])
    return sorted(value for value in values if lower <= value <= upper)


def recommend_parameter_changes(
    recipe: dict[str, float],
    equipment_model: str,
    chamber_id: str,
    *,
    process: str = "Trench Etch",
    layer: str = "Dataset scope (unspecified layer)",
    model_dir: str | Path | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    directory = _resolve_model_dir(model_dir)
    metadata = load_metadata(str(directory))
    baseline = predict_wafer(
        recipe,
        equipment_model,
        chamber_id,
        process=process,
        layer=layer,
        model_dir=directory,
    )
    baseline_quality = float(baseline["predicted_quality_index_pct"])
    baseline_closure = float(baseline["parameter_closure_to_rev15_pct"])
    baseline_worst_zone = float(baseline["worst_zone_quality_pct"])
    baseline_spec_probability = float(
        baseline["predicted_mean_spec_pass_probability_pct"]
    )
    baseline_hard_pass_rate = float(baseline["predicted_overall_spec_pass_rate_pct"])
    baseline_defect_count = float(baseline["predicted_total_defect_count"])
    baseline_severity = float(baseline["predicted_mean_defect_severity"])
    candidate_recipes: list[dict[str, float]] = []
    candidate_keys: list[tuple[str, float]] = []
    for parameter in PARAMETER_COLUMNS:
        current = float(recipe[parameter])
        for proposed in _observed_parameter_values(metadata, parameter, current):
            if abs(proposed - current) <= 1e-12:
                continue
            trial_recipe = dict(recipe)
            trial_recipe[parameter] = proposed
            candidate_recipes.append(trial_recipe)
            candidate_keys.append((parameter, proposed))

    candidate_predictions = _predict_quality_summaries(
        candidate_recipes,
        equipment_model,
        chamber_id,
        model_dir=directory,
    )
    candidates: list[dict[str, Any]] = []
    for (parameter, proposed), prediction in zip(candidate_keys, candidate_predictions):
        current = float(recipe[parameter])
        quality_gain = float(prediction["predicted_quality_index_pct"]) - baseline_quality
        closure_gain = float(prediction["parameter_closure_to_rev15_pct"]) - baseline_closure
        worst_zone_gain = float(prediction["worst_zone_quality_pct"]) - baseline_worst_zone
        spec_probability_gain = (
            float(prediction["predicted_mean_spec_pass_probability_pct"])
            - baseline_spec_probability
        )
        optimal = float(metadata["optimal_recipe"][parameter])
        candidates.append(
            {
                "parameter": parameter,
                "direction": "increase" if proposed > current else "decrease",
                "current": current,
                "proposed": proposed,
                "step": abs(proposed - current),
                "predicted_quality_gain_pct_point": quality_gain,
                "parameter_closure_gain_pct_point": closure_gain,
                "worst_zone_quality_gain_pct_point": worst_zone_gain,
                "predicted_mean_spec_pass_probability_change_pct_point": spec_probability_gain,
                "predicted_overall_spec_pass_rate_change_pct_point": round(
                    float(prediction["predicted_overall_spec_pass_rate_pct"])
                    - baseline_hard_pass_rate,
                    12,
                ),
                "predicted_total_defect_count_change": (
                    float(prediction["predicted_total_defect_count"])
                    - baseline_defect_count
                ),
                "predicted_mean_defect_severity_change": (
                    float(prediction["predicted_mean_defect_severity"])
                    - baseline_severity
                ),
                "moves_toward_rev15": abs(proposed - optimal) < abs(current - optimal),
                "predicted_quality_index_pct": prediction["predicted_quality_index_pct"],
                "predicted_mean_spec_pass_probability_pct": prediction[
                    "predicted_mean_spec_pass_probability_pct"
                ],
                "predicted_worst_zone": prediction["worst_zone"],
            }
        )

    epsilon = 1e-9

    def ranking_key(row: dict[str, Any]) -> tuple[float, ...]:
        return (
            float(row["predicted_quality_gain_pct_point"]),
            float(row["worst_zone_quality_gain_pct_point"]),
            float(row["predicted_mean_spec_pass_probability_change_pct_point"]),
            -float(row["predicted_total_defect_count_change"]),
            -float(row["predicted_mean_defect_severity_change"]),
        )

    def improves_quality_or_tied_submetric(row: dict[str, Any]) -> bool:
        quality_gain = float(row["predicted_quality_gain_pct_point"])
        if quality_gain > epsilon:
            return True
        if abs(quality_gain) > epsilon:
            return False
        return any(
            value > epsilon
            for value in (
                float(row["worst_zone_quality_gain_pct_point"]),
                float(row["predicted_mean_spec_pass_probability_change_pct_point"]),
                -float(row["predicted_total_defect_count_change"]),
                -float(row["predicted_mean_defect_severity_change"]),
            )
        )

    best_by_parameter: list[dict[str, Any]] = []
    parameter_sensitivity: list[dict[str, Any]] = []
    for parameter in PARAMETER_COLUMNS:
        rows = [row for row in candidates if row["parameter"] == parameter]
        tested_quality = [baseline_quality, *[float(row["predicted_quality_index_pct"]) for row in rows]]
        response_fields = (
            "predicted_quality_gain_pct_point",
            "worst_zone_quality_gain_pct_point",
            "predicted_mean_spec_pass_probability_change_pct_point",
            "predicted_overall_spec_pass_rate_change_pct_point",
            "predicted_total_defect_count_change",
            "predicted_mean_defect_severity_change",
        )
        response_detected = any(
            abs(float(row[field])) > 1e-12 for row in rows for field in response_fields
        )
        quality_response_detected = any(
            abs(float(row["predicted_quality_gain_pct_point"])) > 1e-12 for row in rows
        )
        ranked_rows = sorted(
            rows,
            key=ranking_key,
            reverse=True,
        )
        best = ranked_rows[0] if ranked_rows else None
        if best is not None:
            best_by_parameter.append(best)
        parameter_sensitivity.append(
            {
                "parameter": parameter,
                "current": float(recipe[parameter]),
                "tested_values": _observed_parameter_values(
                    metadata, parameter, float(recipe[parameter])
                ),
                "quality_response_detected": quality_response_detected,
                "quality_component_response_detected": response_detected,
                "predicted_quality_min_pct": min(tested_quality),
                "predicted_quality_max_pct": max(tested_quality),
                "best_tested_value": None if best is None else best["proposed"],
                "best_predicted_quality_gain_pct_point": (
                    0.0 if best is None else best["predicted_quality_gain_pct_point"]
                ),
            }
        )

    positive = [row for row in best_by_parameter if improves_quality_or_tied_submetric(row)]
    positive.sort(key=ranking_key, reverse=True)
    selected = positive[:top_n]
    for row in selected:
        display_tie = round(float(row["predicted_quality_index_pct"]), 2) == round(
            baseline_quality, 2
        )
        row["recommendation_reason"] = (
            "display_tie_submetric_improvement" if display_tie else "quality_improvement"
        )
    exploratory_tradeoffs = [row for row in positive if not row["moves_toward_rev15"]]

    candidate_lookup = {
        (str(row["parameter"]), float(row["proposed"])): row for row in candidates
    }
    rev15_completion: list[dict[str, Any]] = []
    for parameter in PARAMETER_COLUMNS:
        current = float(recipe[parameter])
        target = float(metadata["optimal_recipe"][parameter])
        if abs(current - target) <= epsilon:
            continue
        row = dict(candidate_lookup[(parameter, target)])
        if improves_quality_or_tied_submetric(row):
            alignment = "aligned"
        elif all(abs(value) <= epsilon for value in ranking_key(row)):
            alignment = "neutral"
        else:
            alignment = "model_conflict"
        row["model_alignment"] = alignment
        row["completion_note"] = (
            "현재 입력값과 Rev15 기준값의 차이입니다. AI 품질추천과 별도이며, "
            "Change_Notes의 이력 순서를 재현하는 항목이 아닙니다."
        )
        rev15_completion.append(row)
    rev15_completion.sort(
        key=lambda row: (
            bool(row["model_alignment"] == "aligned"),
            float(row["parameter_closure_gain_pct_point"]),
        ),
        reverse=True,
    )
    return {
        "baseline_quality_index_pct": baseline_quality,
        "baseline_parameter_closure_pct": baseline_closure,
        "baseline_worst_zone_quality_pct": baseline_worst_zone,
        "baseline_mean_spec_pass_probability_pct": baseline_spec_probability,
        "baseline_predicted_total_defect_count": baseline_defect_count,
        "baseline_predicted_mean_defect_severity": baseline_severity,
        "recommendations": selected,
        "rev15_completion_recommendations": rev15_completion,
        "exploratory_tradeoffs": exploratory_tradeoffs[:top_n],
        "parameter_sensitivity": parameter_sensitivity,
        "optimization_priority": (
            "AI lane: unrounded relative quality score v3 first, then worst-zone quality, mean spec-pass "
            "probability, lower defect count, and lower severity. Exact quality ties remain eligible when "
            "a detailed quality component improves. Rev15 completion is a separate lane."
        ),
        "method": (
            "One-factor-at-a-time screening at values actually observed from Base through Rev15. "
            "Each recommendation starts from the exact submitted custom recipe and selects the best "
            "relative-quality-v3 improvement for that parameter. A move may point away from Rev15 when the "
            "deployed model predicts better quality; this is a model screening result, not causal DOE proof."
        ),
    }
