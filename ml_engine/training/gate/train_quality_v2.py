from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate

from train_models import (
    DEFAULT_RECIPE_CSV,
    DEFAULT_XLSX,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    classification_pipeline,
    load_and_prepare,
    regression_pipeline,
)


SPEC_PASS_TARGETS = [
    "Top_CD_Spec_Pass",
    "Mid_CD_Spec_Pass",
    "Bottom_CD_Spec_Pass",
    "Depth_Spec_Pass",
]
DEFECT_REGRESSION_TARGETS = ["Defect_Count", "Defect_Severity_Score"]
FAMILIES = ["RandomForest", "XGBoost", "MLP"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train quality-v2 site spec-pass, defect-count, and defect-severity models."
    )
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--recipe-csv", type=Path, default=DEFAULT_RECIPE_CSV)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--recipe-cv-folds", type=int, default=4)
    return parser.parse_args()


def _defect_references(data: pd.DataFrame) -> dict:
    wafer_keys = ["Recipe_Version", "Wafer_Key"]
    wafer = (
        data.groupby(wafer_keys, as_index=False)
        .agg(
            Total_Defect_Count=("Defect_Count", "sum"),
            Mean_Defect_Severity=("Defect_Severity_Score", "mean"),
        )
    )
    recipe_reference = wafer.groupby("Recipe_Version").agg(
        Total_Defect_Count=("Total_Defect_Count", "mean"),
        Mean_Defect_Severity=("Mean_Defect_Severity", "mean"),
    )

    zone_wafer = (
        data.groupby(["Recipe_Version", "Wafer_Key", "Zone"], as_index=False)
        .agg(
            Total_Defect_Count=("Defect_Count", "sum"),
            Mean_Defect_Severity=("Defect_Severity_Score", "mean"),
        )
    )
    zone_reference = zone_wafer.groupby(["Recipe_Version", "Zone"]).agg(
        Total_Defect_Count=("Total_Defect_Count", "mean"),
        Mean_Defect_Severity=("Mean_Defect_Severity", "mean"),
    )

    overall = {
        "base_total_defect_count": float(recipe_reference.loc["Base", "Total_Defect_Count"]),
        "optimal_total_defect_count": float(recipe_reference.loc["Rev15", "Total_Defect_Count"]),
        "base_mean_defect_severity": float(recipe_reference.loc["Base", "Mean_Defect_Severity"]),
        "optimal_mean_defect_severity": float(recipe_reference.loc["Rev15", "Mean_Defect_Severity"]),
    }
    zones = {}
    for zone in sorted(map(str, data["Zone"].unique())):
        zones[zone] = {
            "base_total_defect_count": float(zone_reference.loc[("Base", zone), "Total_Defect_Count"]),
            "optimal_total_defect_count": float(zone_reference.loc[("Rev15", zone), "Total_Defect_Count"]),
            "base_mean_defect_severity": float(zone_reference.loc[("Base", zone), "Mean_Defect_Severity"]),
            "optimal_mean_defect_severity": float(zone_reference.loc[("Rev15", zone), "Mean_Defect_Severity"]),
        }
    return {"overall": overall, "zones": zones}


def _tool_support(data: pd.DataFrame) -> dict:
    """Persist the observed wafer support for every valid tool pair and recipe."""
    wafer = data[
        ["Recipe_Version", "Wafer_Key", "Equipment_Model", "Chamber_ID"]
    ].drop_duplicates()
    pair_counts = (
        wafer.groupby(["Equipment_Model", "Chamber_ID"], sort=True)
        .size()
        .rename("wafer_count")
    )
    version_pair_counts = wafer.groupby(
        ["Recipe_Version", "Equipment_Model", "Chamber_ID"], sort=True
    ).size()
    version_order = ["Base", *[f"Rev{index}" for index in range(1, 16)]]
    nested_counts: dict[str, dict[str, dict[str, int]]] = {
        version: {} for version in version_order
    }
    for (version, equipment, chamber), count in version_pair_counts.items():
        nested_counts[str(version)].setdefault(str(equipment), {})[str(chamber)] = int(
            count
        )
    return {
        "total_training_wafers": int(len(wafer)),
        "valid_pairs": [
            {
                "equipment_model": str(equipment),
                "chamber_id": str(chamber),
                "wafer_count": int(count),
            }
            for (equipment, chamber), count in pair_counts.items()
        ],
        "recipe_pair_wafer_counts": nested_counts,
    }


def train_quality_models(
    data: pd.DataFrame,
    output_dir: Path,
    cv_folds: int,
    recipe_cv_folds: int,
) -> None:
    started_at = time.time()
    models_dir = output_dir / "models"
    results_dir = output_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    groups = data["Wafer_Key"]
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_index, test_index = next(split.split(data[FEATURE_COLUMNS], groups=groups))
    train_groups = groups.iloc[train_index]
    test_groups = groups.iloc[test_index]
    if set(train_groups) & set(test_groups):
        raise AssertionError("Wafer leakage detected between train and test.")

    X_train = data.iloc[train_index][FEATURE_COLUMNS]
    X_test = data.iloc[test_index][FEATURE_COLUMNS]
    X_all = data[FEATURE_COLUMNS]
    recipe_groups = data["Recipe_Version"]
    wafer_cv = GroupKFold(n_splits=cv_folds)
    recipe_cv = GroupKFold(n_splits=recipe_cv_folds)

    metrics_rows: list[dict] = []
    confusion_rows: list[dict] = []
    fitted: dict[tuple[str, str], object] = {}
    prediction_frame = data.iloc[test_index][
        [
            "Wafer_Key",
            "Recipe_Version",
            "Lot_ID",
            "Wafer_ID",
            "Site_ID",
            "Zone",
            *SPEC_PASS_TARGETS,
            *DEFECT_REGRESSION_TARGETS,
        ]
    ].copy()

    for target in DEFECT_REGRESSION_TARGETS:
        y_train = data.iloc[train_index][target].astype(float)
        y_test = data.iloc[test_index][target].astype(float)
        y_all = data[target].astype(float)
        for family in FAMILIES:
            wafer_scores = cross_validate(
                regression_pipeline(family),
                X_train,
                y_train,
                groups=train_groups,
                cv=wafer_cv,
                scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
                n_jobs=1,
                error_score="raise",
            )
            recipe_scores = cross_validate(
                regression_pipeline(family),
                X_all,
                y_all,
                groups=recipe_groups,
                cv=recipe_cv,
                scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
                n_jobs=1,
                error_score="raise",
            )
            pipeline = regression_pipeline(family)
            pipeline.fit(X_train, y_train)
            predicted = np.asarray(pipeline.predict(X_test), dtype=float)
            if target == "Defect_Count":
                predicted = np.clip(predicted, 0.0, None)
            else:
                predicted = np.clip(predicted, 0.0, 10.0)
            file_name = f"{target}__{family}.joblib"
            joblib.dump(pipeline, models_dir / file_name, compress=3)
            fitted[(target, family)] = pipeline
            prediction_frame[f"Pred_{target}__{family}"] = predicted
            metrics_rows.append(
                {
                    "Task": "Regression",
                    "Target": target,
                    "Model": family,
                    "Wafer_CV_MAE_Mean": float(-np.mean(wafer_scores["test_mae"])),
                    "Wafer_CV_RMSE_Mean": float(-np.mean(wafer_scores["test_rmse"])),
                    "Wafer_CV_R2_Mean": float(np.mean(wafer_scores["test_r2"])),
                    "Test_MAE": float(mean_absolute_error(y_test, predicted)),
                    "Test_RMSE": float(mean_squared_error(y_test, predicted) ** 0.5),
                    "Test_R2": float(r2_score(y_test, predicted)),
                    "Recipe_Holdout_CV_MAE_Mean": float(-np.mean(recipe_scores["test_mae"])),
                    "Recipe_Holdout_CV_RMSE_Mean": float(-np.mean(recipe_scores["test_rmse"])),
                    "Recipe_Holdout_CV_R2_Mean": float(np.mean(recipe_scores["test_r2"])),
                    "Model_File": file_name,
                }
            )

    for target in SPEC_PASS_TARGETS:
        y_train = data.iloc[train_index][target].astype(int)
        y_test = data.iloc[test_index][target].astype(int)
        y_all = data[target].astype(int)
        scale_pos_weight_train = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
        scale_pos_weight_all = float((y_all == 0).sum() / max(1, (y_all == 1).sum()))
        for family in FAMILIES:
            wafer_scores = cross_validate(
                classification_pipeline(family, scale_pos_weight_train),
                X_train,
                y_train,
                groups=train_groups,
                cv=wafer_cv,
                scoring={
                    "balanced_accuracy": "balanced_accuracy",
                    "f1": "f1",
                    "roc_auc": "roc_auc",
                    "average_precision": "average_precision",
                },
                n_jobs=1,
                error_score="raise",
            )
            recipe_scores = cross_validate(
                classification_pipeline(family, scale_pos_weight_all),
                X_all,
                y_all,
                groups=recipe_groups,
                cv=recipe_cv,
                scoring={
                    "balanced_accuracy": "balanced_accuracy",
                    "f1": "f1",
                    "roc_auc": "roc_auc",
                    "average_precision": "average_precision",
                },
                n_jobs=1,
                error_score="raise",
            )
            pipeline = classification_pipeline(family, scale_pos_weight_train)
            pipeline.fit(X_train, y_train)
            probability = np.asarray(pipeline.predict_proba(X_test)[:, 1], dtype=float)
            predicted = (probability >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
            file_name = f"{target}__{family}.joblib"
            joblib.dump(pipeline, models_dir / file_name, compress=3)
            fitted[(target, family)] = pipeline
            prediction_frame[f"Pred_{target}_Probability__{family}"] = probability
            metrics_rows.append(
                {
                    "Task": "Classification",
                    "Target": target,
                    "Model": family,
                    "Wafer_CV_Balanced_Accuracy_Mean": float(np.mean(wafer_scores["test_balanced_accuracy"])),
                    "Wafer_CV_F1_Mean": float(np.mean(wafer_scores["test_f1"])),
                    "Wafer_CV_ROC_AUC_Mean": float(np.mean(wafer_scores["test_roc_auc"])),
                    "Wafer_CV_Average_Precision_Mean": float(np.mean(wafer_scores["test_average_precision"])),
                    "Test_Balanced_Accuracy": float(balanced_accuracy_score(y_test, predicted)),
                    "Test_Precision": float(precision_score(y_test, predicted, zero_division=0)),
                    "Test_Recall": float(recall_score(y_test, predicted, zero_division=0)),
                    "Test_F1": float(f1_score(y_test, predicted, zero_division=0)),
                    "Test_ROC_AUC": float(roc_auc_score(y_test, probability)),
                    "Test_Average_Precision": float(average_precision_score(y_test, probability)),
                    "Recipe_Holdout_CV_Balanced_Accuracy_Mean": float(np.mean(recipe_scores["test_balanced_accuracy"])),
                    "Recipe_Holdout_CV_F1_Mean": float(np.mean(recipe_scores["test_f1"])),
                    "Recipe_Holdout_CV_ROC_AUC_Mean": float(np.mean(recipe_scores["test_roc_auc"])),
                    "Recipe_Holdout_CV_Average_Precision_Mean": float(np.mean(recipe_scores["test_average_precision"])),
                    "Threshold": 0.5,
                    "TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TP": int(tp),
                    "Model_File": file_name,
                }
            )
            confusion_rows.extend(
                [
                    {"Target": target, "Model": family, "Actual": 0, "Predicted": 0, "Count": int(tn)},
                    {"Target": target, "Model": family, "Actual": 0, "Predicted": 1, "Count": int(fp)},
                    {"Target": target, "Model": family, "Actual": 1, "Predicted": 0, "Count": int(fn)},
                    {"Target": target, "Model": family, "Actual": 1, "Predicted": 1, "Count": int(tp)},
                ]
            )

    metrics = pd.DataFrame(metrics_rows)
    selected: dict[str, dict] = {}
    for target in DEFECT_REGRESSION_TARGETS:
        row = metrics[(metrics["Task"] == "Regression") & (metrics["Target"] == target)].sort_values(
            ["Recipe_Holdout_CV_MAE_Mean", "Wafer_CV_MAE_Mean"]
        ).iloc[0]
        selected[target] = {
            "family": str(row["Model"]),
            "file": str(row["Model_File"]),
            "selection_rule": "lowest whole-recipe-holdout MAE, then wafer-group CV MAE",
        }
    for target in SPEC_PASS_TARGETS:
        row = metrics[(metrics["Task"] == "Classification") & (metrics["Target"] == target)].sort_values(
            ["Recipe_Holdout_CV_Balanced_Accuracy_Mean", "Recipe_Holdout_CV_ROC_AUC_Mean"],
            ascending=False,
        ).iloc[0]
        selected[target] = {
            "family": str(row["Model"]),
            "file": str(row["Model_File"]),
            "selection_rule": "highest whole-recipe-holdout balanced accuracy, then ROC-AUC",
            "threshold": 0.5,
        }

    importance_rows: list[dict] = []
    for target, selection in selected.items():
        family = selection["family"]
        pipeline = fitted[(target, family)]
        if target in SPEC_PASS_TARGETS:
            y_test = data.iloc[test_index][target].astype(int)
            scoring = "balanced_accuracy"
            task = "Classification"
        else:
            y_test = data.iloc[test_index][target].astype(float)
            scoring = "neg_mean_absolute_error"
            task = "Regression"
        importance = permutation_importance(
            pipeline,
            X_test,
            y_test,
            scoring=scoring,
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        for feature, mean_value, std_value in zip(
            FEATURE_COLUMNS, importance.importances_mean, importance.importances_std
        ):
            importance_rows.append(
                {
                    "Task": task,
                    "Target": target,
                    "Model": family,
                    "Feature": feature,
                    "Permutation_Importance_Mean": float(mean_value),
                    "Permutation_Importance_Std": float(std_value),
                }
            )

    metrics.to_csv(results_dir / "quality_v2_model_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(confusion_rows).to_csv(
        results_dir / "quality_v2_confusion_matrices.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(importance_rows).sort_values(
        ["Target", "Permutation_Importance_Mean"], ascending=[True, False]
    ).to_csv(results_dir / "quality_v2_permutation_importance.csv", index=False, encoding="utf-8-sig")
    prediction_frame.to_csv(results_dir / "quality_v2_test_predictions.csv", index=False, encoding="utf-8-sig")

    metadata_path = models_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = 3
    metadata["selected_models"].update(selected)
    metadata["tool_support"] = _tool_support(data)
    metadata["quality_score_version"] = 3
    metadata["quality_v2"] = {
        "spec_pass_targets": SPEC_PASS_TARGETS,
        "defect_regression_targets": DEFECT_REGRESSION_TARGETS,
        "selected_models": selected,
        "defect_score_reference": _defect_references(data),
        "weights": {
            "spec_pass": 0.60,
            "defect_count": 0.20,
            "defect_severity": 0.20,
            "wafer_core": 0.85,
            "worst_zone_guardrail": 0.15,
        },
        "quality_formula": (
            "relative-quality-v3 core = 0.60*mean direct predicted spec-pass probability + "
            "0.20*Base-to-Rev15 defect-count score + 0.20*Base-to-Rev15 severity score; "
            "final = 0.85*wafer core + 0.15*worst-zone core; this is not a production pass probability"
        ),
        "target_note": (
            "Defect references are observed Base and Rev15 averages, not engineering-authorized acceptance limits. "
            "Site_ID is a location key; Zone, radius, and angle are model features."
        ),
        "training_seconds": round(time.time() - started_at, 1),
    }
    metadata["simulator_quality_index"] = metadata["quality_v2"]["quality_formula"]
    limitations = metadata.setdefault("limitations", [])
    extra_limit = (
        "Quality-v2 defect normalization uses observed Base and Rev15 averages because official defect-count "
        "and severity acceptance limits were not supplied."
    )
    if extra_limit not in limitations:
        limitations.append(extra_limit)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "selected_models": selected,
                "training_seconds": metadata["quality_v2"]["training_seconds"],
                "wafer_train_groups": int(train_groups.nunique()),
                "wafer_test_groups": int(test_groups.nunique()),
                "wafer_overlap": len(set(train_groups) & set(test_groups)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    _, _, data = load_and_prepare(args.xlsx.resolve(), args.recipe_csv.resolve())
    train_quality_models(data, args.output_dir.resolve(), args.cv_folds, args.recipe_cv_folds)


if __name__ == "__main__":
    main()
