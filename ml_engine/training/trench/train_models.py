from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
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
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from simulator_core import PARAMETER_COLUMNS, REGRESSION_TARGETS


DEFAULT_XLSX = Path(r"C:\Users\tmdgu\iCloudDrive\자격증\SK SU\프로젝트\두번째공정\trench recipe full dataset.xlsx")
DEFAULT_RECIPE_CSV = Path(r"C:\Users\tmdgu\iCloudDrive\자격증\SK SU\프로젝트\두번째공정\trench recipe master.csv")
RANDOM_STATE = 42
NUMERIC_SITE_COLUMNS = ["Radius_frac", "Angle_sin", "Angle_cos"]
CATEGORICAL_COLUMNS = ["Equipment_Model", "Chamber_ID", "Zone"]
FEATURE_COLUMNS = [*PARAMETER_COLUMNS, *NUMERIC_SITE_COLUMNS, *CATEGORICAL_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train wafer-grouped semiconductor recipe models.")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--recipe-csv", type=Path, default=DEFAULT_RECIPE_CSV)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cv-folds", type=int, default=3)
    return parser.parse_args()


def build_preprocessor(dense: bool) -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler() if dense else "passthrough")])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, [*PARAMETER_COLUMNS, *NUMERIC_SITE_COLUMNS]), ("cat", categorical, CATEGORICAL_COLUMNS)],
        sparse_threshold=0.0 if dense else 1.0,
    )


def regression_pipeline(family: str) -> Pipeline:
    if family == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=360,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return Pipeline([("prep", build_preprocessor(False)), ("model", model)])
    if family == "XGBoost":
        model = XGBRegressor(
            n_estimators=420,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=4,
            random_state=RANDOM_STATE,
        )
        return Pipeline([("prep", build_preprocessor(False)), ("model", model)])
    if family == "MLP":
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=RANDOM_STATE,
        )
        return Pipeline([("prep", build_preprocessor(True)), ("model", model)])
    raise ValueError(f"Unknown family: {family}")


def classification_pipeline(family: str, scale_pos_weight: float) -> Pipeline:
    if family == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=420,
            min_samples_leaf=2,
            max_features=0.8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return Pipeline([("prep", build_preprocessor(False)), ("model", model)])
    if family == "XGBoost":
        model = XGBClassifier(
            n_estimators=440,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            n_jobs=4,
            random_state=RANDOM_STATE,
        )
        return Pipeline([("prep", build_preprocessor(False)), ("model", model)])
    if family == "MLP":
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=RANDOM_STATE,
        )
        return Pipeline([("prep", build_preprocessor(True)), ("model", model)])
    raise ValueError(f"Unknown family: {family}")


def load_and_prepare(xlsx: Path, recipe_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recipe = pd.read_csv(recipe_csv, encoding="utf-8-sig")
    recipe_sheet = pd.read_excel(xlsx, sheet_name="Recipe_Master")
    wafer = pd.read_excel(xlsx, sheet_name="Wafer_Summary")
    site = pd.read_excel(xlsx, sheet_name="Site_Level_Raw")
    if recipe.shape != recipe_sheet.shape:
        raise ValueError("Recipe CSV and workbook Recipe_Master shapes differ.")
    for column in recipe.columns:
        if pd.api.types.is_numeric_dtype(recipe[column]):
            if not np.allclose(recipe[column], recipe_sheet[column], equal_nan=True):
                raise ValueError(f"Recipe CSV and workbook differ in {column}.")
        elif not recipe[column].astype(str).equals(recipe_sheet[column].astype(str)):
            raise ValueError(f"Recipe CSV and workbook differ in {column}.")
    merged = site.merge(recipe[["Recipe_Version", *PARAMETER_COLUMNS]], on="Recipe_Version", how="left", validate="many_to_one")
    if merged[PARAMETER_COLUMNS].isna().any().any():
        raise ValueError("At least one site row did not match a recipe version.")
    merged["Wafer_Key"] = merged[["Recipe_Version", "Lot_ID", "Wafer_ID"]].astype(str).agg("|".join, axis=1)
    angle = np.deg2rad(merged["Angle_deg"].fillna(0.0).astype(float))
    merged["Angle_sin"] = np.sin(angle)
    merged["Angle_cos"] = np.cos(angle)
    merged["Particle_Defect"] = merged["Defect_Count"].gt(0).astype(int)
    return recipe, wafer, merged


def fit_models(recipe: pd.DataFrame, wafer: pd.DataFrame, data: pd.DataFrame, output_dir: Path, cv_folds: int) -> None:
    started_at = time.time()
    models_dir = output_dir / "models"
    results_dir = output_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    groups = data["Wafer_Key"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_index, test_index = next(splitter.split(data[FEATURE_COLUMNS], groups=groups))
    train_groups = set(groups.iloc[train_index])
    test_groups = set(groups.iloc[test_index])
    if train_groups & test_groups:
        raise AssertionError("Wafer leakage detected between train and test.")
    X_train = data.iloc[train_index][FEATURE_COLUMNS]
    X_test = data.iloc[test_index][FEATURE_COLUMNS]
    groups_train = groups.iloc[train_index]

    split_wafer = (
        data[["Wafer_Key", "Recipe_Version", "Lot_ID", "Wafer_ID", "Equipment_Model", "Chamber_ID", "Particle_Defect"]]
        .groupby(["Wafer_Key", "Recipe_Version", "Lot_ID", "Wafer_ID", "Equipment_Model", "Chamber_ID"], as_index=False)
        .agg(Particle_Positive_Site_Rate=("Particle_Defect", "mean"))
    )
    split_wafer["Split"] = np.where(split_wafer["Wafer_Key"].isin(test_groups), "test", "train")
    split_wafer.to_csv(results_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")

    metrics_rows: list[dict] = []
    importance_rows: list[dict] = []
    prediction_frame = data.iloc[test_index][["Wafer_Key", "Recipe_Version", "Lot_ID", "Wafer_ID", "Site_ID", *REGRESSION_TARGETS, "Particle_Defect"]].copy()
    families = ["RandomForest", "XGBoost", "MLP"]
    group_cv = GroupKFold(n_splits=cv_folds)

    for target in REGRESSION_TARGETS:
        y_train = data.iloc[train_index][target].astype(float)
        y_test = data.iloc[test_index][target].astype(float)
        for family in families:
            pipeline = regression_pipeline(family)
            cv_result = cross_validate(
                pipeline,
                X_train,
                y_train,
                groups=groups_train,
                cv=group_cv,
                scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
                n_jobs=1,
                error_score="raise",
            )
            pipeline.fit(X_train, y_train)
            predicted = pipeline.predict(X_test)
            file_name = f"{target}__{family}.joblib"
            joblib.dump(pipeline, models_dir / file_name, compress=3)
            metrics_rows.append(
                {
                    "Task": "Regression",
                    "Target": target,
                    "Model": family,
                    "CV_MAE_Mean": float(-np.mean(cv_result["test_mae"])),
                    "CV_MAE_Std": float(np.std(-cv_result["test_mae"], ddof=1)),
                    "CV_RMSE_Mean": float(-np.mean(cv_result["test_rmse"])),
                    "CV_R2_Mean": float(np.mean(cv_result["test_r2"])),
                    "Test_MAE": float(mean_absolute_error(y_test, predicted)),
                    "Test_RMSE": float(mean_squared_error(y_test, predicted) ** 0.5),
                    "Test_R2": float(r2_score(y_test, predicted)),
                    "Model_File": file_name,
                }
            )
            prediction_frame[f"Pred_{target}__{family}"] = predicted
            importance = permutation_importance(
                pipeline,
                X_test,
                y_test,
                scoring="neg_mean_absolute_error",
                n_repeats=5,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            for feature, mean_value, std_value in zip(FEATURE_COLUMNS, importance.importances_mean, importance.importances_std):
                importance_rows.append(
                    {
                        "Task": "Regression",
                        "Target": target,
                        "Model": family,
                        "Feature": feature,
                        "Permutation_Importance_Mean": float(mean_value),
                        "Permutation_Importance_Std": float(std_value),
                    }
                )

    y_train_cls = data.iloc[train_index]["Particle_Defect"].astype(int)
    y_test_cls = data.iloc[test_index]["Particle_Defect"].astype(int)
    scale_pos_weight = float((y_train_cls == 0).sum() / max(1, (y_train_cls == 1).sum()))
    confusion_rows = []
    for family in families:
        pipeline = classification_pipeline(family, scale_pos_weight)
        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train_cls,
            groups=groups_train,
            cv=group_cv,
            scoring={"balanced_accuracy": "balanced_accuracy", "f1": "f1", "roc_auc": "roc_auc", "average_precision": "average_precision"},
            n_jobs=1,
            error_score="raise",
        )
        pipeline.fit(X_train, y_train_cls)
        probability = pipeline.predict_proba(X_test)[:, 1]
        predicted = (probability >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_cls, predicted, labels=[0, 1]).ravel()
        file_name = f"Particle_Defect__{family}.joblib"
        joblib.dump(pipeline, models_dir / file_name, compress=3)
        metrics_rows.append(
            {
                "Task": "Classification",
                "Target": "Particle_Defect",
                "Model": family,
                "CV_Balanced_Accuracy_Mean": float(np.mean(cv_result["test_balanced_accuracy"])),
                "CV_F1_Mean": float(np.mean(cv_result["test_f1"])),
                "CV_ROC_AUC_Mean": float(np.mean(cv_result["test_roc_auc"])),
                "CV_Average_Precision_Mean": float(np.mean(cv_result["test_average_precision"])),
                "Test_Balanced_Accuracy": float(balanced_accuracy_score(y_test_cls, predicted)),
                "Test_Precision": float(precision_score(y_test_cls, predicted, zero_division=0)),
                "Test_Recall": float(recall_score(y_test_cls, predicted, zero_division=0)),
                "Test_F1": float(f1_score(y_test_cls, predicted, zero_division=0)),
                "Test_ROC_AUC": float(roc_auc_score(y_test_cls, probability)),
                "Test_Average_Precision": float(average_precision_score(y_test_cls, probability)),
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
                {"Model": family, "Actual": 0, "Predicted": 0, "Count": int(tn)},
                {"Model": family, "Actual": 0, "Predicted": 1, "Count": int(fp)},
                {"Model": family, "Actual": 1, "Predicted": 0, "Count": int(fn)},
                {"Model": family, "Actual": 1, "Predicted": 1, "Count": int(tp)},
            ]
        )
        prediction_frame[f"Pred_Particle_Probability__{family}"] = probability
        prediction_frame[f"Pred_Particle_Class__{family}"] = predicted
        importance = permutation_importance(
            pipeline,
            X_test,
            y_test_cls,
            scoring="average_precision",
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        for feature, mean_value, std_value in zip(FEATURE_COLUMNS, importance.importances_mean, importance.importances_std):
            importance_rows.append(
                {
                    "Task": "Classification",
                    "Target": "Particle_Defect",
                    "Model": family,
                    "Feature": feature,
                    "Permutation_Importance_Mean": float(mean_value),
                    "Permutation_Importance_Std": float(std_value),
                }
            )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(results_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(confusion_rows).to_csv(results_dir / "particle_confusion_matrices.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(importance_rows).sort_values(
        ["Target", "Model", "Permutation_Importance_Mean"], ascending=[True, True, False]
    ).to_csv(results_dir / "permutation_importance.csv", index=False, encoding="utf-8-sig")
    prediction_frame.to_csv(results_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    selected_models = {}
    regression_metrics = metrics[metrics["Task"] == "Regression"]
    for target in REGRESSION_TARGETS:
        row = regression_metrics[regression_metrics["Target"] == target].sort_values("CV_MAE_Mean").iloc[0]
        selected_models[target] = {
            "family": row["Model"],
            "file": row["Model_File"],
            "selection_rule": "lowest grouped-CV MAE on training wafers",
        }
    classification_metrics = metrics[metrics["Task"] == "Classification"]
    row = classification_metrics.sort_values(
        ["CV_Average_Precision_Mean", "CV_Balanced_Accuracy_Mean"], ascending=False
    ).iloc[0]
    selected_models["Particle_Defect"] = {
        "family": row["Model"],
        "file": row["Model_File"],
        "selection_rule": "highest grouped-CV average precision, then balanced accuracy",
        "threshold": 0.5,
    }

    base_row = recipe.iloc[0]
    optimal_row = recipe.iloc[-1]
    per_parameter_closure = pd.DataFrame(index=recipe.index)
    for parameter in PARAMETER_COLUMNS:
        denominator = abs(float(optimal_row[parameter]) - float(base_row[parameter]))
        per_parameter_closure[parameter] = 100.0 * (
            1.0 - (recipe[parameter].astype(float) - float(optimal_row[parameter])).abs() / denominator
        )
    formula_check = recipe[["Recipe_Version", "Avg_Param_Closure_to_Optimal_pct", "Changed_Params_This_Rev", "Change_Notes"]].copy()
    formula_check["Recovered_Formula_pct"] = per_parameter_closure.mean(axis=1)
    formula_check["Absolute_Error_pct_point"] = (
        formula_check["Avg_Param_Closure_to_Optimal_pct"] - formula_check["Recovered_Formula_pct"]
    ).abs()
    formula_check.to_csv(results_dir / "score_formula_verification.csv", index=False, encoding="utf-8-sig")

    site_template = (
        data[["Site_ID", "Zone", "Radius_frac", "Angle_deg"]]
        .drop_duplicates("Site_ID")
        .sort_values("Site_ID")
        .replace({np.nan: None})
        .to_dict(orient="records")
    )
    parameter_steps = {}
    parameter_ranges = {}
    for parameter in PARAMETER_COLUMNS:
        unique_values = np.sort(recipe[parameter].astype(float).unique())
        differences = np.diff(unique_values)
        positive_differences = differences[differences > 1e-12]
        data_range = float(unique_values[-1] - unique_values[0])
        fallback = data_range / 10.0 if data_range > 0 else max(abs(float(unique_values[0])) * 0.01, 0.1)
        parameter_steps[parameter] = float(np.min(positive_differences)) if positive_differences.size else fallback
        parameter_ranges[parameter] = [float(unique_values[0]), float(unique_values[-1])]

    particle_variation = data.groupby("Wafer_Key")["Defect_Count"].nunique()
    site_defect_sum = data.groupby("Wafer_Key")["Defect_Count"].sum()
    wafer_keyed = wafer.assign(
        Wafer_Key=wafer[["Recipe_Version", "Lot_ID", "Wafer_ID"]].astype(str).agg("|".join, axis=1)
    ).set_index("Wafer_Key")
    count_reconciliation = wafer_keyed["Total_Defect_Count"].sub(site_defect_sum).abs()
    data_quality = {
        "recipe_rows": int(len(recipe)),
        "wafer_rows": int(len(wafer)),
        "site_rows": int(len(data)),
        "unique_wafers": int(data["Wafer_Key"].nunique()),
        "sites_per_wafer_min": int(data.groupby("Wafer_Key").size().min()),
        "sites_per_wafer_max": int(data.groupby("Wafer_Key").size().max()),
        "site_key_duplicates": int(data.duplicated(["Wafer_Key", "Site_ID"]).sum()),
        "wafers_with_site_varying_defect_count": int((particle_variation > 1).sum()),
        "wafer_count_reconciliation_exact": int((count_reconciliation < 1e-12).sum()),
        "particle_site_positive_rate": float(data["Particle_Defect"].mean()),
        "train_wafer_count": len(train_groups),
        "test_wafer_count": len(test_groups),
        "train_test_wafer_overlap": len(train_groups & test_groups),
    }
    (results_dir / "data_quality_summary.json").write_text(
        json.dumps(data_quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    empirical_spec_limits = {
        "Top_CD_nm": {"lower": None, "upper": float(data.loc[data["Top_CD_Spec_Pass"], "Top_CD_nm"].max()), "note": "Lower failure boundary is not identifiable from observed rows."},
        "Mid_CD_nm": {"lower": None, "upper": float(data.loc[data["Mid_CD_Spec_Pass"], "Mid_CD_nm"].max()), "note": "Lower failure boundary is not identifiable from observed rows."},
        "Bottom_CD_nm": {"lower": float(data.loc[data["Bottom_CD_Spec_Pass"], "Bottom_CD_nm"].min()), "upper": float(data.loc[data["Bottom_CD_Spec_Pass"], "Bottom_CD_nm"].max()), "note": "Empirical pass envelope, not an engineering-authorized specification."},
        "Depth_nm": {"lower": float(data.loc[data["Depth_Spec_Pass"], "Depth_nm"].min()), "upper": float(data.loc[data["Depth_Spec_Pass"], "Depth_nm"].max()), "note": "Empirical pass envelope, not an engineering-authorized specification."},
    }

    metadata = {
        "schema_version": 1,
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "source_files": [xlsx_name for xlsx_name in ["trench recipe full dataset.xlsx", "trench recipe master.csv"]],
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": [*PARAMETER_COLUMNS, *NUMERIC_SITE_COLUMNS],
        "categorical_features": CATEGORICAL_COLUMNS,
        "regression_targets": REGRESSION_TARGETS,
        "classification_target": "Particle_Defect = 1 when site Defect_Count > 0",
        "group_key": "Recipe_Version|Lot_ID|Wafer_ID",
        "selected_models": selected_models,
        "base_recipe": {parameter: float(base_row[parameter]) for parameter in PARAMETER_COLUMNS},
        "optimal_recipe": {parameter: float(optimal_row[parameter]) for parameter in PARAMETER_COLUMNS},
        "recipe_versions": {
            str(row["Recipe_Version"]): {parameter: float(row[parameter]) for parameter in PARAMETER_COLUMNS}
            for _, row in recipe.iterrows()
        },
        "parameter_ranges": parameter_ranges,
        "parameter_steps": parameter_steps,
        "equipment_models": sorted(map(str, data["Equipment_Model"].unique())),
        "chambers": sorted(map(str, data["Chamber_ID"].unique())),
        "site_template": site_template,
        "empirical_spec_limits": empirical_spec_limits,
        "score_formula": "round(mean_p(100 * (1 - abs(x_p - Rev15_p) / abs(Base_p - Rev15_p))), 1)",
        "simulator_quality_index": "0.70 * empirical predicted CD/Depth spec-pass rate + 0.30 * predicted particle-free site rate",
        "limitations": [
            "Only 16 highly correlated recipe combinations are observed.",
            "Process and Layer columns are absent and therefore are not model features.",
            "Group split prevents same-wafer leakage but does not test unseen recipe-version generalization.",
            "Feature importance is associative, not causal, because recipe parameters changed sequentially and are collinear.",
            "Empirical spec limits are inferred from pass flags and are not a substitute for engineering-controlled specifications.",
        ],
        "training_seconds": round(time.time() - started_at, 1),
    }
    (models_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"selected_models": selected_models, "data_quality": data_quality, "training_seconds": metadata["training_seconds"]}, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    recipe, wafer, data = load_and_prepare(args.xlsx.resolve(), args.recipe_csv.resolve())
    fit_models(recipe, wafer, data, output_dir, args.cv_folds)


if __name__ == "__main__":
    main()
