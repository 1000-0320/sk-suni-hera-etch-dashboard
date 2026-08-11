from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_validate

from train_models import (
    DEFAULT_RECIPE_CSV,
    DEFAULT_XLSX,
    FEATURE_COLUMNS,
    REGRESSION_TARGETS,
    classification_pipeline,
    load_and_prepare,
    regression_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Secondary generalization check with whole recipe versions held out.")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--recipe-csv", type=Path, default=DEFAULT_RECIPE_CSV)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "recipe_holdout_cv.csv")
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()

    _, _, data = load_and_prepare(args.xlsx.resolve(), args.recipe_csv.resolve())
    X = data[FEATURE_COLUMNS]
    recipe_groups = data["Recipe_Version"]
    cv = GroupKFold(n_splits=args.folds)
    families = ["RandomForest", "XGBoost", "MLP"]
    rows = []
    for target in REGRESSION_TARGETS:
        y = data[target].astype(float)
        for family in families:
            scores = cross_validate(
                regression_pipeline(family),
                X,
                y,
                groups=recipe_groups,
                cv=cv,
                scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
                n_jobs=1,
                error_score="raise",
            )
            rows.append(
                {
                    "Task": "Regression",
                    "Target": target,
                    "Model": family,
                    "Recipe_Holdout_CV_MAE_Mean": float(-np.mean(scores["test_mae"])),
                    "Recipe_Holdout_CV_RMSE_Mean": float(-np.mean(scores["test_rmse"])),
                    "Recipe_Holdout_CV_R2_Mean": float(np.mean(scores["test_r2"])),
                }
            )
    y_cls = data["Particle_Defect"].astype(int)
    scale_pos_weight = float((y_cls == 0).sum() / max(1, (y_cls == 1).sum()))
    for family in families:
        scores = cross_validate(
            classification_pipeline(family, scale_pos_weight),
            X,
            y_cls,
            groups=recipe_groups,
            cv=cv,
            scoring={"balanced_accuracy": "balanced_accuracy", "f1": "f1", "roc_auc": "roc_auc", "average_precision": "average_precision"},
            n_jobs=1,
            error_score="raise",
        )
        rows.append(
            {
                "Task": "Classification",
                "Target": "Particle_Defect",
                "Model": family,
                "Recipe_Holdout_CV_Balanced_Accuracy_Mean": float(np.mean(scores["test_balanced_accuracy"])),
                "Recipe_Holdout_CV_F1_Mean": float(np.mean(scores["test_f1"])),
                "Recipe_Holdout_CV_ROC_AUC_Mean": float(np.mean(scores["test_roc_auc"])),
                "Recipe_Holdout_CV_Average_Precision_Mean": float(np.mean(scores["test_average_precision"])),
            }
        )
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
