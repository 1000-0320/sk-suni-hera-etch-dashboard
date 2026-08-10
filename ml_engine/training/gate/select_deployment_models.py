from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from simulator_core import REGRESSION_TARGETS


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "results" / "recipe_holdout_cv.csv"
METADATA_PATH = ROOT / "models" / "metadata.json"


def main() -> None:
    metrics = pd.read_csv(METRICS_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata["wafer_group_cv_selected_models"] = metadata["selected_models"]
    selected = {}
    for target in REGRESSION_TARGETS:
        row = metrics[(metrics["Task"] == "Regression") & (metrics["Target"] == target)].sort_values(
            "Recipe_Holdout_CV_MAE_Mean"
        ).iloc[0]
        family = str(row["Model"])
        selected[target] = {
            "family": family,
            "file": f"{target}__{family}.joblib",
            "selection_rule": "lowest whole-recipe-holdout CV MAE; chosen for simulator robustness",
        }
    row = metrics[metrics["Task"] == "Classification"].sort_values(
        ["Recipe_Holdout_CV_Average_Precision_Mean", "Recipe_Holdout_CV_Balanced_Accuracy_Mean"],
        ascending=False,
    ).iloc[0]
    family = str(row["Model"])
    selected["Particle_Defect"] = {
        "family": family,
        "file": f"Particle_Defect__{family}.joblib",
        "selection_rule": "highest whole-recipe-holdout CV average precision, then balanced accuracy",
        "threshold": 0.5,
    }
    metadata["selected_models"] = selected
    metadata["deployment_selection_note"] = (
        "Deployment favors whole-recipe holdout robustness over the easier wafer-only CV. "
        "MLP is retained for comparison but excluded from simulator deployment because recipe-holdout performance collapsed."
    )
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
