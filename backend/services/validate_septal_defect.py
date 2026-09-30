import os
import csv
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BACKEND_DIR = os.path.dirname(
    CURRENT_DIR
)

MODEL_DIR = os.path.join(
    BACKEND_DIR,
    "model",
    "septal_defect"
)

TRAINING_FILE = os.path.join(
    MODEL_DIR,
    "septal_defect_training.csv"
)


FEATURES = [
    "left_ventricle_volume_ml",
    "right_ventricle_volume_ml",
    "left_atrium_volume_ml",
    "right_atrium_volume_ml",

    "left_ventricle_length_mm",
    "left_ventricle_width_mm",
    "left_ventricle_depth_mm",

    "right_ventricle_length_mm",
    "right_ventricle_width_mm",
    "right_ventricle_depth_mm",

    "left_atrium_length_mm",
    "left_atrium_width_mm",
    "left_atrium_depth_mm",

    "right_atrium_length_mm",
    "right_atrium_width_mm",
    "right_atrium_depth_mm",

    "lv_rv_volume_ratio",
    "lv_rv_volume_difference_percent"
]


def load_training_data():

    if not os.path.exists(TRAINING_FILE):

        raise FileNotFoundError(
            f"Training file not found:\n{TRAINING_FILE}"
        )

    with open(
        TRAINING_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    X = []

    y_vsd = []

    y_asd = []

    patients = []

    for row in rows:

        X.append([
            float(row[feature])
            for feature in FEATURES
        ])

        y_vsd.append(
            int(row["VSD"])
        )

        y_asd.append(
            int(row["ASD"])
        )

        patients.append(
            row["patient"]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y_vsd, dtype=np.int64),
        np.asarray(y_asd, dtype=np.int64),
        patients
    )


def evaluate_model(
    X,
    y,
    name
):

    print("\n" + "=" * 70)
    print(f"{name} VALIDATION")
    print("=" * 70)

    class_counts = np.bincount(y)

    minimum_class_count = int(
        class_counts.min()
    )

    n_splits = min(
        5,
        minimum_class_count
    )

    if n_splits < 2:

        raise RuntimeError(
            f"Not enough samples for {name} "
            "cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        max_features="sqrt"
    )

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    print(
        f"\nSamples: {len(y)}"
    )

    print(
        f"Positive: {int(y.sum())}"
    )

    print(
        f"Negative: {int(len(y) - y.sum())}"
    )

    print(
        f"\nAccuracy:    {accuracy:.4f}"
    )

    print(
        f"Precision:   {precision:.4f}"
    )

    print(
        f"Sensitivity: {recall:.4f}"
    )

    print(
        f"Specificity: {specificity:.4f}"
    )

    print(
        f"F1 Score:    {f1:.4f}"
    )

    print("\nConfusion Matrix:")
    print(
        f"TN={tn}  FP={fp}"
    )
    print(
        f"FN={fn}  TP={tp}"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1
    }


def main():

    print("=" * 70)
    print("SEPTAL DEFECT MODEL VALIDATION")
    print("=" * 70)

    (
        X,
        y_vsd,
        y_asd,
        patients
    ) = load_training_data()

    print(
        f"\nLoaded {len(patients)} labeled patients."
    )

    vsd_results = evaluate_model(
        X,
        y_vsd,
        "VSD"
    )

    asd_results = evaluate_model(
        X,
        y_asd,
        "ASD"
    )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()