import os
import csv
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

CLINICAL_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "HVSMR",
    "hvsmr_clinical.csv"
)

REFERENCE_FILE = os.path.join(
    BACKEND_DIR,
    "reference",
    "reference_cohort.csv"
)

MODEL_DIR = os.path.join(
    BACKEND_DIR,
    "model",
    "septal_defect"
)

VSD_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "vsd_model.pkl"
)

ASD_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "asd_model.pkl"
)


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# CSV HELPERS
# ============================================================

def read_csv(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        return list(csv.DictReader(file))


def normalize_patient_id(value):

    value = str(value).strip().lower()

    if not value:
        return ""

    if value.startswith("pat"):
        return value

    return f"pat{value}"


def label_value(value):

    return 1 if str(value).strip().upper() == "X" else 0


# ============================================================
# LOAD CLINICAL LABELS
# ============================================================

def load_clinical_labels():

    rows = read_csv(CLINICAL_FILE)

    clinical = {}

    for row in rows:

        patient_id = normalize_patient_id(
            row.get("Pat", "")
        )

        if not patient_id:
            continue

        clinical[patient_id] = {
            "VSD": label_value(row.get("VSD", "")),
            "ASD": label_value(row.get("ASD", ""))
        }

    return clinical


# ============================================================
# LOAD REFERENCE COHORT
# ============================================================

def load_reference_cohort():

    rows = read_csv(REFERENCE_FILE)

    reference = []

    for row in rows:

        patient_id = normalize_patient_id(
            row.get("patient", "")
        )

        if not patient_id:
            continue

        values = []

        valid = True

        for feature in FEATURES:

            try:
                value = float(row[feature])
            except (
                ValueError,
                KeyError,
                TypeError
            ):
                valid = False
                break

            if not np.isfinite(value):
                valid = False
                break

            values.append(value)

        if not valid:
            continue

        reference.append({
            "patient": patient_id,
            "features": np.asarray(
                values,
                dtype=np.float32
            )
        })

    return reference


# ============================================================
# IDENTIFY KNOWN DATASET PATIENT
# ============================================================

def identify_reference_patient(features):

    """
    Try to identify whether the supplied feature vector
    corresponds to a patient already present in the
    HVSMR reference cohort.

    This is ONLY used for dataset evaluation/auditing.

    It is NOT used as a clinical diagnostic method.
    """

    if not os.path.exists(REFERENCE_FILE):
        return None

    try:
        reference_rows = load_reference_cohort()
    except Exception:
        return None

    values = []

    for feature in FEATURES:

        if feature not in features:
            return None

        try:
            value = float(features[feature])
        except (
            ValueError,
            TypeError
        ):
            return None

        if not np.isfinite(value):
            return None

        values.append(value)

    target = np.asarray(
        values,
        dtype=np.float32
    )

    for row in reference_rows:

        reference_features = row["features"]

        if len(reference_features) != len(target):
            continue

        # Exact/near-exact comparison.
        if np.allclose(
            target,
            reference_features,
            rtol=1e-4,
            atol=1e-4
        ):

            return row["patient"]

    return None


# ============================================================
# BUILD TRAINING DATASET
# ============================================================

def build_training_dataset():

    print("\n" + "=" * 70)
    print("BUILDING SEPTAL DEFECT TRAINING DATASET")
    print("=" * 70)

    clinical_rows = read_csv(CLINICAL_FILE)
    reference_rows = read_csv(REFERENCE_FILE)

    clinical = {}

    for row in clinical_rows:

        patient_id = normalize_patient_id(
            row.get("Pat", "")
        )

        if not patient_id:
            continue

        clinical[patient_id] = {
            "VSD": label_value(row.get("VSD", "")),
            "ASD": label_value(row.get("ASD", ""))
        }

    dataset = []

    for row in reference_rows:

        patient_id = normalize_patient_id(
            row.get("patient", "")
        )

        if patient_id not in clinical:
            continue

        values = []

        valid = True

        for feature in FEATURES:

            try:
                value = float(row[feature])
            except (
                ValueError,
                KeyError,
                TypeError
            ):
                valid = False
                break

            if not np.isfinite(value):
                valid = False
                break

            values.append(value)

        if not valid:
            continue

        dataset.append({
            "patient": patient_id,
            "features": values,
            "VSD": clinical[patient_id]["VSD"],
            "ASD": clinical[patient_id]["ASD"]
        })

    if not dataset:

        raise RuntimeError(
            "No valid labeled patients were found."
        )

    # ========================================================
    # DATASET STATISTICS
    # ========================================================

    vsd_positive = sum(
        row["VSD"]
        for row in dataset
    )

    asd_positive = sum(
        row["ASD"]
        for row in dataset
    )

    print(
        f"\nLabeled patients used: {len(dataset)}"
    )

    print(
        f"VSD positive: {vsd_positive}"
    )

    print(
        f"VSD negative: {len(dataset) - vsd_positive}"
    )

    print(
        f"ASD positive: {asd_positive}"
    )

    print(
        f"ASD negative: {len(dataset) - asd_positive}"
    )

    # ========================================================
    # VERIFY PAT 9
    # ========================================================

    pat9 = next(
        (
            row
            for row in dataset
            if row["patient"] == "pat9"
        ),
        None
    )

    if pat9 is not None:

        print("\n" + "-" * 70)
        print("PAT 9 DATASET CHECK")
        print("-" * 70)

        print(
            f"Pat 9 VSD label: {pat9['VSD']}"
        )

        print(
            f"Pat 9 ASD label: {pat9['ASD']}"
        )

        if pat9["VSD"] == 1:
            print(
                "✓ Pat 9 is correctly labeled VSD POSITIVE."
            )
        else:
            print(
                "WARNING: Pat 9 is not labeled VSD positive."
            )

    # ========================================================
    # SAVE TRAINING DATASET
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    training_file = os.path.join(
        MODEL_DIR,
        "septal_defect_training.csv"
    )

    with open(
        training_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = (
            ["patient"]
            + FEATURES
            + ["VSD", "ASD"]
        )

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in dataset:

            output = {
                "patient": row["patient"],
                "VSD": row["VSD"],
                "ASD": row["ASD"]
            }

            for index, feature in enumerate(FEATURES):

                output[feature] = (
                    row["features"][index]
                )

            writer.writerow(output)

    print(
        "\nTraining dataset saved:"
    )

    print(training_file)

    return dataset


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models():

    dataset = build_training_dataset()

    X = np.asarray(
        [
            row["features"]
            for row in dataset
        ],
        dtype=np.float32
    )

    y_vsd = np.asarray(
        [
            row["VSD"]
            for row in dataset
        ],
        dtype=np.int64
    )

    y_asd = np.asarray(
        [
            row["ASD"]
            for row in dataset
        ],
        dtype=np.int64
    )

    if len(np.unique(y_vsd)) < 2:

        raise RuntimeError(
            "VSD training data contains only one class."
        )

    if len(np.unique(y_asd)) < 2:

        raise RuntimeError(
            "ASD training data contains only one class."
        )

    # ========================================================
    # VSD MODEL
    # ========================================================

    print("\nTraining VSD classifier...")

    vsd_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=1
    )

    vsd_model.fit(
        X,
        y_vsd
    )

    print(
        "VSD classifier trained."
    )

    # ========================================================
    # ASD MODEL
    # ========================================================

    print("\nTraining ASD classifier...")

    asd_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=1
    )

    asd_model.fit(
        X,
        y_asd
    )

    print(
        "ASD classifier trained."
    )

    # ========================================================
    # SAVE MODELS
    # ========================================================

    with open(
        VSD_MODEL_FILE,
        "wb"
    ) as file:

        pickle.dump(
            vsd_model,
            file
        )

    with open(
        ASD_MODEL_FILE,
        "wb"
    ) as file:

        pickle.dump(
            asd_model,
            file
        )

    print("\nVSD model saved:")
    print(VSD_MODEL_FILE)

    print("\nASD model saved:")
    print(ASD_MODEL_FILE)

    return (
        vsd_model,
        asd_model
    )


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    if not os.path.exists(
        VSD_MODEL_FILE
    ):

        raise FileNotFoundError(
            f"VSD model not found: "
            f"{VSD_MODEL_FILE}"
        )

    if not os.path.exists(
        ASD_MODEL_FILE
    ):

        raise FileNotFoundError(
            f"ASD model not found: "
            f"{ASD_MODEL_FILE}"
        )

    with open(
        VSD_MODEL_FILE,
        "rb"
    ) as file:

        vsd_model = pickle.load(file)

    with open(
        ASD_MODEL_FILE,
        "rb"
    ) as file:

        asd_model = pickle.load(file)

    return (
        vsd_model,
        asd_model
    )


# ============================================================
# PREDICT SEPTAL DEFECT
# ============================================================

def predict_septal_defect(features):

    """
    Performs ML screening.

    IMPORTANT:
    The ML prediction is kept separate from the
    known HVSMR clinical ground truth.

    For known dataset patients, the response also contains
    a ground_truth section for evaluation.
    """

    vsd_model, asd_model = load_models()

    values = []

    for feature in FEATURES:

        if feature not in features:

            raise KeyError(
                f"Required feature missing: {feature}"
            )

        try:
            value = float(
                features[feature]
            )

        except (
            ValueError,
            TypeError
        ):

            raise ValueError(
                f"Invalid value for feature: "
                f"{feature}"
            )

        if not np.isfinite(value):

            raise ValueError(
                f"Non-finite value for feature: "
                f"{feature}"
            )

        values.append(value)

    X = np.asarray(
        [values],
        dtype=np.float32
    )

    # ========================================================
    # ML PREDICTIONS
    # ========================================================

    vsd_prediction = int(
        vsd_model.predict(X)[0]
    )

    asd_prediction = int(
        asd_model.predict(X)[0]
    )

    # ========================================================
    # ML PROBABILITIES
    # ========================================================

    vsd_probability = None
    asd_probability = None

    if hasattr(
        vsd_model,
        "predict_proba"
    ):

        probabilities = (
            vsd_model.predict_proba(X)[0]
        )

        if len(probabilities) > 1:

            vsd_probability = float(
                probabilities[1]
            )

    if hasattr(
        asd_model,
        "predict_proba"
    ):

        probabilities = (
            asd_model.predict_proba(X)[0]
        )

        if len(probabilities) > 1:

            asd_probability = float(
                probabilities[1]
            )

    # ========================================================
    # ML STATUS
    # ========================================================

    if vsd_prediction == 1:

        vsd_status = (
            "VSD suspected"
        )

    else:

        vsd_status = (
            "VSD not detected"
        )

    if asd_prediction == 1:

        asd_status = (
            "ASD suspected"
        )

    else:

        asd_status = (
            "ASD not detected"
        )

    if (
        vsd_prediction == 1
        or asd_prediction == 1
    ):

        overall = (
            "Septal defect suspected"
        )

    else:

        overall = (
            "No septal defect detected"
        )

    # ========================================================
    # DATASET GROUND-TRUTH AUDIT
    # ========================================================

    ground_truth = None

    known_patient = identify_reference_patient(
        features
    )

    if known_patient is not None:

        clinical = load_clinical_labels()

        if known_patient in clinical:

            clinical_vsd = clinical[
                known_patient
            ]["VSD"]

            clinical_asd = clinical[
                known_patient
            ]["ASD"]

            ground_truth = {

                "patient": known_patient,

                "vsd": {
                    "present": bool(
                        clinical_vsd
                    ),
                    "status": (
                        "VSD present"
                        if clinical_vsd == 1
                        else "VSD absent"
                    )
                },

                "asd": {
                    "present": bool(
                        clinical_asd
                    ),
                    "status": (
                        "ASD present"
                        if clinical_asd == 1
                        else "ASD absent"
                    )
                }
            }

            # =================================================
            # MODEL EVALUATION
            # =================================================

            ground_truth["model_evaluation"] = {

                "vsd_correct": (
                    bool(vsd_prediction)
                    == bool(clinical_vsd)
                ),

                "asd_correct": (
                    bool(asd_prediction)
                    == bool(clinical_asd)
                )
            }

            # Special diagnostic logging
            print("\n" + "=" * 70)
            print(
                "SEPTAL DEFECT DATASET AUDIT"
            )
            print("=" * 70)

            print(
                f"Known patient: {known_patient}"
            )

            print(
                f"Clinical VSD: "
                f"{'PRESENT' if clinical_vsd else 'ABSENT'}"
            )

            print(
                f"ML VSD prediction: "
                f"{'POSITIVE' if vsd_prediction else 'NEGATIVE'}"
            )

            if vsd_probability is not None:

                print(
                    f"ML VSD probability: "
                    f"{vsd_probability:.4f}"
                )

            print(
                f"Clinical ASD: "
                f"{'PRESENT' if clinical_asd else 'ABSENT'}"
            )

            print(
                f"ML ASD prediction: "
                f"{'POSITIVE' if asd_prediction else 'NEGATIVE'}"
            )

            if asd_probability is not None:

                print(
                    f"ML ASD probability: "
                    f"{asd_probability:.4f}"
                )

            print("=" * 70)

    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "vsd": {

            "detected": bool(
                vsd_prediction
            ),

            "status": vsd_status,

            "probability": (
                vsd_probability
            )
        },

        "asd": {

            "detected": bool(
                asd_prediction
            ),

            "status": asd_status,

            "probability": (
                asd_probability
            )
        },

        "overall": overall,

        "ground_truth": ground_truth,

        "note": (
            "Automated screening based on "
            "MRI-derived cardiac features. "
            "This is not a clinical diagnosis."
        )
    }

    return result


# ============================================================
# TRAIN WHEN RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    train_models()