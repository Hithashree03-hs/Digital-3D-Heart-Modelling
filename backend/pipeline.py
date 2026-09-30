import os
import csv
import json
import re

import numpy as np

from backend.services.segmentation import predict_volume
from backend.services.reconstruction import reconstruct_3d
from backend.services.septal_defect import predict_septal_defect


# ============================================================
# BACKEND PATHS
# ============================================================

BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    BACKEND_DIR
)

REFERENCE_FILE = os.path.join(
    BACKEND_DIR,
    "reference",
    "reference_cohort.csv"
)

CLINICAL_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "HVSMR",
    "hvsmr_clinical.csv"
)

OUTPUT_DIR = os.path.join(
    BACKEND_DIR,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# ANATOMICAL LABELS
# ============================================================

LABEL_NAMES = {
    1: "Left Ventricle",
    2: "Right Ventricle",
    3: "Left Atrium",
    4: "Right Atrium",
    5: "Aorta",
    6: "Pulmonary Artery",
    7: "Superior Vena Cava",
    8: "Inferior Vena Cava"
}


# ============================================================
# FEATURES USED FOR MORPHOLOGICAL ANALYSIS
# ============================================================

PRIMARY_FEATURES = [
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
# LOAD CSV
# ============================================================

def load_csv(path):
    """
    Load a CSV file and return a list of dictionaries.
    """

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

        return list(
            csv.DictReader(file)
        )


# ============================================================
# PATIENT ID HELPERS
# ============================================================

def normalize_patient_id(value):
    """
    Convert patient identifiers to a standard form.

    Examples:

        9       -> pat9
        Pat9    -> pat9
        pat9    -> pat9
        PAT25   -> pat25
    """

    value = str(value).strip().lower()

    if not value:
        return ""

    if value.startswith("pat"):
        return value

    return f"pat{value}"


def extract_patient_id_from_filename(input_mri_path):
    """
    Extract the HVSMR patient ID from an MRI filename.

    Supported examples:

        pat9.nii.gz
        pat9_cropped.nii.gz
        Pat25_cropped.nii.gz
        patient_pat31.nii.gz

    Returns:

        pat9
        pat25
        pat31

    Returns None if no PatXX identifier is present.
    """

    filename = os.path.basename(
        input_mri_path
    ).lower()

    match = re.search(
        r"pat(\d+)",
        filename
    )

    if match is None:
        return None

    return normalize_patient_id(
        f"pat{match.group(1)}"
    )


# ============================================================
# HVSMR CLINICAL LABELS
# ============================================================

def load_hvsmr_clinical_labels():
    """
    Load VSD and ASD ground-truth labels from
    hvsmr_clinical.csv.

    X = condition present
    blank = condition absent
    """

    rows = load_csv(
        CLINICAL_FILE
    )

    clinical = {}

    for row in rows:

        patient_id = normalize_patient_id(
            row.get("Pat", "")
        )

        if not patient_id:
            continue

        clinical[patient_id] = {

            "VSD": (
                str(
                    row.get("VSD", "")
                ).strip().upper()
                == "X"
            ),

            "ASD": (
                str(
                    row.get("ASD", "")
                ).strip().upper()
                == "X"
            )
        }

    return clinical


def get_hvsmr_ground_truth(patient_id):
    """
    Return the known HVSMR clinical diagnosis for a patient.

    Returns None when the patient does not exist in the
    local clinical CSV.

    This is dataset ground truth, not a new clinical diagnosis.
    """

    if not patient_id:
        return None

    patient_id = normalize_patient_id(
        patient_id
    )

    clinical = load_hvsmr_clinical_labels()

    if patient_id not in clinical:
        return None

    labels = clinical[
        patient_id
    ]

    vsd_present = bool(
        labels["VSD"]
    )

    asd_present = bool(
        labels["ASD"]
    )

    return {

        "available": True,

        "patient_id": patient_id,

        "vsd": {

            "detected": vsd_present,

            "status": (
                "VSD present"
                if vsd_present
                else "VSD absent"
            )
        },

        "asd": {

            "detected": asd_present,

            "status": (
                "ASD present"
                if asd_present
                else "ASD absent"
            )
        }
    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    prediction,
    spacing
):
    """
    Extract physical cardiac measurements from
    a 3D segmentation volume.

    Measurements are derived from the supplied
    segmentation volume and MRI voxel spacing.
    """

    prediction = np.asarray(
        prediction
    )

    if prediction.ndim != 3:

        raise ValueError(
            "Prediction must be a 3D volume. "
            f"Received shape: {prediction.shape}"
        )

    spacing = np.asarray(
        spacing,
        dtype=float
    )

    if spacing.size < 3:

        raise ValueError(
            "Voxel spacing must contain "
            "three spatial dimensions."
        )

    spacing = spacing[:3]

    if not np.all(
        np.isfinite(spacing)
    ):

        raise ValueError(
            f"Invalid voxel spacing: {spacing}"
        )

    if np.any(
        spacing <= 0
    ):

        raise ValueError(
            f"Voxel spacing must be positive: {spacing}"
        )

    voxel_volume_ml = (
        np.prod(spacing) / 1000.0
    )

    features = {}

    structure_volumes = {}

    for label, structure_name in LABEL_NAMES.items():

        coords = np.argwhere(
            prediction == label
        )

        if coords.size == 0:

            volume_ml = 0.0

            dimensions_mm = (
                0.0,
                0.0,
                0.0
            )

        else:

            voxel_count = len(
                coords
            )

            volume_ml = (
                voxel_count *
                voxel_volume_ml
            )

            min_coords = coords.min(
                axis=0
            )

            max_coords = coords.max(
                axis=0
            )

            dimensions_voxels = (
                max_coords -
                min_coords +
                1
            )

            dimensions_mm = (
                dimensions_voxels *
                spacing
            )

        structure_volumes[
            label
        ] = volume_ml

        prefix = (
            structure_name
            .lower()
            .replace(" ", "_")
        )

        features[
            f"{prefix}_volume_ml"
        ] = round(
            float(volume_ml),
            3
        )

        features[
            f"{prefix}_length_mm"
        ] = round(
            float(dimensions_mm[0]),
            3
        )

        features[
            f"{prefix}_width_mm"
        ] = round(
            float(dimensions_mm[1]),
            3
        )

        features[
            f"{prefix}_depth_mm"
        ] = round(
            float(dimensions_mm[2]),
            3
        )

    # ========================================================
    # LV / RV RELATIONSHIP
    # ========================================================

    lv_volume = structure_volumes[1]
    rv_volume = structure_volumes[2]

    if rv_volume > 0:

        lv_rv_ratio = (
            lv_volume /
            rv_volume
        )

        mean_ventricular_volume = (
            lv_volume +
            rv_volume
        ) / 2.0

        if mean_ventricular_volume > 0:

            lv_rv_difference = (
                abs(
                    lv_volume -
                    rv_volume
                )
                /
                mean_ventricular_volume
            ) * 100.0

        else:

            lv_rv_difference = 0.0

    else:

        lv_rv_ratio = 0.0
        lv_rv_difference = 0.0

    features[
        "lv_rv_volume_ratio"
    ] = round(
        float(lv_rv_ratio),
        3
    )

    features[
        "lv_rv_volume_difference_percent"
    ] = round(
        float(lv_rv_difference),
        3
    )

    return features


# ============================================================
# REFERENCE COHORT STATISTICS
# ============================================================

def build_reference_statistics():
    """
    Build dataset-derived reference intervals from
    backend/reference/reference_cohort.csv.

    For every feature:

        Q1
        Median
        Q3
        IQR
        Lower bound
        Upper bound

    Interval:

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
    """

    reference_rows = load_csv(
        REFERENCE_FILE
    )

    if not reference_rows:

        raise ValueError(
            "Reference cohort is empty: "
            f"{REFERENCE_FILE}"
        )

    reference_stats = {}

    print("\n" + "=" * 70)
    print("BUILDING REFERENCE COHORT STATISTICS")
    print("=" * 70)

    print(
        "Reference rows:",
        len(reference_rows)
    )

    for feature in PRIMARY_FEATURES:

        values = []

        for row in reference_rows:

            try:

                raw_value = row.get(
                    feature
                )

                if (
                    raw_value is None
                    or raw_value == ""
                ):

                    continue

                value = float(
                    raw_value
                )

                if (
                    np.isfinite(value)
                    and value > 0
                ):

                    values.append(
                        value
                    )

            except (
                ValueError,
                TypeError,
                KeyError
            ):

                continue

        if len(values) < 5:

            print(
                f"Skipping {feature}: "
                f"only {len(values)} valid values"
            )

            continue

        values = np.asarray(
            values,
            dtype=float
        )

        q1 = np.percentile(
            values,
            25
        )

        median = np.percentile(
            values,
            50
        )

        q3 = np.percentile(
            values,
            75
        )

        iqr = q3 - q1

        lower = max(
            0.0,
            q1 - (
                1.5 * iqr
            )
        )

        upper = (
            q3 +
            (
                1.5 * iqr
            )
        )

        reference_stats[
            feature
        ] = {

            "q1": float(q1),

            "median": float(
                median
            ),

            "q3": float(q3),

            "iqr": float(
                iqr
            ),

            "lower": float(
                lower
            ),

            "upper": float(
                upper
            ),

            "sample_count": int(
                len(values)
            )
        }

    if not reference_stats:

        raise ValueError(
            "No usable reference statistics "
            "could be calculated from "
            f"{REFERENCE_FILE}"
        )

    print(
        "Reference features:",
        len(reference_stats)
    )

    print(
        "Reference statistics successfully built."
    )

    return reference_stats


# ============================================================
# ABNORMALITY ANALYSIS
# ============================================================

def analyze_abnormality(
    features,
    reference_stats
):
    """
    Compare patient measurements with
    the project reference cohort.

    This is morphological screening,
    not a clinical diagnosis.
    """

    if reference_stats is None:

        raise ValueError(
            "Reference statistics are None. "
            "build_reference_statistics() "
            "must return a dictionary."
        )

    if not isinstance(
        reference_stats,
        dict
    ):

        raise TypeError(
            "reference_stats must be a dictionary."
        )

    if features is None:

        raise ValueError(
            "Patient features are None."
        )

    feature_results = []

    mild_count = 0
    marked_count = 0

    for feature in PRIMARY_FEATURES:

        if feature not in reference_stats:
            continue

        if feature not in features:
            continue

        try:

            patient_value = float(
                features[feature]
            )

        except (
            ValueError,
            TypeError
        ):

            continue

        stats = reference_stats[
            feature
        ]

        lower = float(
            stats["lower"]
        )

        upper = float(
            stats["upper"]
        )

        median = float(
            stats["median"]
        )

        q1 = float(
            stats["q1"]
        )

        q3 = float(
            stats["q3"]
        )

        iqr = float(
            stats["iqr"]
        )

        # ====================================================
        # DETERMINE DEVIATION
        # ====================================================

        if patient_value < lower:

            deviation = "Low"

        elif patient_value > upper:

            deviation = "High"

        else:

            deviation = (
                "Within reference range"
            )

        # ====================================================
        # DISTANCE FROM MEDIAN
        # ====================================================

        if median > 0:

            percent_from_median = (
                abs(
                    patient_value -
                    median
                )
                /
                median
            ) * 100.0

        else:

            percent_from_median = 0.0

        # ====================================================
        # SEVERITY
        # ====================================================

        if (
            deviation ==
            "Within reference range"
        ):

            severity = (
                "Normal range"
            )

        elif percent_from_median >= 75:

            severity = (
                "Marked deviation"
            )

            marked_count += 1

        else:

            severity = (
                "Mild deviation"
            )

            mild_count += 1

        feature_results.append({

            "feature":
                feature,

            "patient_value":
                round(
                    patient_value,
                    3
                ),

            "reference_median":
                round(
                    median,
                    3
                ),

            "reference_q1":
                round(
                    q1,
                    3
                ),

            "reference_q3":
                round(
                    q3,
                    3
                ),

            "reference_lower":
                round(
                    lower,
                    3
                ),

            "reference_upper":
                round(
                    upper,
                    3
                ),

            "reference_iqr":
                round(
                    iqr,
                    3
                ),

            "deviation":
                deviation,

            "percent_from_median":
                round(
                    percent_from_median,
                    2
                ),

            "severity":
                severity
        })

    # ========================================================
    # OVERALL ASSESSMENT
    # ========================================================

    if marked_count >= 2:

        overall = (
            "Multiple marked "
            "morphological deviations detected"
        )

    elif (
        marked_count == 1
        or mild_count >= 3
    ):

        overall = (
            "Morphological deviations detected"
        )

    else:

        overall = (
            "No marked morphological "
            "deviation detected"
        )

    return {

        "features":
            feature_results,

        "mild_deviations":
            mild_count,

        "marked_deviations":
            marked_count,

        "overall_assessment":
            overall
    }


# ============================================================
# SAVE FEATURES CSV
# ============================================================

def save_features_csv(
    features,
    output_path
):
    """
    Save extracted cardiac features.
    """

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Feature",
            "Value"
        ])

        for name, value in features.items():

            writer.writerow([
                name,
                value
            ])


# ============================================================
# SAVE ABNORMALITY CSV
# ============================================================

def save_abnormality_csv(
    abnormality,
    output_path
):
    """
    Save abnormality analysis results.
    """

    feature_results = abnormality.get(
        "features",
        []
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        if not feature_results:
            return

        fieldnames = list(
            feature_results[0].keys()
        )

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            feature_results
        )


# ============================================================
# BUILD FINAL SEPTAL DEFECT RESULT
# ============================================================

def build_septal_defect_result(
    ai_result,
    clinical_ground_truth
):
    """
    Combine AI screening and known HVSMR ground truth.

    IMPORTANT:

    For known HVSMR patients, the clinical diagnosis is used
    as the dataset ground truth for the project visualization.

    The original AI prediction is preserved separately.

    For an unknown patient, the AI result is used directly.
    """

    if ai_result is None:

        raise ValueError(
            "AI septal defect result is None."
        )

    result = dict(
        ai_result
    )

    # ========================================================
    # UNKNOWN / NON-HVSMR PATIENT
    # ========================================================

    if clinical_ground_truth is None:

        result[
            "screening_source"
        ] = (
            "AI feature-based screening"
        )

        result[
            "ground_truth_available"
        ] = False

        return result

    # ========================================================
    # KNOWN HVSMR PATIENT
    # ========================================================

    result[
        "ai_prediction"
    ] = {

        "vsd":
            ai_result.get(
                "vsd",
                {}
            ),

        "asd":
            ai_result.get(
                "asd",
                {}
            ),

        "overall":
            ai_result.get(
                "overall",
                "Result unavailable"
            )
    }

    clinical_vsd = bool(
        clinical_ground_truth[
            "vsd"
        ][
            "detected"
        ]
    )

    clinical_asd = bool(
        clinical_ground_truth[
            "asd"
        ][
            "detected"
        ]
    )

    ai_vsd = bool(
        ai_result
        .get(
            "vsd",
            {}
        )
        .get(
            "detected",
            False
        )
    )

    ai_asd = bool(
        ai_result
        .get(
            "asd",
            {}
        )
        .get(
            "detected",
            False
        )
    )

    # ========================================================
    # VSD — USE DATASET GROUND TRUTH FOR KNOWN PATIENT
    # ========================================================

    if clinical_vsd:

        result["vsd"] = {

            "detected": True,

            "status":
                "VSD detected",

            "source":
                "HVSMR clinical ground truth"
        }

    else:

        result["vsd"] = {

            "detected": False,

            "status":
                "VSD not detected",

            "source":
                "HVSMR clinical ground truth"
        }

    # ========================================================
    # ASD — USE DATASET GROUND TRUTH FOR KNOWN PATIENT
    # ========================================================

    if clinical_asd:

        result["asd"] = {

            "detected": True,

            "status":
                "ASD detected",

            "source":
                "HVSMR clinical ground truth"
        }

    else:

        result["asd"] = {

            "detected": False,

            "status":
                "ASD not detected",

            "source":
                "HVSMR clinical ground truth"
        }

    # ========================================================
    # OVERALL
    # ========================================================

    if (
        clinical_vsd
        or clinical_asd
    ):

        result["overall"] = (
            "Septal defect detected"
        )

    else:

        result["overall"] = (
            "No septal defect detected"
        )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    result[
        "ground_truth"
    ] = clinical_ground_truth

    # ========================================================
    # AI EVALUATION
    # ========================================================

    result[
        "ai_evaluation"
    ] = {

        "vsd_correct":
            ai_vsd == clinical_vsd,

        "asd_correct":
            ai_asd == clinical_asd,

        "overall_correct":
            (
                (
                    ai_vsd
                    or ai_asd
                )
                ==
                (
                    clinical_vsd
                    or clinical_asd
                )
            )
    }

    result[
        "ground_truth_available"
    ] = True

    result[
        "screening_source"
    ] = (
        "HVSMR clinical ground truth "
        "for known dataset patient"
    )

    return result


# ============================================================
# COMPLETE PATIENT ANALYSIS
# ============================================================

def analyze_patient(
    input_mri_path,
    patient_id=None
):
    """
    Complete patient analysis pipeline.

    Pipeline:

        MRI
          ↓
        Patient ID extraction
          ↓
        HVSMR clinical label lookup
          ↓
        2D U-Net segmentation
          ↓
        3D segmentation volume
          ↓
        3D reconstruction
          ↓
        Cardiac feature extraction
          ↓
        Reference-cohort abnormality analysis
          ↓
        AI septal defect screening
          ↓
        HVSMR ground-truth comparison
          ↓
        JSON result
    """

    # ========================================================
    # CHECK INPUT MRI
    # ========================================================

    if not os.path.exists(
        input_mri_path
    ):

        raise FileNotFoundError(
            f"MRI not found: "
            f"{input_mri_path}"
        )

    # ========================================================
    # IDENTIFY PATIENT
    # ========================================================

    filename_patient_id = (
        extract_patient_id_from_filename(
            input_mri_path
        )
    )

    if filename_patient_id is not None:

        patient_id = (
            filename_patient_id
        )

    elif patient_id is not None:

        patient_id = normalize_patient_id(
            patient_id
        )

    else:

        patient_id = (
            "uploaded_patient"
        )

    # ========================================================
    # DISPLAY PATIENT
    # ========================================================

    print("\n" + "=" * 70)

    print(
        f"ANALYZING PATIENT: {patient_id}"
    )

    print("=" * 70)

    print(
        "Input MRI:",
        os.path.basename(
            input_mri_path
        )
    )

    # ========================================================
    # 0. HVSMR CLINICAL GROUND TRUTH
    # ========================================================

    clinical_ground_truth = None

    try:

        clinical_ground_truth = (
            get_hvsmr_ground_truth(
                patient_id
            )
        )

    except Exception as error:

        print(
            "\nWARNING: Could not load "
            "HVSMR clinical labels."
        )

        print(
            "Reason:",
            error
        )

    print("\n" + "=" * 70)
    print("HVSMR CLINICAL GROUND TRUTH")
    print("=" * 70)

    if clinical_ground_truth is not None:

        print(
            "\nPatient:",
            patient_id
        )

        print(
            "Clinical VSD:",
            clinical_ground_truth[
                "vsd"
            ][
                "status"
            ]
        )

        print(
            "Clinical ASD:",
            clinical_ground_truth[
                "asd"
            ][
                "status"
            ]
        )

    else:

        print(
            "\nNo HVSMR clinical label found."
        )

        print(
            "AI screening will be used."
        )

    # ========================================================
    # 1. SEGMENTATION
    # ========================================================

    print("\n" + "=" * 70)
    print("SEGMENTATION")
    print("=" * 70)

    prediction, mri = predict_volume(
        input_mri_path
    )

    if prediction is None:

        raise ValueError(
            "Segmentation returned None."
        )

    if mri is None:

        raise ValueError(
            "MRI image object returned by "
            "segmentation is None."
        )

    prediction = np.asarray(
        prediction,
        dtype=np.int16
    )

    if prediction.ndim != 3:

        raise ValueError(
            "Segmentation prediction must "
            "be a 3D volume. "
            f"Received: {prediction.shape}"
        )

    # ========================================================
    # 2. ORIGINAL MRI GEOMETRY
    # ========================================================

    spacing = np.asarray(
        mri.header.get_zooms()[:3],
        dtype=float
    )

    if spacing.size < 3:

        raise ValueError(
            "MRI does not contain valid "
            "3D voxel spacing."
        )

    if not np.all(
        np.isfinite(spacing)
    ):

        raise ValueError(
            f"Invalid MRI voxel spacing: {spacing}"
        )

    if np.any(
        spacing <= 0
    ):

        raise ValueError(
            f"Voxel spacing must be positive: {spacing}"
        )

    print(
        "\nVoxel spacing:",
        spacing,
        "mm"
    )

    print(
        "Prediction shape:",
        prediction.shape
    )

    print(
        "Prediction labels:",
        np.unique(
            prediction
        )
    )

    # ========================================================
    # 3. PATIENT OUTPUT DIRECTORY
    # ========================================================

    patient_dir = os.path.join(
        OUTPUT_DIR,
        patient_id
    )

    os.makedirs(
        patient_dir,
        exist_ok=True
    )

    # ========================================================
    # 4. SAVE ORIGINAL U-NET SEGMENTATION
    # ========================================================

    prediction_path = os.path.join(
        patient_dir,
        "segmentation.npy"
    )

    np.save(
        prediction_path,
        prediction
    )

    print(
        "\nSegmentation saved:",
        prediction_path
    )

    # ========================================================
    # 5. 3D RECONSTRUCTION
    # ========================================================

    models_dir = os.path.join(
        patient_dir,
        "3d_models"
    )

    os.makedirs(
        models_dir,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print("3D RECONSTRUCTION")
    print("=" * 70)

    reconstruction_result = reconstruct_3d(
        prediction=prediction,
        spacing=spacing,
        output_dir=models_dir,
        patient_id=patient_id
    )

    if reconstruction_result is None:

        raise ValueError(
            "3D reconstruction returned None."
        )

    model_count = int(
        reconstruction_result.get(
            "model_count",
            0
        )
    )

    print(
        f"\n3D reconstruction complete: "
        f"{model_count}/8 structures"
    )

    # ========================================================
    # 6. FEATURE EXTRACTION
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION")
    print("=" * 70)

    features = extract_features(
        prediction,
        spacing
    )

    if features is None:

        raise ValueError(
            "Feature extraction returned None."
        )

    print(
        "Extracted features:",
        len(features)
    )

    # ========================================================
    # 7. REFERENCE COHORT
    # ========================================================

    print("\n" + "=" * 70)
    print("REFERENCE COHORT")
    print("=" * 70)

    reference_stats = (
        build_reference_statistics()
    )

    if reference_stats is None:

        raise ValueError(
            "Reference statistics could not "
            "be generated."
        )

    print(
        "Reference statistics available for:",
        len(reference_stats),
        "features"
    )

    # ========================================================
    # 8. ABNORMALITY ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("ABNORMALITY ANALYSIS")
    print("=" * 70)

    abnormality = analyze_abnormality(
        features,
        reference_stats
    )

    if abnormality is None:

        raise ValueError(
            "Abnormality analysis returned None."
        )

    # ========================================================
    # 9. AI SEPTAL DEFECT SCREENING
    # ========================================================

    print("\n" + "=" * 70)
    print("AI SEPTAL DEFECT SCREENING")
    print("=" * 70)

    ai_septal_defect = (
        predict_septal_defect(
            features
        )
    )

    if ai_septal_defect is None:

        raise ValueError(
            "Septal defect screening "
            "returned None."
        )

    # ========================================================
    # 10. COMBINE AI + GROUND TRUTH
    # ========================================================

    septal_defect = (
        build_septal_defect_result(
            ai_septal_defect,
            clinical_ground_truth
        )
    )

    # ========================================================
    # DISPLAY SCREENING RESULT
    # ========================================================

    print(
        "\nVSD:",
        septal_defect
        .get("vsd", {})
        .get(
            "status",
            "Result unavailable"
        )
    )

    print(
        "ASD:",
        septal_defect
        .get("asd", {})
        .get(
            "status",
            "Result unavailable"
        )
    )

    print(
        "Overall:",
        septal_defect.get(
            "overall",
            "Result unavailable"
        )
    )

    # ========================================================
    # DISPLAY AI RESULT SEPARATELY
    # ========================================================

    if (
        "ai_prediction"
        in septal_defect
    ):

        print("\n" + "-" * 70)
        print("AI MODEL PREDICTION")
        print("-" * 70)

        print(
            "AI VSD:",
            septal_defect[
                "ai_prediction"
            ]
            .get(
                "vsd",
                {}
            )
            .get(
                "status",
                "Unavailable"
            )
        )

        print(
            "AI ASD:",
            septal_defect[
                "ai_prediction"
            ]
            .get(
                "asd",
                {}
            )
            .get(
                "status",
                "Unavailable"
            )
        )

        if (
            "ai_evaluation"
            in septal_defect
        ):

            evaluation = (
                septal_defect[
                    "ai_evaluation"
                ]
            )

            print(
                "AI VSD correct:",
                evaluation[
                    "vsd_correct"
                ]
            )

            print(
                "AI ASD correct:",
                evaluation[
                    "asd_correct"
                ]
            )

            print(
                "AI overall correct:",
                evaluation[
                    "overall_correct"
                ]
            )

    # ========================================================
    # 11. SAVE FEATURES
    # ========================================================

    feature_file = os.path.join(
        patient_dir,
        "features.csv"
    )

    save_features_csv(
        features,
        feature_file
    )

    # ========================================================
    # 12. SAVE ABNORMALITY ANALYSIS
    # ========================================================

    abnormality_file = os.path.join(
        patient_dir,
        "abnormality_analysis.csv"
    )

    save_abnormality_csv(
        abnormality,
        abnormality_file
    )

    # ========================================================
    # 13. COMPLETE RESULT JSON
    # ========================================================

    json_file = os.path.join(
        patient_dir,
        "result.json"
    )

    result = {

        "patient_id":
            patient_id,

        "input_mri":
            os.path.basename(
                input_mri_path
            ),

        "voxel_spacing_mm":
            spacing.tolist(),

        # ----------------------------------------------------
        # CLINICAL GROUND TRUTH
        # ----------------------------------------------------

        "hvsmr_ground_truth":
            clinical_ground_truth,

        # ----------------------------------------------------
        # SEGMENTATION
        # ----------------------------------------------------

        "segmentation": {

            "prediction_file":
                prediction_path,

            "shape":
                list(
                    prediction.shape
                ),

            "labels_present":
                [
                    int(label)
                    for label in np.unique(
                        prediction
                    )
                ],

            "num_structures":
                len(
                    LABEL_NAMES
                ),

            "structures":
                LABEL_NAMES
        },

        # ----------------------------------------------------
        # 3D RECONSTRUCTION
        # ----------------------------------------------------

        "3d_reconstruction":
            reconstruction_result,

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        "feature_extraction":
            features,

        # ----------------------------------------------------
        # REFERENCE STATISTICS
        # ----------------------------------------------------

        "reference_statistics":
            reference_stats,

        # ----------------------------------------------------
        # MORPHOLOGICAL ABNORMALITY
        # ----------------------------------------------------

        "abnormality_analysis":
            abnormality,

        # ----------------------------------------------------
        # SEPTAL DEFECT
        # ----------------------------------------------------

        "septal_defect_screening":
            septal_defect,

        # ----------------------------------------------------
        # OUTPUT FILES
        # ----------------------------------------------------

        "output_files": {

            "segmentation":
                prediction_path,

            "features":
                feature_file,

            "abnormality_analysis":
                abnormality_file,

            "three_d_models":
                models_dir
        },

        # ----------------------------------------------------
        # LIMITATIONS
        # ----------------------------------------------------

        "limitations": [

            "Current HVSMR segmentation "
            "does not contain a dedicated "
            "myocardium class.",

            "Myocardial wall thickness "
            "is therefore not calculated.",

            "Abnormality analysis is "
            "morphological screening and "
            "not a clinical diagnosis.",

            "The displayed reference intervals "
            "are dataset-derived statistical "
            "intervals from the project reference "
            "cohort and are not clinical normal "
            "ranges.",

            "For known HVSMR patients, the "
            "clinical CSV provides dataset "
            "ground truth for VSD and ASD.",

            "AI septal defect predictions are "
            "stored separately from the "
            "HVSMR ground truth.",

            "For patients not present in the "
            "clinical CSV, septal defect screening "
            "uses the AI feature-based model.",

            "This system is a research/project "
            "demonstration and is not a clinical "
            "diagnostic system."
        ]
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "COMPLETE PATIENT ANALYSIS FINISHED"
    )
    print("=" * 70)

    print(
        "\nPatient ID:",
        patient_id
    )

    print(
        "LV Volume:",
        features[
            "left_ventricle_volume_ml"
        ],
        "mL"
    )

    print(
        "RV Volume:",
        features[
            "right_ventricle_volume_ml"
        ],
        "mL"
    )

    print(
        "LV/RV Ratio:",
        features[
            "lv_rv_volume_ratio"
        ]
    )

    print(
        "\nMild deviations:",
        abnormality[
            "mild_deviations"
        ]
    )

    print(
        "Marked deviations:",
        abnormality[
            "marked_deviations"
        ]
    )

    print(
        "Overall assessment:",
        abnormality[
            "overall_assessment"
        ]
    )

    print(
        "\nVSD:",
        septal_defect
        .get("vsd", {})
        .get(
            "status",
            "Result unavailable"
        )
    )

    print(
        "ASD:",
        septal_defect
        .get("asd", {})
        .get(
            "status",
            "Result unavailable"
        )
    )

    print(
        "Septal defect overall:",
        septal_defect.get(
            "overall",
            "Result unavailable"
        )
    )

    if clinical_ground_truth is not None:

        print(
            "\nHVSMR ground truth:"
        )

        print(
            "VSD:",
            clinical_ground_truth[
                "vsd"
            ][
                "status"
            ]
        )

        print(
            "ASD:",
            clinical_ground_truth[
                "asd"
            ][
                "status"
            ]
        )

    print(
        "\n3D structures:",
        model_count,
        "/ 8"
    )

    print(
        "Result JSON:",
        json_file
    )

    print(
        "3D models directory:",
        models_dir
    )

    print("=" * 70)

    return result