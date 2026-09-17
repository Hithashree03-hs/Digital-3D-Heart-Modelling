import os
import csv
import json

import numpy as np

from backend.services.segmentation import predict_volume
from backend.services.reconstruction import reconstruct_3d


# ============================================================
# BACKEND PATHS
# ============================================================

BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

REFERENCE_FILE = os.path.join(
    BACKEND_DIR,
    "reference",
    "reference_cohort.csv"
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
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with open(
        path,
        "r",
        newline=""
    ) as file:
        return list(
            csv.DictReader(file)
        )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(prediction, spacing):
    """
    Extract physical cardiac measurements from
    a 3D segmentation volume.
    """

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

            voxel_count = len(coords)

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

        structure_volumes[label] = (
            volume_ml
        )

        prefix = (
            structure_name
            .lower()
            .replace(" ", "_")
        )

        features[
            f"{prefix}_volume_ml"
        ] = round(
            volume_ml,
            3
        )

        features[
            f"{prefix}_length_mm"
        ] = round(
            dimensions_mm[0],
            3
        )

        features[
            f"{prefix}_width_mm"
        ] = round(
            dimensions_mm[1],
            3
        )

        features[
            f"{prefix}_depth_mm"
        ] = round(
            dimensions_mm[2],
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

        lv_rv_difference = (
            abs(
                lv_volume -
                rv_volume
            )
            /
            (
                (lv_volume + rv_volume)
                / 2
            )
        ) * 100

    else:

        lv_rv_ratio = 0.0
        lv_rv_difference = 0.0

    features[
        "lv_rv_volume_ratio"
    ] = round(
        lv_rv_ratio,
        3
    )

    features[
        "lv_rv_volume_difference_percent"
    ] = round(
        lv_rv_difference,
        3
    )

    return features


# ============================================================
# REFERENCE COHORT STATISTICS
# ============================================================

def build_reference_statistics():

    reference_rows = load_csv(
        REFERENCE_FILE
    )

    reference_stats = {}

    for feature in PRIMARY_FEATURES:

        values = []

        for row in reference_rows:

            try:

                value = float(
                    row[feature]
                )

                if value > 0:
                    values.append(value)

            except (
                ValueError,
                KeyError,
                TypeError
            ):

                continue

        if len(values) < 5:
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
            q1 - 1.5 * iqr
        )

        upper = (
            q3 + 1.5 * iqr
        )

        reference_stats[
            feature
        ] = {
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower": float(lower),
            "upper": float(upper)
        }

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
    the training reference cohort.

    This is morphological screening,
    not a clinical diagnosis.
    """

    feature_results = []

    mild_count = 0
    marked_count = 0

    for feature in PRIMARY_FEATURES:

        if feature not in reference_stats:
            continue

        if feature not in features:
            continue

        patient_value = float(
            features[feature]
        )

        stats = reference_stats[
            feature
        ]

        lower = stats["lower"]
        upper = stats["upper"]
        median = stats["median"]

        if patient_value < lower:

            deviation = "Low"

        elif patient_value > upper:

            deviation = "High"

        else:

            deviation = (
                "Within reference range"
            )

        if median > 0:

            percent_from_median = (
                abs(
                    patient_value -
                    median
                )
                / median
            ) * 100

        else:

            percent_from_median = 0.0

        if (
            deviation ==
            "Within reference range"
        ):

            severity = "Normal range"

        elif percent_from_median >= 75:

            severity = "Marked deviation"
            marked_count += 1

        else:

            severity = "Mild deviation"
            mild_count += 1

        feature_results.append({
            "feature": feature,
            "patient_value": round(
                patient_value,
                3
            ),
            "reference_median": round(
                median,
                3
            ),
            "reference_q1": round(
                stats["q1"],
                3
            ),
            "reference_q3": round(
                stats["q3"],
                3
            ),
            "reference_lower": round(
                lower,
                3
            ),
            "reference_upper": round(
                upper,
                3
            ),
            "deviation": deviation,
            "percent_from_median": round(
                percent_from_median,
                2
            ),
            "severity": severity
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
        "features": feature_results,
        "mild_deviations": mild_count,
        "marked_deviations": marked_count,
        "overall_assessment": overall
    }


# ============================================================
# COMPLETE PATIENT ANALYSIS
# ============================================================

def analyze_patient(
    input_mri_path,
    patient_id="uploaded_patient"
):
    """
    Complete patient-independent analysis pipeline.

    The input can be any compatible patient's
    3D NIfTI MRI volume.
    """

    if not os.path.exists(
        input_mri_path
    ):

        raise FileNotFoundError(
            f"MRI not found: "
            f"{input_mri_path}"
        )

    print("\n" + "=" * 70)
    print(
        f"ANALYZING PATIENT: {patient_id}"
    )
    print("=" * 70)

    # ========================================================
    # 1. SEGMENTATION
    # ========================================================

    prediction, mri = predict_volume(
        input_mri_path
    )

    # ========================================================
    # 2. ORIGINAL MRI GEOMETRY
    # ========================================================

    spacing = np.array(
        mri.header.get_zooms()[:3],
        dtype=float
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
    # 4. SAVE SEGMENTATION
    # ========================================================

    prediction_path = os.path.join(
        patient_dir,
        "segmentation.npy"
    )

    np.save(
        prediction_path,
        prediction
    )

    # ========================================================
    # 5. 3D RECONSTRUCTION
    # ========================================================

    print("\n" + "=" * 70)
    print("3D RECONSTRUCTION")
    print("=" * 70)

    models_dir = os.path.join(
        patient_dir,
        "3d_models"
    )

    models = reconstruct_3d(
        prediction=prediction,
        spacing=spacing,
        output_dir=models_dir,
        patient_id=patient_id
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

    # ========================================================
    # 7. REFERENCE COHORT
    # ========================================================

    reference_stats = (
        build_reference_statistics()
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

    # ========================================================
    # 9. SAVE FEATURES
    # ========================================================

    feature_file = os.path.join(
        patient_dir,
        "features.csv"
    )

    with open(
        feature_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Feature",
            "Value"
        ])

        for name, value in features.items():

            writer.writerow([
                name,
                value
            ])

    # ========================================================
    # 10. SAVE ABNORMALITY ANALYSIS
    # ========================================================

    abnormality_file = os.path.join(
        patient_dir,
        "abnormality_analysis.csv"
    )

    with open(
        abnormality_file,
        "w",
        newline=""
    ) as file:

        if abnormality["features"]:

            fieldnames = list(
                abnormality[
                    "features"
                ][0].keys()
            )

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                abnormality["features"]
            )

    # ========================================================
    # 11. SAVE COMPLETE JSON
    # ========================================================

    result = {

        "patient_id":
            patient_id,

        "input_mri":
            os.path.basename(
                input_mri_path
            ),

        "voxel_spacing_mm":
            spacing.tolist(),

        "segmentation": {
            "prediction_file":
                prediction_path,

            "shape":
                list(
                    prediction.shape
                )
        },

        "3d_reconstruction":
            models,

        "feature_extraction":
            features,

        "abnormality_analysis":
            abnormality,

        "limitations": [
            "Current HVSMR segmentation "
            "does not contain a dedicated "
            "myocardium class.",
            "Myocardial wall thickness "
            "is therefore not calculated.",
            "Abnormality analysis is "
            "morphological screening and "
            "not a clinical diagnosis."
        ]
    }

    json_file = os.path.join(
        patient_dir,
        "result.json"
    )

    with open(
        json_file,
        "w"
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
    print("COMPLETE PATIENT ANALYSIS FINISHED")
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
        "\nResult JSON:",
        json_file
    )

    print(
        "3D models directory:",
        models_dir
    )

    return result