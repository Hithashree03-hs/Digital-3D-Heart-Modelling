import os
import csv
import numpy as np

REFERENCE_FILE = "outputs/test_results/reference_cohort.csv"

PATIENT_FEATURES_FILE = (
    "outputs/test_results/patient_analysis/"
    "all_test_patient_features.csv"
)

OUTPUT_DIR = "outputs/test_results/patient_analysis"


# ============================================================
# PRIMARY CARDIAC FEATURES
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


DISPLAY_NAMES = {
    "left_ventricle_volume_ml": "Left Ventricle Volume",
    "right_ventricle_volume_ml": "Right Ventricle Volume",
    "left_atrium_volume_ml": "Left Atrium Volume",
    "right_atrium_volume_ml": "Right Atrium Volume",

    "left_ventricle_length_mm": "Left Ventricle Length",
    "left_ventricle_width_mm": "Left Ventricle Width",
    "left_ventricle_depth_mm": "Left Ventricle Depth",

    "right_ventricle_length_mm": "Right Ventricle Length",
    "right_ventricle_width_mm": "Right Ventricle Width",
    "right_ventricle_depth_mm": "Right Ventricle Depth",

    "left_atrium_length_mm": "Left Atrium Length",
    "left_atrium_width_mm": "Left Atrium Width",
    "left_atrium_depth_mm": "Left Atrium Depth",

    "right_atrium_length_mm": "Right Atrium Length",
    "right_atrium_width_mm": "Right Atrium Width",
    "right_atrium_depth_mm": "Right Atrium Depth",

    "lv_rv_volume_ratio": "LV/RV Volume Ratio",
    "lv_rv_volume_difference_percent": "LV/RV Volume Difference (%)"
}


# ============================================================
# READ CSV
# ============================================================
def load_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


reference_rows = load_csv(REFERENCE_FILE)
patient_rows = load_csv(PATIENT_FEATURES_FILE)


print("\n========================================")
print("REFINED PATIENT-WISE ABNORMALITY ANALYSIS")
print("========================================")
print(f"Reference patients : {len(reference_rows)}")
print(f"Test patients      : {len(patient_rows)}")


# ============================================================
# BUILD ROBUST REFERENCE RANGES
#
# We use Q1/Q3 and IQR because the training cohort contains
# considerable variation and some extreme values.
# ============================================================
reference_stats = {}

for feature in PRIMARY_FEATURES:

    values = []

    for row in reference_rows:

        try:
            value = float(row[feature])

            if value > 0:
                values.append(value)

        except (ValueError, KeyError):
            pass

    if len(values) < 5:
        continue

    values = np.array(values, dtype=float)

    q1 = np.percentile(values, 25)
    median = np.percentile(values, 50)
    q3 = np.percentile(values, 75)

    iqr = q3 - q1

    lower = max(0.0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr

    reference_stats[feature] = {
        "q1": q1,
        "median": median,
        "q3": q3,
        "lower": lower,
        "upper": upper
    }


# ============================================================
# ANALYZE EACH PATIENT
# ============================================================
all_results = []
patient_summaries = []

for patient_row in patient_rows:

    patient = patient_row["patient"]

    print("\n----------------------------------------")
    print(f"Patient: {patient}")
    print("----------------------------------------")

    patient_results = []

    mild_count = 0
    marked_count = 0

    for feature in PRIMARY_FEATURES:

        if feature not in reference_stats:
            continue

        try:
            patient_value = float(patient_row[feature])
        except (ValueError, KeyError):
            continue

        stats = reference_stats[feature]

        lower = stats["lower"]
        upper = stats["upper"]

        # ----------------------------------------------------
        # Check deviation
        # ----------------------------------------------------
        if patient_value < lower:
            deviation = "Low"

        elif patient_value > upper:
            deviation = "High"

        else:
            deviation = "Within reference range"

        # ----------------------------------------------------
        # How far from reference median?
        # ----------------------------------------------------
        median = stats["median"]

        if median > 0:
            percent_from_median = (
                abs(patient_value - median)
                / median
            ) * 100
        else:
            percent_from_median = 0.0

        # ----------------------------------------------------
        # Classify severity of morphological deviation
        # ----------------------------------------------------
        if deviation == "Within reference range":

            severity = "Normal range"

        elif percent_from_median >= 75:

            severity = "Marked deviation"
            marked_count += 1

        else:

            severity = "Mild deviation"
            mild_count += 1

        result = {
            "patient": patient,
            "feature": feature,
            "feature_name": DISPLAY_NAMES.get(
                feature,
                feature
            ),
            "patient_value": round(patient_value, 3),
            "reference_median": round(
                median, 3
            ),
            "reference_q1": round(
                stats["q1"], 3
            ),
            "reference_q3": round(
                stats["q3"], 3
            ),
            "reference_lower_limit": round(
                lower, 3
            ),
            "reference_upper_limit": round(
                upper, 3
            ),
            "deviation": deviation,
            "percent_from_median": round(
                percent_from_median, 2
            ),
            "severity": severity
        }

        patient_results.append(result)
        all_results.append(result)

        if deviation != "Within reference range":
            print(
                f"{DISPLAY_NAMES.get(feature, feature)}"
                f" = {patient_value:.2f}"
                f" → {severity}"
            )

    # ========================================================
    # OVERALL PATIENT ASSESSMENT
    # ========================================================

    total_deviations = mild_count + marked_count

    # We avoid calling the patient "diseased".
    if marked_count >= 2:

        overall = (
            "Multiple marked morphological deviations "
            "detected"
        )

    elif marked_count == 1 or mild_count >= 3:

        overall = (
            "Morphological deviations detected"
        )

    else:

        overall = (
            "No marked morphological deviation detected"
        )

    print("\nOverall assessment:")
    print(f"  Mild deviations   : {mild_count}")
    print(f"  Marked deviations : {marked_count}")
    print(f"  Result            : {overall}")

    # ========================================================
    # SAVE INDIVIDUAL CSV
    # ========================================================

    patient_dir = os.path.join(
        OUTPUT_DIR,
        patient
    )

    os.makedirs(
        patient_dir,
        exist_ok=True
    )

    csv_path = os.path.join(
        patient_dir,
        "abnormality_analysis.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        if patient_results:

            writer = csv.DictWriter(
                f,
                fieldnames=patient_results[0].keys()
            )

            writer.writeheader()
            writer.writerows(patient_results)

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_path = os.path.join(
        patient_dir,
        "abnormality_summary.txt"
    )

    with open(summary_path, "w") as f:

        f.write(
            "PATIENT-WISE STRUCTURAL ABNORMALITY ANALYSIS\n"
        )
        f.write(
            "============================================\n\n"
        )

        f.write(f"Patient: {patient}\n\n")

        f.write(
            f"Features analyzed: "
            f"{len(patient_results)}\n"
        )

        f.write(
            f"Mild deviations: "
            f"{mild_count}\n"
        )

        f.write(
            f"Marked deviations: "
            f"{marked_count}\n\n"
        )

        f.write(
            "Overall assessment:\n"
        )

        f.write(
            overall + "\n\n"
        )

        f.write(
            "Interpretation:\n"
        )

        f.write(
            "The result represents morphological "
            "comparison with the training reference "
            "cohort. It is not a clinical diagnosis.\n"
        )

    patient_summaries.append({
        "patient": patient,
        "mild_deviations": mild_count,
        "marked_deviations": marked_count,
        "overall_assessment": overall
    })


# ============================================================
# SAVE COMBINED FEATURE-LEVEL RESULTS
# ============================================================
combined_file = os.path.join(
    OUTPUT_DIR,
    "all_test_patient_abnormality_analysis.csv"
)

if all_results:

    with open(
        combined_file,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=all_results[0].keys()
        )

        writer.writeheader()
        writer.writerows(all_results)


# ============================================================
# SAVE PATIENT SUMMARY
# ============================================================
summary_file = os.path.join(
    OUTPUT_DIR,
    "patient_abnormality_summary.csv"
)

with open(
    summary_file,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "patient",
            "mild_deviations",
            "marked_deviations",
            "overall_assessment"
        ]
    )

    writer.writeheader()
    writer.writerows(patient_summaries)


print("\n========================================")
print("ABNORMALITY ANALYSIS COMPLETED")
print("========================================")
print(
    f"Patients analyzed : {len(patient_rows)}"
)

print(
    f"\nCombined feature results:"
    f"\n{combined_file}"
)

print(
    f"\nPatient summary:"
    f"\n{summary_file}"
)