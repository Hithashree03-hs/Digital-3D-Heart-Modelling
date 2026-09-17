import csv
import os


# ============================================================
# INPUT FILES
# ============================================================

FEATURE_FILE = (
    "outputs/test_results/patient_analysis/"
    "all_test_patient_features.csv"
)

ABNORMALITY_FILE = (
    "outputs/test_results/patient_analysis/"
    "patient_abnormality_summary.csv"
)


# ============================================================
# OUTPUT FILE
# ============================================================

OUTPUT_FILE = (
    "outputs/test_results/patient_analysis/"
    "final_patient_results.csv"
)


# ============================================================
# FEATURES TO INCLUDE IN FINAL TABLE
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


DISPLAY_NAMES = {
    "left_ventricle_volume_ml": "LV Volume (mL)",
    "right_ventricle_volume_ml": "RV Volume (mL)",
    "left_atrium_volume_ml": "LA Volume (mL)",
    "right_atrium_volume_ml": "RA Volume (mL)",

    "left_ventricle_length_mm": "LV Length (mm)",
    "left_ventricle_width_mm": "LV Width (mm)",
    "left_ventricle_depth_mm": "LV Depth (mm)",

    "right_ventricle_length_mm": "RV Length (mm)",
    "right_ventricle_width_mm": "RV Width (mm)",
    "right_ventricle_depth_mm": "RV Depth (mm)",

    "left_atrium_length_mm": "LA Length (mm)",
    "left_atrium_width_mm": "LA Width (mm)",
    "left_atrium_depth_mm": "LA Depth (mm)",

    "right_atrium_length_mm": "RA Length (mm)",
    "right_atrium_width_mm": "RA Width (mm)",
    "right_atrium_depth_mm": "RA Depth (mm)",

    "lv_rv_volume_ratio": "LV/RV Volume Ratio",
    "lv_rv_volume_difference_percent":
        "LV/RV Volume Difference (%)"
}


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(path):

    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


features = load_csv(FEATURE_FILE)
abnormality = load_csv(ABNORMALITY_FILE)


# ============================================================
# CREATE ABNORMALITY LOOKUP
# ============================================================

abnormality_lookup = {}

for row in abnormality:

    patient = row["patient"]

    abnormality_lookup[patient] = {
        "mild_deviations":
            row["mild_deviations"],

        "marked_deviations":
            row["marked_deviations"],

        "overall_assessment":
            row["overall_assessment"]
    }


# ============================================================
# CREATE FINAL TABLE
# ============================================================

final_rows = []

for patient_row in features:

    patient = patient_row["patient"]

    final_row = {
        "Patient": patient
    }

    for feature in FEATURES:

        output_name = DISPLAY_NAMES.get(
            feature,
            feature
        )

        final_row[output_name] = patient_row[feature]

    # --------------------------------------------------------
    # Add abnormality information
    # --------------------------------------------------------

    if patient in abnormality_lookup:

        final_row["Mild Deviations"] = (
            abnormality_lookup[patient]
            ["mild_deviations"]
        )

        final_row["Marked Deviations"] = (
            abnormality_lookup[patient]
            ["marked_deviations"]
        )

        final_row["Overall Assessment"] = (
            abnormality_lookup[patient]
            ["overall_assessment"]
        )

    else:

        final_row["Mild Deviations"] = ""
        final_row["Marked Deviations"] = ""
        final_row["Overall Assessment"] = ""

    final_rows.append(final_row)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=final_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(final_rows)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PATIENT-WISE RESULTS CREATED")
print("=" * 70)

print(
    f"\nPatients included: {len(final_rows)}"
)

print(
    "\nPatients:"
)

for row in final_rows:
    print(
        f"  {row['Patient']}: "
        f"{row['Overall Assessment']}"
    )

print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)