import os
import csv
import numpy as np
import nibabel as nib

# ============================================================
# 42 TRAINING PATIENTS USED DURING U-NET TRAINING
# ============================================================
TRAIN_PATIENTS = [
    "pat38", "pat23", "pat54", "pat11", "pat16", "pat20",
    "pat55", "pat58", "pat33", "pat19", "pat9", "pat36",
    "pat31", "pat45", "pat30", "pat49", "pat3", "pat21",
    "pat50", "pat4", "pat29", "pat10", "pat59", "pat22",
    "pat41", "pat24", "pat0", "pat44", "pat25", "pat52",
    "pat18", "pat28", "pat39", "pat26", "pat48", "pat12",
    "pat35", "pat42", "pat32", "pat53", "pat13", "pat46"
]

# ============================================================
# PATHS
# ============================================================
MRI_DIR = "../dataset/HVSMR/cropped/cropped"

OUTPUT_DIR = "outputs/test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "reference_cohort.csv"
)

# ============================================================
# HVSMR LABELS
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
# EXTRACT FEATURES FROM GROUND-TRUTH SEGMENTATION
# ============================================================
def extract_patient_features(patient):

    mri_path = os.path.join(
        MRI_DIR,
        f"{patient}_cropped.nii.gz"
    )

    seg_path = os.path.join(
        MRI_DIR,
        f"{patient}_cropped_seg.nii.gz"
    )

    if not os.path.exists(mri_path):
        print(f"[SKIP] MRI not found: {mri_path}")
        return None

    if not os.path.exists(seg_path):
        print(f"[SKIP] Segmentation not found: {seg_path}")
        return None

    mri = nib.load(mri_path)
    seg = nib.load(seg_path)

    mask = np.asarray(seg.dataobj)

    spacing = np.array(
        mri.header.get_zooms()[:3],
        dtype=float
    )

    voxel_volume_mm3 = np.prod(spacing)
    voxel_volume_ml = voxel_volume_mm3 / 1000.0

    features = {
        "patient": patient
    }

    structure_volumes = {}

    for label, name in LABEL_NAMES.items():

        coords = np.argwhere(mask == label)

        if coords.size == 0:

            volume_ml = 0.0
            dimensions_mm = (0.0, 0.0, 0.0)

        else:

            # Volume
            voxel_count = len(coords)

            volume_ml = (
                voxel_count *
                voxel_volume_ml
            )

            # Bounding-box dimensions
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)

            dimensions_voxels = (
                max_coords -
                min_coords +
                1
            )

            dimensions_mm = (
                dimensions_voxels *
                spacing
            )

        structure_volumes[label] = volume_ml

        prefix = name.lower().replace(" ", "_")

        features[f"{prefix}_volume_ml"] = volume_ml
        features[f"{prefix}_length_mm"] = dimensions_mm[0]
        features[f"{prefix}_width_mm"] = dimensions_mm[1]
        features[f"{prefix}_depth_mm"] = dimensions_mm[2]

    # ========================================================
    # LV / RV FEATURES
    # ========================================================

    lv_volume = structure_volumes[1]
    rv_volume = structure_volumes[2]

    if rv_volume > 0:

        lv_rv_ratio = (
            lv_volume /
            rv_volume
        )

        lv_rv_difference_percent = (
            abs(lv_volume - rv_volume)
            / ((lv_volume + rv_volume) / 2)
        ) * 100

    else:

        lv_rv_ratio = 0.0
        lv_rv_difference_percent = 0.0

    features["lv_rv_volume_ratio"] = lv_rv_ratio

    features[
        "lv_rv_volume_difference_percent"
    ] = lv_rv_difference_percent

    return features


# ============================================================
# MAIN
# ============================================================

all_features = []

print("\n========================================")
print("BUILDING REFERENCE COHORT")
print("========================================")

for patient in TRAIN_PATIENTS:

    print(f"Processing {patient}...")

    result = extract_patient_features(patient)

    if result is not None:
        all_features.append(result)


# ============================================================
# SAVE PATIENT FEATURES
# ============================================================

if not all_features:

    print("\nNo training patients were processed.")
    raise SystemExit


fieldnames = list(all_features[0].keys())

with open(
    OUTPUT_CSV,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_features)


# ============================================================
# STATISTICS
# ============================================================

print("\n========================================")
print("REFERENCE COHORT STATISTICS")
print("========================================")

for feature in fieldnames:

    if feature == "patient":
        continue

    values = np.array([
        float(row[feature])
        for row in all_features
    ])

    mean_value = np.mean(values)
    std_value = np.std(values, ddof=1)
    median_value = np.median(values)
    min_value = np.min(values)
    max_value = np.max(values)

    print(
        f"\n{feature}"
        f"\n  Mean   : {mean_value:.3f}"
        f"\n  SD     : {std_value:.3f}"
        f"\n  Median : {median_value:.3f}"
        f"\n  Min    : {min_value:.3f}"
        f"\n  Max    : {max_value:.3f}"
    )


print("\n========================================")
print("REFERENCE COHORT COMPLETED")
print("========================================")
print(f"Patients processed : {len(all_features)}")
print(f"Saved to           : {OUTPUT_CSV}")