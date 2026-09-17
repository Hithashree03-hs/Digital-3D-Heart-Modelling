import os
import csv
import numpy as np
import nibabel as nib

# ============================================================
# TEST PATIENTS
# ============================================================
TEST_PATIENTS = [
    "pat56",
    "pat8",
    "pat14",
    "pat15",
    "pat17",
    "pat47",
    "pat1",
    "pat7",
    "pat40"
]

# ============================================================
# PATHS
# ============================================================
PREDICTION_DIR = "outputs/test_results"
MRI_DIR = "../dataset/HVSMR/cropped/cropped"

OUTPUT_DIR = "outputs/test_results/patient_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# FEATURE EXTRACTION
# ============================================================
def extract_patient_features(patient):
    prediction_path = os.path.join(
        PREDICTION_DIR,
        f"{patient}_prediction.npy"
    )

    mri_path = os.path.join(
        MRI_DIR,
        f"{patient}_cropped.nii.gz"
    )

    if not os.path.exists(prediction_path):
        print(f"[SKIP] Prediction not found: {prediction_path}")
        return None

    if not os.path.exists(mri_path):
        print(f"[SKIP] MRI not found: {mri_path}")
        return None

    # --------------------------------------------------------
    # Load prediction
    # --------------------------------------------------------
    prediction = np.load(prediction_path)

    # --------------------------------------------------------
    # Load original MRI to obtain original geometry/spacing
    # --------------------------------------------------------
    mri = nib.load(mri_path)
    original_shape = mri.shape
    spacing = np.array(mri.header.get_zooms()[:3], dtype=float)

    print(f"\nProcessing {patient}")
    print(f"Original MRI shape : {original_shape}")
    print(f"Prediction shape   : {prediction.shape}")
    print(f"Voxel spacing (mm) : {spacing}")

    # --------------------------------------------------------
    # Restore prediction to original MRI dimensions
    # --------------------------------------------------------
    if prediction.shape != original_shape:

        from scipy.ndimage import zoom

        zoom_factors = np.array(original_shape) / np.array(prediction.shape)

        prediction = zoom(
            prediction,
            zoom_factors,
            order=0
        )

        prediction = prediction.astype(np.int16)

    voxel_volume_mm3 = np.prod(spacing)
    voxel_volume_ml = voxel_volume_mm3 / 1000.0

    features = {
        "patient": patient
    }

    # --------------------------------------------------------
    # Extract volume + dimensions
    # --------------------------------------------------------
    structure_volumes = {}

    for label, name in LABEL_NAMES.items():

        coords = np.argwhere(prediction == label)

        if coords.size == 0:
            volume_ml = 0.0
            dimensions_mm = (0.0, 0.0, 0.0)

        else:
            # Volume
            voxel_count = len(coords)
            volume_ml = voxel_count * voxel_volume_ml

            # Bounding-box dimensions
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)

            dimensions_voxels = max_coords - min_coords + 1
            dimensions_mm = dimensions_voxels * spacing

        structure_volumes[label] = volume_ml

        prefix = name.lower().replace(" ", "_")

        features[f"{prefix}_volume_ml"] = round(volume_ml, 3)
        features[f"{prefix}_length_mm"] = round(dimensions_mm[0], 3)
        features[f"{prefix}_width_mm"] = round(dimensions_mm[1], 3)
        features[f"{prefix}_depth_mm"] = round(dimensions_mm[2], 3)

    # --------------------------------------------------------
    # LV / RV comparison
    # --------------------------------------------------------
    lv_volume = structure_volumes[1]
    rv_volume = structure_volumes[2]

    if rv_volume > 0:
        lv_rv_ratio = lv_volume / rv_volume
        lv_rv_difference_percent = (
            abs(lv_volume - rv_volume)
            / ((lv_volume + rv_volume) / 2)
        ) * 100
    else:
        lv_rv_ratio = 0.0
        lv_rv_difference_percent = 0.0

    features["lv_rv_volume_ratio"] = round(lv_rv_ratio, 3)
    features["lv_rv_volume_difference_percent"] = round(
        lv_rv_difference_percent, 3
    )

    return features


# ============================================================
# MAIN
# ============================================================
all_features = []

for patient in TEST_PATIENTS:

    result = extract_patient_features(patient)

    if result is None:
        continue

    all_features.append(result)

    # --------------------------------------------------------
    # Individual patient CSV
    # --------------------------------------------------------
    patient_dir = os.path.join(
        OUTPUT_DIR,
        patient
    )

    os.makedirs(patient_dir, exist_ok=True)

    patient_csv = os.path.join(
        patient_dir,
        "features.csv"
    )

    with open(patient_csv, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["Feature", "Value"])

        for key, value in result.items():
            if key != "patient":
                writer.writerow([key, value])

    print(f"[SAVED] {patient_csv}")


# ============================================================
# Save combined CSV
# ============================================================
if all_features:

    combined_csv = os.path.join(
        OUTPUT_DIR,
        "all_test_patient_features.csv"
    )

    fieldnames = list(all_features[0].keys())

    with open(combined_csv, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_features)

    print("\n========================================")
    print("FEATURE EXTRACTION COMPLETED")
    print("========================================")
    print(f"Patients processed : {len(all_features)}")
    print(f"Combined file      : {combined_csv}")

else:
    print("\nNo patient features were generated.")