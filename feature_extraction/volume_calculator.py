import numpy as np
import nibabel as nib


# ==========================================================
# HVSMR CARDIAC VOLUME CALCULATOR
# ==========================================================

LABELS = {
    1: "Left Ventricle",
    2: "Right Ventricle",
    3: "Left Atrium",
    4: "Right Atrium",
    5: "Aorta",
    6: "Pulmonary Artery",
    7: "Superior Vena Cava",
    8: "Inferior Vena Cava",
}


def calculate_volumes(volume_path):
    """
    Calculate physical volume of each HVSMR cardiac structure.

    volume_path can belong to ANY selected patient.
    The patient ID is not hardcoded here.
    """

    print("\nLoading segmentation:")
    print(volume_path)

    nii = nib.load(volume_path)
    volume = nii.get_fdata()

    spacing = nii.header.get_zooms()[:3]

    voxel_volume_mm3 = (
        float(spacing[0])
        * float(spacing[1])
        * float(spacing[2])
    )

    print(
        "Voxel spacing:",
        tuple(float(x) for x in spacing),
        "mm"
    )

    print(
        "Single voxel volume:",
        f"{voxel_volume_mm3:.4f} mm³"
    )

    results = {}

    for label, name in LABELS.items():

        voxel_count = int(np.sum(volume == label))

        volume_mm3 = voxel_count * voxel_volume_mm3
        volume_ml = volume_mm3 / 1000.0

        results[name] = {
            "label": label,
            "voxels": voxel_count,
            "volume_mm3": float(volume_mm3),
            "volume_ml": float(volume_ml),
        }

    return results


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate HVSMR cardiac volumes for a selected patient."
    )

    parser.add_argument(
        "segmentation",
        help="Path to the patient's predicted_segmentation.nii.gz"
    )

    args = parser.parse_args()

    features = calculate_volumes(args.segmentation)

    print("\n========== HVSMR VOLUME REPORT ==========\n")

    for structure, values in features.items():

        print(structure)
        print(f"Voxel Count : {values['voxels']:,}")
        print(f"Volume      : {values['volume_mm3']:.2f} mm³")
        print(f"Volume      : {values['volume_ml']:.2f} mL")
        print()

