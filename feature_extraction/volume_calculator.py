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


# ==========================================================
# Calculate Volumes
# ==========================================================

def calculate_volumes(volume_path):

    """
    Calculate physical volume of each HVSMR cardiac structure.

    Parameters
    ----------
    volume_path : str
        Path to predicted segmentation NIfTI file.

    Returns
    -------
    dict
        Volume measurements for each cardiac structure.
    """

    print("\nLoading segmentation:")
    print(volume_path)

    nii = nib.load(volume_path)

    volume = nii.get_fdata()

    # ------------------------------------------------------
    # Get physical voxel spacing
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Calculate volume for each structure
    # ------------------------------------------------------

    for label, name in LABELS.items():

        voxel_count = int(
            np.sum(volume == label)
        )

        volume_mm3 = (
            voxel_count
            * voxel_volume_mm3
        )

        volume_ml = volume_mm3 / 1000.0

        results[name] = {

            "label": label,

            "voxels": voxel_count,

            "volume_mm3": float(
                volume_mm3
            ),

            "volume_ml": float(
                volume_ml
            )

        }

    return results


# ==========================================================
# Standalone Testing
# ==========================================================

if __name__ == "__main__":

    volume_path = (
        "../outputs/hvsmr/pat7/"
        "predicted_segmentation.nii.gz"
    )

    features = calculate_volumes(
        volume_path
    )

    print(
        "\n========== HVSMR VOLUME REPORT ==========\n"
    )

    for structure, values in features.items():

        print(structure)

        print(
            f"Voxel Count : {values['voxels']:,}"
        )

        print(
            f"Volume      : "
            f"{values['volume_mm3']:.2f} mm³"
        )

        print(
            f"Volume      : "
            f"{values['volume_ml']:.2f} mL"
        )

        print()