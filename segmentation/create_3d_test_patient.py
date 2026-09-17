import os

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import trimesh
from skimage.measure import marching_cubes


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
# ANATOMICAL LABELS
# ============================================================

LABELS = {
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
# PROCESS ONE PATIENT
# ============================================================

def reconstruct_patient(patient_id):

    prediction_path = (
        f"outputs/test_results/"
        f"{patient_id}_prediction.npy"
    )

    mri_path = (
        f"../dataset/HVSMR/cropped/cropped/"
        f"{patient_id}_cropped.nii.gz"
    )

    output_dir = (
        f"outputs/test_3d_models/"
        f"{patient_id}"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print(
        f"3D RECONSTRUCTION - {patient_id.upper()}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(prediction_path):

        print(
            f"Prediction not found: "
            f"{prediction_path}"
        )

        return False

    if not os.path.exists(mri_path):

        print(
            f"MRI not found: "
            f"{mri_path}"
        )

        return False

    # --------------------------------------------------------
    # Load prediction
    # --------------------------------------------------------

    prediction = np.load(
        prediction_path
    )

    print(
        "\nPrediction shape:",
        prediction.shape
    )

    # --------------------------------------------------------
    # Load original MRI
    # --------------------------------------------------------

    mri = nib.load(
        mri_path
    )

    original_shape = mri.shape

    spacing = np.array(
        mri.header.get_zooms()[:3],
        dtype=float
    )

    print(
        "Original MRI shape:",
        original_shape
    )

    print(
        "Voxel spacing:",
        spacing,
        "mm"
    )

    # --------------------------------------------------------
    # Restore prediction to original geometry
    # --------------------------------------------------------

    print(
        "\nRestoring prediction "
        "to original MRI geometry..."
    )

    prediction_tensor = torch.from_numpy(
        prediction.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0)

    prediction_original = F.interpolate(
        prediction_tensor,
        size=original_shape,
        mode="nearest"
    )

    prediction_original = (
        prediction_original
        .squeeze()
        .numpy()
        .astype(np.int16)
    )

    print(
        "Restored prediction shape:",
        prediction_original.shape
    )

    # --------------------------------------------------------
    # Create individual structures
    # --------------------------------------------------------

    successful_structures = []

    for label, structure_name in LABELS.items():

        print("\n" + "-" * 70)
        print(
            f"Processing {structure_name}"
        )

        mask = (
            prediction_original == label
        )

        voxel_count = int(
            mask.sum()
        )

        print(
            "Voxel count:",
            voxel_count
        )

        if voxel_count < 10:

            print(
                "Too few voxels. Skipping."
            )

            continue

        # ----------------------------------------------------
        # Marching cubes
        # ----------------------------------------------------

        try:

            vertices, faces, normals, values = (
                marching_cubes(
                    mask.astype(np.uint8),
                    level=0.5,
                    spacing=spacing
                )
            )

        except Exception as e:

            print(
                f"Marching cubes failed: {e}"
            )

            continue

        print(
            "Vertices:",
            len(vertices)
        )

        print(
            "Faces:",
            len(faces)
        )

        # ----------------------------------------------------
        # Create mesh
        # ----------------------------------------------------

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=True
        )

        safe_name = (
            structure_name
            .lower()
            .replace(" ", "_")
        )

        # ----------------------------------------------------
        # Save STL
        # ----------------------------------------------------

        stl_path = os.path.join(
            output_dir,
            f"{patient_id}_{safe_name}.stl"
        )

        mesh.export(
            stl_path
        )

        print(
            "Saved STL:",
            stl_path
        )

        # ----------------------------------------------------
        # Save PLY
        # ----------------------------------------------------

        ply_path = os.path.join(
            output_dir,
            f"{patient_id}_{safe_name}.ply"
        )

        mesh.export(
            ply_path
        )

        print(
            "Saved PLY:",
            ply_path
        )

        successful_structures.append(
            structure_name
        )

    # ========================================================
    # COMBINED HEART SURFACE
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "Creating combined 3D heart surface"
    )
    print("=" * 70)

    combined_mask = (
        prediction_original != 0
    )

    if combined_mask.sum() > 0:

        try:

            vertices, faces, normals, values = (
                marching_cubes(
                    combined_mask.astype(np.uint8),
                    level=0.5,
                    spacing=spacing
                )
            )

            combined_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=True
            )

            combined_stl = os.path.join(
                output_dir,
                f"{patient_id}_heart_combined.stl"
            )

            combined_ply = os.path.join(
                output_dir,
                f"{patient_id}_heart_combined.ply"
            )

            combined_mesh.export(
                combined_stl
            )

            combined_mesh.export(
                combined_ply
            )

            print(
                "Combined STL:",
                combined_stl
            )

            print(
                "Combined PLY:",
                combined_ply
            )

        except Exception as e:

            print(
                f"Combined reconstruction failed: {e}"
            )

    else:

        print(
            "No foreground cardiac structures found."
        )

    print(
        f"\nStructures reconstructed:"
        f" {len(successful_structures)}"
    )

    print(
        f"Output folder:\n{output_dir}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

print("\n" + "=" * 70)
print("BATCH 3D HEART RECONSTRUCTION")
print("=" * 70)

completed = 0

for patient_id in TEST_PATIENTS:

    if reconstruct_patient(patient_id):

        completed += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BATCH 3D RECONSTRUCTION COMPLETED")
print("=" * 70)

print(
    f"Patients successfully processed: "
    f"{completed}/{len(TEST_PATIENTS)}"
)

print(
    "\n3D models are located in:"
)

print(
    "outputs/test_3d_models/"
)