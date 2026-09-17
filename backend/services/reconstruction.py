import os

import numpy as np
from skimage.measure import marching_cubes
import trimesh


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


def reconstruct_3d(
    prediction,
    spacing,
    output_dir,
    patient_id
):
    """
    Create individual STL/PLY meshes for all segmented
    cardiac structures and combined heart surface.

    Parameters
    ----------
    prediction : np.ndarray
        3D segmentation volume.

    spacing : np.ndarray
        Voxel spacing in mm (x, y, z).

    output_dir : str
        Directory where 3D models will be saved.

    patient_id : str
        Identifier assigned to the uploaded patient.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    generated_models = []

    # ========================================================
    # INDIVIDUAL STRUCTURES
    # ========================================================

    for label, structure_name in LABEL_NAMES.items():

        mask = (
            prediction == label
        )

        voxel_count = int(
            mask.sum()
        )

        if voxel_count < 10:
            continue

        try:

            vertices, faces, normals, values = (
                marching_cubes(
                    mask.astype(np.uint8),
                    level=0.5,
                    spacing=spacing
                )
            )

        except Exception as exc:

            print(
                f"3D reconstruction failed for "
                f"{structure_name}: {exc}"
            )

            continue

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

        stl_path = os.path.join(
            output_dir,
            f"{patient_id}_{safe_name}.stl"
        )

        ply_path = os.path.join(
            output_dir,
            f"{patient_id}_{safe_name}.ply"
        )

        mesh.export(stl_path)
        mesh.export(ply_path)

        generated_models.append({
            "structure": structure_name,
            "stl": stl_path,
            "ply": ply_path
        })

    # ========================================================
    # COMBINED HEART SURFACE
    # ========================================================

    combined_mask = (
        prediction != 0
    )

    combined_model = None

    if combined_mask.any():

        try:

            vertices, faces, normals, values = (
                marching_cubes(
                    combined_mask.astype(np.uint8),
                    level=0.5,
                    spacing=spacing
                )
            )

            combined_model = trimesh.Trimesh(
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

            combined_model.export(
                combined_stl
            )

            combined_model.export(
                combined_ply
            )

        except Exception as exc:

            print(
                f"Combined 3D reconstruction failed: {exc}"
            )

    return {
        "individual_models": generated_models,
        "combined_model": {
            "stl": combined_stl
            if combined_model is not None
            else None,
            "ply": combined_ply
            if combined_model is not None
            else None
        }
    }