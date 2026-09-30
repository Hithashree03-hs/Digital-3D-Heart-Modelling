import os
import json
import warnings

import numpy as np
import nibabel as nib

from scipy import ndimage
from skimage.measure import marching_cubes
import trimesh


# ============================================================
# Cardiac structure labels from HVSMR
# ============================================================

STRUCTURE_LABELS = {
    1: "Left Ventricle",
    2: "Right Ventricle",
    3: "Left Atrium",
    4: "Right Atrium",
    5: "Aorta",
    6: "Pulmonary Artery",
    7: "Superior Vena Cava",
    8: "Inferior Vena Cava",
}


# Structures are grouped because thin vessels need more
# conservative cleanup than the large chambers.
CHAMBER_LABELS = {1, 2, 3, 4}
VESSEL_LABELS = {5, 6, 7, 8}


# ============================================================
# Utility functions
# ============================================================

def _safe_spacing(spacing):
    """
    Return valid positive voxel spacing.
    """
    spacing = np.asarray(spacing, dtype=float)

    if spacing.size < 3:
        return np.array([1.0, 1.0, 1.0], dtype=float)

    spacing = spacing[:3]

    spacing[~np.isfinite(spacing)] = 1.0
    spacing[spacing <= 0] = 1.0

    return spacing


def _make_structure_kernel():
    """
    Small 3D connectivity kernel.

    This is intentionally conservative. We do not use a large
    morphological kernel because it could merge nearby anatomy.
    """
    return ndimage.generate_binary_structure(3, 2)


def _component_sizes(mask):
    """
    Return connected-component labels and component sizes.
    """
    connectivity = _make_structure_kernel()

    component_map, count = ndimage.label(
        mask,
        structure=connectivity
    )

    if count == 0:
        return component_map, np.array([], dtype=np.int64)

    sizes = np.bincount(component_map.ravel())

    # Ignore background.
    return component_map, sizes[1:]


def _keep_relevant_components(mask, label):
    """
    Remove tiny isolated fragments while preserving nearby components.

    Strategy:
      1. Apply a very small closing to repair 1-voxel gaps.
      2. Find connected components.
      3. Keep the dominant component.
      4. Keep reasonably large secondary components.
      5. Keep components that are spatially close to the dominant
         component.
      6. Perform one final conservative closing.

    This is intentionally less aggressive for vessels.
    """

    if not np.any(mask):
        return mask.astype(bool)

    original_voxels = int(mask.sum())

    # --------------------------------------------------------
    # Step 1: very small closing
    # --------------------------------------------------------
    kernel = _make_structure_kernel()

    repaired = ndimage.binary_closing(
        mask,
        structure=kernel,
        iterations=1
    )

    # Never allow cleanup to create a structure from nothing.
    if not np.any(repaired):
        repaired = mask.copy()

    # --------------------------------------------------------
    # Step 2: fill enclosed holes
    # --------------------------------------------------------
    repaired = ndimage.binary_fill_holes(repaired)

    # --------------------------------------------------------
    # Step 3: connected components
    # --------------------------------------------------------
    component_map, sizes = _component_sizes(repaired)

    if len(sizes) <= 1:
        return repaired.astype(bool)

    order = np.argsort(sizes)[::-1]

    largest_id = int(order[0] + 1)
    largest_size = int(sizes[order[0]])

    # --------------------------------------------------------
    # Different thresholds for chambers and vessels.
    #
    # Vessels are thinner and naturally more fragmented, so
    # their secondary components are treated more carefully.
    # --------------------------------------------------------
    if label in CHAMBER_LABELS:
        relative_secondary_threshold = 0.035
        proximity_iterations = 3
        minimum_voxels = 30
    else:
        relative_secondary_threshold = 0.015
        proximity_iterations = 4
        minimum_voxels = 15

    # --------------------------------------------------------
    # Step 4: region around dominant component.
    #
    # Components touching this region are considered potentially
    # related rather than immediately discarded.
    # --------------------------------------------------------
    largest_component = component_map == largest_id

    nearby_region = ndimage.binary_dilation(
        largest_component,
        structure=kernel,
        iterations=proximity_iterations
    )

    keep_ids = {largest_id}

    for component_index in order[1:]:
        component_id = int(component_index + 1)
        component_size = int(sizes[component_index])

        # Remove extremely tiny fragments.
        if component_size < minimum_voxels:
            continue

        # Keep a component if it is reasonably large.
        if component_size >= largest_size * relative_secondary_threshold:
            keep_ids.add(component_id)
            continue

        # Keep small components if they are directly near the main
        # anatomical component. This helps repair slice-to-slice gaps.
        component_mask = component_map == component_id

        if np.any(component_mask & nearby_region):
            keep_ids.add(component_id)

    # --------------------------------------------------------
    # Build cleaned structure
    # --------------------------------------------------------
    cleaned = np.isin(component_map, list(keep_ids))

    # --------------------------------------------------------
    # Step 5: final conservative closing
    # --------------------------------------------------------
    cleaned = ndimage.binary_closing(
        cleaned,
        structure=kernel,
        iterations=1
    )

    cleaned = ndimage.binary_fill_holes(cleaned)

    # Safety check:
    # If cleanup somehow removed almost everything, fall back
    # to the original prediction rather than destroying anatomy.
    cleaned_voxels = int(cleaned.sum())

    if cleaned_voxels < max(10, int(original_voxels * 0.25)):
        return mask.astype(bool)

    return cleaned.astype(bool)


def refine_segmentation(prediction):
    """
    Refine the complete predicted label volume.

    Input:
        prediction: 3D integer label volume

    Output:
        refined 3D integer label volume

    Labels:
        0 = background
        1 = LV
        2 = RV
        3 = LA
        4 = RA
        5 = Aorta
        6 = Pulmonary artery
        7 = SVC
        8 = IVC
    """

    prediction = np.asarray(prediction)

    if prediction.ndim != 3:
        raise ValueError(
            f"Expected a 3D prediction volume, got shape {prediction.shape}"
        )

    prediction = np.rint(prediction).astype(np.int16)

    refined = np.zeros_like(prediction, dtype=np.int16)

    print("\n" + "=" * 60)
    print("3D SEGMENTATION REFINEMENT")
    print("=" * 60)

    print("Input shape:", prediction.shape)
    print("Input labels:", np.unique(prediction))

    for label, name in STRUCTURE_LABELS.items():

        structure_mask = prediction == label

        original_count = int(structure_mask.sum())

        if original_count == 0:
            print(f"{name:<22} no voxels")
            continue

        cleaned_mask = _keep_relevant_components(
            structure_mask,
            label
        )

        cleaned_count = int(cleaned_mask.sum())

        # ----------------------------------------------------
        # Write refined structure.
        #
        # Because predictions can theoretically overlap after
        # morphology, only write into currently empty voxels.
        # The original label order is preserved.
        # ----------------------------------------------------
        available = cleaned_mask & (refined == 0)

        refined[available] = label

        print(
            f"{name:<22} "
            f"{original_count:>8} -> {cleaned_count:>8} voxels"
        )

    print("Refined labels:", np.unique(refined))
    print("=" * 60)

    return refined


# ============================================================
# Mesh cleanup
# ============================================================

def _clean_mesh(mesh):
    """
    Conservative mesh cleanup compatible with newer trimesh
    versions.

    Avoids deprecated remove_duplicate_faces().
    """

    if mesh is None:
        return None

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    # Remove obviously invalid geometry.
    try:
        mesh.update_faces(mesh.unique_faces())
    except Exception:
        pass

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    # Remove degenerate faces where supported.
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        pass

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    # Fix normals.
    try:
        mesh.fix_normals()
    except Exception:
        pass

    return mesh



def _smooth_mesh(mesh, structure_name):
    """
    Smooth a reconstructed surface while preserving its overall shape.

    Taubin smoothing is used instead of ordinary Laplacian smoothing
    because it reduces staircase/faceted artifacts with less shrinkage.

    Large chambers receive moderate smoothing; thin vessels receive
    fewer iterations so small anatomical structures are not erased.
    """

    if mesh is None or len(mesh.vertices) == 0:
        return mesh

    # Conservative settings:
    # chambers -> smoother surface
    # vessels  -> preserve thin anatomy
    if structure_name in STRUCTURE_LABELS.values():
        if structure_name in {
            "Left Ventricle",
            "Right Ventricle",
            "Left Atrium",
            "Right Atrium",
        }:
            iterations = 8
        else:
            iterations = 4
    else:
        iterations = 6

    try:
        # Taubin smoothing suppresses high-frequency staircase artifacts
        # while reducing the volume shrinkage of ordinary Laplacian smoothing.
        trimesh.smoothing.filter_taubin(
            mesh,
            lamb=0.45,
            nu=0.50,
            iterations=iterations,
        )
    except Exception as exc:
        print(
            f"Mesh smoothing skipped for {structure_name}: {exc}"
        )

    # Re-clean after smoothing because smoothing can leave unused
    # vertices or geometry that needs normal correction.
    mesh = _clean_mesh(mesh)

    try:
        mesh.fix_normals()
    except Exception:
        pass

    return mesh


def _create_mesh(mask, spacing, structure_name):
    """
    Convert a binary structure volume into a surface mesh.
    """

    if not np.any(mask):
        return None

    # Marching cubes needs at least a small amount of surrounding
    # background. The padding prevents edge-cut surfaces.
    padded = np.pad(
        mask.astype(np.uint8),
        pad_width=1,
        mode="constant",
        constant_values=0
    )

    try:
        vertices, faces, normals, values = marching_cubes(
            padded,
            level=0.5,
            spacing=spacing
        )
    except Exception as exc:
        warnings.warn(
            f"Marching cubes failed for {structure_name}: {exc}"
        )
        return None

    # Because of the one-voxel padding, shift coordinates back
    # by one voxel in physical units.
    vertices -= spacing

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False
    )

    mesh = _clean_mesh(mesh)

    if mesh is None or len(mesh.vertices) == 0:
        return None

    # Smooth the surface after marching cubes. This improves the
    # staircase/faceted appearance while preserving patient-specific
    # anatomical positioning.
    mesh = _smooth_mesh(mesh, structure_name)

    if mesh is None or len(mesh.vertices) == 0:
        return None

    return mesh


# ============================================================
# Main reconstruction function
# ============================================================

def reconstruct_3d(
    prediction,
    spacing,
    output_dir,
    patient_id
):
    """
    Complete 3D reconstruction pipeline.

    Parameters
    ----------
    prediction : numpy.ndarray
        3D integer segmentation volume.

    spacing : tuple/list
        MRI voxel spacing in mm.

    output_dir : str
        Directory where 3D models will be stored.

    patient_id : str
        Patient identifier.

    Returns
    -------
    dict
        Reconstruction information compatible with the backend
        pipeline.
    """

    os.makedirs(output_dir, exist_ok=True)

    spacing = _safe_spacing(spacing)

    prediction = np.asarray(prediction)

    if prediction.ndim != 3:
        raise ValueError(
            f"Prediction must be 3D. Received {prediction.shape}"
        )

    print("\n")
    print("=" * 70)
    print("3D HEART RECONSTRUCTION")
    print("=" * 70)
    print("Patient:", patient_id)
    print("Prediction shape:", prediction.shape)
    print("Voxel spacing:", tuple(spacing))
    print("Original labels:", np.unique(prediction))

    # --------------------------------------------------------
    # REFINE SEGMENTATION
    # --------------------------------------------------------
    refined_prediction = refine_segmentation(prediction)

    # Save refined segmentation for debugging/reproducibility.
    refined_path = os.path.join(
        output_dir,
        "refined_segmentation.npy"
    )

    np.save(
        refined_path,
        refined_prediction.astype(np.int16)
    )

    # --------------------------------------------------------
    # Individual structure meshes
    # --------------------------------------------------------
    individual_models = []

    all_meshes = []

    for label, structure_name in STRUCTURE_LABELS.items():

        print("\n" + "-" * 60)
        print(f"Reconstructing: {structure_name}")
        print("-" * 60)

        mask = refined_prediction == label

        voxel_count = int(mask.sum())

        print("Voxel count:", voxel_count)

        if voxel_count < 10:
            print("Skipped: insufficient voxels")
            continue

        mesh = _create_mesh(
            mask,
            spacing,
            structure_name
        )

        if mesh is None:
            print("Mesh generation failed")
            continue

        # ----------------------------------------------------
        # Validate mesh
        # ----------------------------------------------------
        try:
            is_watertight = bool(mesh.is_watertight)
        except Exception:
            is_watertight = False

        print("Vertices:", len(mesh.vertices))
        print("Faces:", len(mesh.faces))
        print("Watertight:", is_watertight)

        safe_name = (
            structure_name
            .lower()
            .replace(" ", "_")
        )

        stl_path = os.path.join(
            output_dir,
            f"{safe_name}.stl"
        )

        ply_path = os.path.join(
            output_dir,
            f"{safe_name}.ply"
        )

        try:
            mesh.export(stl_path)
            mesh.export(ply_path)
        except Exception as exc:
            print("Mesh export failed:", exc)
            continue

        individual_models.append({
            "label": int(label),
            "name": structure_name,
            "stl": stl_path,
            "ply": ply_path,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": is_watertight,
            "voxel_count": voxel_count,
        })

        all_meshes.append(mesh)

        print("STL:", stl_path)
        print("PLY:", ply_path)

    # --------------------------------------------------------
    # Combined heart surface
    # --------------------------------------------------------
    combined_model = {
        "stl": None,
        "ply": None
    }

    if all_meshes:

        print("\n" + "=" * 60)
        print("GENERATING COMBINED HEART")
        print("=" * 60)

        try:
            combined = trimesh.util.concatenate(all_meshes)

            combined = _clean_mesh(combined)

            combined_stl = os.path.join(
                output_dir,
                "combined_heart.stl"
            )

            combined_ply = os.path.join(
                output_dir,
                "combined_heart.ply"
            )

            combined.export(combined_stl)
            combined.export(combined_ply)

            combined_model = {
                "stl": combined_stl,
                "ply": combined_ply,
                "vertices": int(len(combined.vertices)),
                "faces": int(len(combined.faces)),
            }

            print("Combined vertices:", len(combined.vertices))
            print("Combined faces:", len(combined.faces))
            print("Combined STL:", combined_stl)
            print("Combined PLY:", combined_ply)

        except Exception as exc:
            print("Combined mesh generation failed:", exc)

    # --------------------------------------------------------
    # Reconstruction result
    # --------------------------------------------------------
    result = {
        "patient_id": patient_id,
        "prediction_shape": [
            int(x) for x in prediction.shape
        ],
        "voxel_spacing": [
            float(x) for x in spacing
        ],
        "refined_segmentation": refined_path,
        "individual_models": individual_models,
        "combined_model": combined_model,
        "model_count": len(individual_models),
        "mesh_processing": {
            "surface_smoothing": "Taubin",
            "chamber_iterations": 8,
            "vessel_iterations": 4,
            "lambda": 0.45,
            "nu": 0.50,
        },
    }

    # Save reconstruction metadata.
    metadata_path = os.path.join(
        output_dir,
        "reconstruction.json"
    )

    try:
        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                result,
                f,
                indent=2
            )
    except Exception as exc:
        print("Could not save reconstruction metadata:", exc)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("3D RECONSTRUCTION COMPLETE")
    print("=" * 70)

    print(
        f"Structures generated: "
        f"{len(individual_models)} / {len(STRUCTURE_LABELS)}"
    )

    for item in individual_models:
        print(
            f"  ✓ {item['name']:<22} "
            f"{item['vertices']:>7} vertices | "
            f"{item['faces']:>7} faces"
        )

    if combined_model["stl"]:
        print("✓ Combined heart generated")

    print("=" * 70)

    return result


# ============================================================
# Optional helper: reconstruct directly from NIfTI
# ============================================================

def reconstruct_from_nifti(
    nifti_path,
    output_dir,
    patient_id
):
    """
    Convenience function for testing reconstruction directly
    from a NIfTI segmentation file.
    """

    nii = nib.load(nifti_path)

    prediction = np.asarray(
        nii.dataobj,
        dtype=np.int16
    )

    spacing = nii.header.get_zooms()[:3]

    return reconstruct_3d(
        prediction=prediction,
        spacing=spacing,
        output_dir=output_dir,
        patient_id=patient_id
    )


# ============================================================
# Direct test
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="HVSMR 3D reconstruction test"
    )

    parser.add_argument(
        "--seg",
        required=True,
        help="Path to segmentation NIfTI file"
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--patient",
        default="test_patient",
        help="Patient identifier"
    )

    args = parser.parse_args()

    reconstruct_from_nifti(
        nifti_path=args.seg,
        output_dir=args.out,
        patient_id=args.patient
    )