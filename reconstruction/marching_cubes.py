"""
Digital 3D Heart Modelling
--------------------------
3D mesh reconstruction from a segmented ACDC volume.

Labels:
    0 = Background
    1 = RV
    2 = MYO
    3 = LV

Input:
    outputs/patient101_sax_ed_volume.npy

Original image:
    dataset/ACDC/test/patient101/patient101_sax_ed.nii.gz

Outputs:
    outputs/rv_vertices.npy
    outputs/rv_faces.npy
    outputs/rv_normals.npy

    outputs/myo_vertices.npy
    outputs/myo_faces.npy
    outputs/myo_normals.npy

    outputs/lv_vertices.npy
    outputs/lv_faces.npy
    outputs/lv_normals.npy
"""

import os
import sys
import traceback

import numpy as np
import nibabel as nib

from scipy import ndimage
from skimage.measure import marching_cubes


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "ACDC"
)

SEGMENTATION_PATH = os.path.join(
    OUTPUT_DIR,
    "patient101_sax_ed_volume.npy"
)

NIFTI_PATH = os.path.join(
    DATASET_DIR,
    "test",
    "patient101",
    "patient101_sax_ed.nii.gz"
)


# ============================================================
# LABEL CONFIGURATION
# ============================================================

LABELS = {
    1: "rv",
    2: "myo",
    3: "lv",
}


# ============================================================
# RECONSTRUCTION PARAMETERS
# ============================================================

# Target Z spacing in millimetres.
#
# Original ACDC spacing:
#     X = 1.0 mm
#     Y = 1.0 mm
#     Z = 10.0 mm
#
# Target:
#     Z = 1.25 mm
#
# Therefore:
#     10.0 / 1.25 = 8x interpolation
#
TARGET_Z_SPACING = 1.25

# Remove connected components smaller than this number
# of voxels.
MIN_OBJECT_SIZE = 20

# Isosurface threshold for binary mask.
MARCHING_LEVEL = 0.5


# ============================================================
# PRINT HELPERS
# ============================================================

def print_separator(char="=", length=68):
    print(char * length)


def print_section(title):
    print()
    print_separator("=")
    print(title)
    print_separator("=")


# ============================================================
# PATH CHECK
# ============================================================

def check_input_files(
    volume_path,
    nifti_path
):
    """
    Check that all required input files exist.
    """

    print_section("Checking input files...")

    if not os.path.exists(volume_path):

        raise FileNotFoundError(
            "\nSegmentation file not found:\n"
            f"{volume_path}\n\n"
            "Run the segmentation/prediction pipeline first."
        )

    if not os.path.exists(nifti_path):

        raise FileNotFoundError(
            "\nOriginal NIfTI file not found:\n"
            f"{nifti_path}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("Input files found.")


# ============================================================
# LOAD SEGMENTATION
# ============================================================

def load_segmentation(path):
    """
    Load the segmentation volume.
    """

    print("Loading segmentation:")
    print(path)

    volume = np.load(path)

    if volume.ndim != 3:

        raise ValueError(
            "Expected a 3D segmentation volume, "
            f"but received shape {volume.shape}"
        )

    volume = np.asarray(volume)

    labels = np.unique(volume)

    print()
    print(f"Volume Shape : {volume.shape}")
    print(f"Labels Found : {labels}")

    return volume


# ============================================================
# LOAD ORIGINAL NIFTI
# ============================================================

def load_nifti(path):
    """
    Load original NIfTI image and obtain voxel spacing.
    """

    print_section("Loading original NIfTI:")

    print(path)

    nii = nib.load(path)

    image = nii.get_fdata()

    spacing = np.asarray(
        nii.header.get_zooms()[:3],
        dtype=np.float32
    )

    print()
    print(f"Original NIfTI shape   : {image.shape}")
    print(
        "Original voxel spacing : "
        f"{tuple(float(x) for x in spacing)}"
    )

    return nii, image, spacing


# ============================================================
# DIMENSION CHECK
# ============================================================

def check_dimensions(
    segmentation,
    nifti_image
):
    """
    Verify that segmentation and NIfTI have matching
    spatial dimensions.
    """

    print_section("Checking dimensions:")

    print(
        f"Segmentation shape : "
        f"{segmentation.shape}"
    )

    spacing = np.asarray(
        nifti_image.header.get_zooms()[:3],
        dtype=np.float32
    )

    print(
        "NIfTI voxel spacing: "
        f"{tuple(float(x) for x in spacing)}"
    )

    if segmentation.shape != nifti_image.shape:

        raise ValueError(
            "\nDimension mismatch!\n"
            f"Segmentation: {segmentation.shape}\n"
            f"NIfTI       : {nifti_image.shape}"
        )


# ============================================================
# LARGEST CONNECTED COMPONENT
# ============================================================

def keep_largest_component(mask):
    """
    Keep only the largest connected component.

    scipy.ndimage is used instead of
    skimage.morphology to avoid version-specific
    API problems.
    """

    print("Finding largest connected component...")

    if not np.any(mask):

        return np.zeros_like(
            mask,
            dtype=bool
        )

    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=2
    )

    labeled, number_of_components = ndimage.label(
        mask,
        structure=structure
    )

    if number_of_components == 0:

        return np.zeros_like(
            mask,
            dtype=bool
        )

    component_sizes = np.bincount(
        labeled.ravel()
    )

    # Ignore background.
    component_sizes[0] = 0

    largest_label = int(
        np.argmax(component_sizes)
    )

    largest_size = int(
        component_sizes[largest_label]
    )

    print(
        "Largest component size : "
        f"{float(largest_size):.1f}"
    )

    cleaned = (
        labeled == largest_label
    )

    return cleaned.astype(bool)


# ============================================================
# FILL HOLES
# ============================================================

def fill_holes(mask):
    """
    Fill holes inside a 3D binary mask.
    """

    print("Filling holes...")

    filled = ndimage.binary_fill_holes(
        mask
    )

    return filled.astype(bool)


# ============================================================
# REMOVE SMALL COMPONENTS
# ============================================================

def remove_small_components(
    mask,
    min_size=MIN_OBJECT_SIZE
):
    """
    Remove connected components smaller
    than min_size voxels.
    """

    print("Removing small objects...")

    if min_size <= 0:

        return mask

    if not np.any(mask):

        return mask

    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=2
    )

    labeled, number_of_components = ndimage.label(
        mask,
        structure=structure
    )

    if number_of_components == 0:

        return np.zeros_like(
            mask,
            dtype=bool
        )

    component_sizes = np.bincount(
        labeled.ravel()
    )

    keep = (
        component_sizes >= min_size
    )

    # Never keep background.
    keep[0] = False

    cleaned = keep[labeled]

    return cleaned.astype(bool)


# ============================================================
# CLEAN SEGMENTATION MASK
# ============================================================

def clean_mask(mask):
    """
    Perform basic segmentation cleanup.
    """

    mask = np.asarray(
        mask,
        dtype=bool
    )

    print(
        "Initial voxels         : "
        f"{int(np.count_nonzero(mask))}"
    )

    # --------------------------------------------------------
    # Largest connected component
    # --------------------------------------------------------

    mask = keep_largest_component(
        mask
    )

    # --------------------------------------------------------
    # Fill holes
    # --------------------------------------------------------

    mask = fill_holes(
        mask
    )

    # --------------------------------------------------------
    # Remove small components
    # --------------------------------------------------------

    mask = remove_small_components(
        mask,
        MIN_OBJECT_SIZE
    )

    print(
        "Voxels after cleanup   : "
        f"{int(np.count_nonzero(mask))}"
    )

    return mask


# ============================================================
# Z INTERPOLATION
# ============================================================

def interpolate_z(
    mask,
    original_spacing,
    target_z_spacing=TARGET_Z_SPACING
):
    """
    Interpolate the binary segmentation along Z.

    The important point is that the output spacing is
    explicitly set to TARGET_Z_SPACING.

    For example:

        Original Z spacing = 10.0 mm
        Target Z spacing   = 1.25 mm

        Interpolation factor = 10.0 / 1.25 = 8

    Therefore:

        (192, 192, 10)
                    ->
        (192, 192, 80)

    and the physical spacing is:

        (1.0, 1.0, 1.25) mm
    """

    original_spacing = np.asarray(
        original_spacing,
        dtype=np.float32
    )

    original_z_spacing = float(
        original_spacing[2]
    )

    target_z_spacing = float(
        target_z_spacing
    )

    if original_z_spacing <= 0:

        raise ValueError(
            "Invalid original Z spacing: "
            f"{original_z_spacing}"
        )

    if target_z_spacing <= 0:

        raise ValueError(
            "Invalid target Z spacing: "
            f"{target_z_spacing}"
        )

    # --------------------------------------------------------
    # Calculate Z interpolation factor
    # --------------------------------------------------------

    z_factor = (
        original_z_spacing /
        target_z_spacing
    )

    print(
        "Z interpolation factor : "
        f"{z_factor:.4f}"
    )

    # --------------------------------------------------------
    # No interpolation required
    # --------------------------------------------------------

    if np.isclose(
        z_factor,
        1.0,
        rtol=1e-5,
        atol=1e-5
    ):

        interpolated = mask.astype(
            bool
        )

        new_spacing = np.array(
            original_spacing,
            dtype=np.float32
        )

        print(
            "Interpolated Shape    : "
            f"{interpolated.shape}"
        )

        print(
            "Final voxel count     : "
            f"{int(np.count_nonzero(interpolated))}"
        )

        print(
            "New voxel spacing     : "
            f"{tuple(float(x) for x in new_spacing)}"
        )

        return (
            interpolated,
            new_spacing
        )

    # --------------------------------------------------------
    # Interpolate ONLY the Z axis.
    #
    # X = 1
    # Y = 1
    # Z = calculated factor
    # --------------------------------------------------------

    zoom_factors = (
        1.0,
        1.0,
        z_factor
    )

    print(
        "Interpolating using zoom factors: "
        f"{zoom_factors}"
    )

    interpolated = ndimage.zoom(
        mask.astype(np.float32),
        zoom=zoom_factors,
        order=1,
        mode="nearest",
        prefilter=True
    )

    # --------------------------------------------------------
    # Convert interpolated grayscale volume back into
    # a binary segmentation.
    # --------------------------------------------------------

    interpolated = (
        interpolated >= 0.5
    )

    interpolated = interpolated.astype(
        bool
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT calculate:
    #
    #     (N - 1) * spacing / (new_N - 1)
    #
    # because that changes the requested voxel spacing
    # from 1.25 mm to approximately 1.139 mm.
    #
    # The desired voxel spacing is explicitly:
    #
    #     target_z_spacing = 1.25 mm
    # --------------------------------------------------------

    new_spacing = np.array(
        [
            float(original_spacing[0]),
            float(original_spacing[1]),
            target_z_spacing
        ],
        dtype=np.float32
    )

    print(
        "Interpolated Shape    : "
        f"{interpolated.shape}"
    )

    print(
        "Final voxel count     : "
        f"{int(np.count_nonzero(interpolated))}"
    )

    print(
        "New voxel spacing     : "
        f"{tuple(float(x) for x in new_spacing)}"
    )

    return (
        interpolated,
        new_spacing
    )


# ============================================================
# MARCHING CUBES
# ============================================================

def generate_mesh(
    mask,
    spacing
):
    """
    Generate a triangular surface mesh using
    Marching Cubes.
    """

    if not np.any(mask):

        raise ValueError(
            "Mask is empty. Cannot generate mesh."
        )

    print("Running Marching Cubes...")

    spacing = tuple(
        float(x)
        for x in spacing
    )

    vertices, faces, normals, values = (
        marching_cubes(
            volume=mask.astype(
                np.float32
            ),
            level=MARCHING_LEVEL,
            spacing=spacing
        )
    )

    vertices = np.asarray(
        vertices,
        dtype=np.float32
    )

    faces = np.asarray(
        faces,
        dtype=np.int32
    )

    normals = np.asarray(
        normals,
        dtype=np.float32
    )

    print()
    print("Mesh generated successfully.")
    print()

    print(
        f"Vertices : {vertices.shape}"
    )

    print(
        f"Faces    : {faces.shape}"
    )

    print(
        f"Normals  : {normals.shape}"
    )

    return (
        vertices,
        faces,
        normals
    )


# ============================================================
# SAVE MESH
# ============================================================

def save_mesh(
    name,
    vertices,
    faces,
    normals
):
    """
    Save mesh arrays as NPY files.
    """

    vertices_path = os.path.join(
        OUTPUT_DIR,
        f"{name}_vertices.npy"
    )

    faces_path = os.path.join(
        OUTPUT_DIR,
        f"{name}_faces.npy"
    )

    normals_path = os.path.join(
        OUTPUT_DIR,
        f"{name}_normals.npy"
    )

    np.save(
        vertices_path,
        vertices
    )

    np.save(
        faces_path,
        faces
    )

    np.save(
        normals_path,
        normals
    )

    print()
    print("Saved files:")

    print(
        f"    {vertices_path}"
    )

    print(
        f"    {faces_path}"
    )

    print(
        f"    {normals_path}"
    )

    return (
        vertices_path,
        faces_path,
        normals_path
    )


# ============================================================
# PROCESS ONE LABEL
# ============================================================

def process_label(
    segmentation,
    label,
    name,
    original_spacing
):
    """
    Process one anatomical structure.

    Labels:
        1 = RV
        2 = MYO
        3 = LV
    """

    print_section(
        f"Processing label {label}: "
        f"{name.upper()}"
    )

    # --------------------------------------------------------
    # Extract binary mask
    # --------------------------------------------------------

    mask = (
        segmentation == label
    )

    mask_voxels = int(
        np.count_nonzero(mask)
    )

    print(
        "Mask voxels            : "
        f"{mask_voxels}"
    )

    if mask_voxels == 0:

        raise ValueError(
            f"No voxels found for label "
            f"{label} ({name})."
        )

    print()
    print("-" * 60)

    print(
        f"Generating {name.upper()} mesh..."
    )

    print("-" * 60)

    # --------------------------------------------------------
    # Clean mask
    # --------------------------------------------------------

    mask = clean_mask(
        mask
    )

    # --------------------------------------------------------
    # Z interpolation
    # --------------------------------------------------------

    print(
        "Interpolating Z direction..."
    )

    interpolated_mask, new_spacing = (
        interpolate_z(
            mask=mask,
            original_spacing=original_spacing,
            target_z_spacing=TARGET_Z_SPACING
        )
    )

    # --------------------------------------------------------
    # Generate mesh
    # --------------------------------------------------------

    vertices, faces, normals = (
        generate_mesh(
            mask=interpolated_mask,
            spacing=new_spacing
        )
    )

    # --------------------------------------------------------
    # Save mesh
    # --------------------------------------------------------

    save_mesh(
        name=name,
        vertices=vertices,
        faces=faces,
        normals=normals
    )

    print()

    print(
        f"{name.upper()} mesh saved successfully."
    )

    return {
        "name": name,
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "spacing": new_spacing
    }


# ============================================================
# MAIN RECONSTRUCTION FUNCTION
# ============================================================

def reconstruct(
    volume_path=SEGMENTATION_PATH,
    nifti_path=NIFTI_PATH
):
    """
    Main heart reconstruction pipeline.
    """

    print()
    print_separator("=")
    print("Digital 3D Heart Reconstruction")
    print_separator("=")

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    check_input_files(
        volume_path,
        nifti_path
    )

    # --------------------------------------------------------
    # Load segmentation
    # --------------------------------------------------------

    segmentation = load_segmentation(
        volume_path
    )

    # --------------------------------------------------------
    # Load NIfTI
    # --------------------------------------------------------

    (
        nii,
        nifti_image,
        original_spacing
    ) = load_nifti(
        nifti_path
    )

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    check_dimensions(
        segmentation,
        nii
    )

    # --------------------------------------------------------
    # Process anatomical structures
    # --------------------------------------------------------

    successful_meshes = []
    failed_meshes = []

    for label, name in LABELS.items():

        try:

            process_label(
                segmentation=segmentation,
                label=label,
                name=name,
                original_spacing=original_spacing
            )

            successful_meshes.append(
                name
            )

        except Exception as error:

            failed_meshes.append(
                name
            )

            print()
            print(
                f"ERROR while processing "
                f"{name.upper()}:"
            )

            print(
                str(error)
            )

            print()

            traceback.print_exc()

            print()

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print_separator("=")

    print(
        "Digital 3D Heart Reconstruction "
        "Completed"
    )

    print_separator("=")

    print()

    print(
        "Successful meshes : "
        f"{successful_meshes}"
    )

    print(
        "Failed meshes     : "
        f"{failed_meshes}"
    )

    print()

    print("Output folder:")
    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )

    print()
    print_separator("=")

    if len(failed_meshes) == 0:

        print()
        print(
            "All requested meshes were "
            "generated successfully."
        )

    else:

        print()
        print(
            "Some meshes failed. "
            "Check the error messages above."
        )

    print()

    return (
        successful_meshes,
        failed_meshes
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        reconstruct(
            volume_path=SEGMENTATION_PATH,
            nifti_path=NIFTI_PATH
        )

    except KeyboardInterrupt:

        print()
        print(
            "Reconstruction interrupted by user."
        )

        sys.exit(1)

    except Exception as error:

        print()
        print_separator("=")
        print("RECONSTRUCTION FAILED")
        print_separator("=")

        print()
        print(
            str(error)
        )

        print()

        traceback.print_exc()

        sys.exit(1)