import os

import numpy as np
import nibabel as nib

from scipy import ndimage
from skimage.measure import marching_cubes
from skimage.morphology import remove_small_objects

import matplotlib.pyplot as plt


# ============================================================
# HVSMR 3D HEART RECONSTRUCTION
# CLEANED + ALIGNED VERSION
# ============================================================


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

PATIENT = "pat7"

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "hvsmr",
    PATIENT
)

SEGMENTATION_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "hvsmr",
    PATIENT,
    "predicted_segmentation.nii.gz"
)


# ------------------------------------------------------------
# HVSMR LABELS
# ------------------------------------------------------------

LABELS = {
    1: "LV",
    2: "RV",
    3: "LA",
    4: "RA",
    5: "Aorta",
    6: "Pulmonary Artery",
    7: "SVC",
    8: "IVC",
}


# ------------------------------------------------------------
# CLEANUP SETTINGS
# ------------------------------------------------------------

# Remove very small isolated regions.
MIN_COMPONENT_SIZE = 100

# Light Gaussian smoothing.
GAUSSIAN_SIGMA = 0.7

# Small morphological closing to fill tiny gaps.
CLOSING_ITERATIONS = 1


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("HVSMR 3D HEART RECONSTRUCTION")
print("CLEANED + PHYSICALLY ALIGNED VERSION")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nPatient:")
print(PATIENT)

print("\nSegmentation:")
print(SEGMENTATION_FILE)


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(SEGMENTATION_FILE):

    raise FileNotFoundError(
        "\nPredicted segmentation file not found:\n"
        + SEGMENTATION_FILE
        + "\n\nRun predict_volume.py first."
    )


# ============================================================
# LOAD NIFTI
# ============================================================

print("\n" + "=" * 70)
print("LOADING PREDICTED SEGMENTATION")
print("=" * 70)

nii = nib.load(
    SEGMENTATION_FILE
)

segmentation = nii.get_fdata()

# Physical voxel spacing.
spacing = np.asarray(
    nii.header.get_zooms()[:3],
    dtype=np.float64
)

# IMPORTANT:
# Preserve the full NIfTI affine so that all reconstructed
# structures remain in the same physical coordinate system.
affine = nii.affine


print(
    "Shape:",
    segmentation.shape
)

print(
    "Voxel spacing:",
    tuple(
        float(x)
        for x in spacing
    ),
    "mm"
)

print(
    "Labels:",
    np.unique(segmentation)
)

print("\nNIfTI affine:")
print(affine)


# ============================================================
# MESH STORAGE
# ============================================================

meshes = {}


# ============================================================
# HELPER: CLEAN MASK
# ============================================================

def clean_mask(mask):
    """
    Clean a binary cardiac structure mask.

    Steps:
    1. Remove very small isolated components.
    2. Apply light morphological closing.
    3. Apply light Gaussian smoothing.
    4. Threshold back to binary.
    5. Remove tiny components again.
    """

    mask = mask.astype(bool)

    # --------------------------------------------------------
    # REMOVE SMALL ISOLATED COMPONENTS
    # --------------------------------------------------------

    mask = remove_small_objects(
        mask,
        min_size=MIN_COMPONENT_SIZE,
        connectivity=3
    )

    # --------------------------------------------------------
    # SMALL MORPHOLOGICAL CLOSING
    # --------------------------------------------------------

    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=1
    )

    mask = ndimage.binary_closing(
        mask,
        structure=structure,
        iterations=CLOSING_ITERATIONS
    )

    # --------------------------------------------------------
    # LIGHT GAUSSIAN SMOOTHING
    # --------------------------------------------------------

    smooth = ndimage.gaussian_filter(
        mask.astype(np.float32),
        sigma=GAUSSIAN_SIGMA
    )

    mask = smooth >= 0.5

    # --------------------------------------------------------
    # FINAL SMALL COMPONENT REMOVAL
    # --------------------------------------------------------

    mask = remove_small_objects(
        mask,
        min_size=MIN_COMPONENT_SIZE,
        connectivity=3
    )

    return mask


# ============================================================
# HELPER: CONVERT VOXEL COORDINATES TO PHYSICAL COORDINATES
# ============================================================

def apply_affine_to_vertices(vertices, affine):
    """
    Marching Cubes returns vertices in array coordinates.

    Convert them into the physical coordinate system defined
    by the NIfTI affine.
    """

    ones = np.ones(
        (vertices.shape[0], 1),
        dtype=np.float64
    )

    homogeneous = np.concatenate(
        [
            vertices.astype(np.float64),
            ones
        ],
        axis=1
    )

    transformed = (
        homogeneous @ affine.T
    )

    return transformed[:, :3]


# ============================================================
# RECONSTRUCT EACH STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("GENERATING CLEAN ALIGNED 3D SURFACES")
print("=" * 70)


for label, name in LABELS.items():

    print("\n" + "-" * 60)
    print(
        f"Processing label {label}: {name}"
    )
    print("-" * 60)

    # --------------------------------------------------------
    # ORIGINAL MASK
    # --------------------------------------------------------

    mask = segmentation == label

    original_voxels = int(
        np.sum(mask)
    )

    print(
        "Original voxels:",
        f"{original_voxels:,}"
    )

    if original_voxels == 0:

        print(
            "WARNING: No voxels found. Skipping."
        )

        continue

    # --------------------------------------------------------
    # CLEAN MASK
    # --------------------------------------------------------

    cleaned_mask = clean_mask(
        mask
    )

    cleaned_voxels = int(
        np.sum(cleaned_mask)
    )

    print(
        "Cleaned voxels:",
        f"{cleaned_voxels:,}"
    )

    if cleaned_voxels == 0:

        print(
            "WARNING: Empty mask after cleanup."
        )

        continue

    # --------------------------------------------------------
    # MARCHING CUBES
    # --------------------------------------------------------

    print(
        "Running Marching Cubes..."
    )

    try:

        vertices_voxel, faces, normals, values = (
            marching_cubes(
                cleaned_mask.astype(
                    np.float32
                ),
                level=0.5,
                spacing=spacing
            )
        )

    except Exception as e:

        print(
            "WARNING: Marching Cubes failed:"
        )

        print(e)

        continue

    print(
        "Vertices before affine:",
        f"{len(vertices_voxel):,}"
    )

    print(
        "Faces:",
        f"{len(faces):,}"
    )

    # --------------------------------------------------------
    # PHYSICAL ALIGNMENT
    # --------------------------------------------------------
    #
    # Marching Cubes + spacing gives physical distances,
    # but the NIfTI affine may also contain translation and
    # orientation information.
    #
    # Apply the affine so EVERY structure uses the same
    # MRI physical coordinate system.
    #

    vertices = apply_affine_to_vertices(
        vertices_voxel,
        affine
    )

    # --------------------------------------------------------
    # SAVE INDIVIDUAL MESH
    # --------------------------------------------------------

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    mesh_file = os.path.join(
        OUTPUT_DIR,
        f"{PATIENT}_{safe_name}.npz"
    )

    np.savez_compressed(
        mesh_file,
        vertices=vertices,
        faces=faces,
        normals=normals
    )

    print(
        "Saved:",
        mesh_file
    )

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    meshes[label] = {
        "name": name,
        "vertices": vertices,
        "faces": faces
    }


# ============================================================
# CHECK
# ============================================================

if len(meshes) == 0:

    raise RuntimeError(
        "\nNo cardiac structures were reconstructed."
    )


# ============================================================
# SAVE COMPLETE MESH
# ============================================================

print("\n" + "=" * 70)
print("SAVING COMPLETE MESH")
print("=" * 70)

complete_mesh_file = os.path.join(
    OUTPUT_DIR,
    f"{PATIENT}_all_meshes.npz"
)

save_data = {}

for label, mesh in meshes.items():

    save_data[
        f"label_{label}_vertices"
    ] = mesh["vertices"]

    save_data[
        f"label_{label}_faces"
    ] = mesh["faces"]


np.savez_compressed(
    complete_mesh_file,
    **save_data
)

print(
    "\nComplete mesh saved:"
)

print(
    complete_mesh_file
)


# ============================================================
# STATIC VISUALIZATION FOR VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("CREATING VALIDATION VISUALIZATION")
print("=" * 70)


fig = plt.figure(
    figsize=(12, 10)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


for label, mesh in meshes.items():

    vertices = mesh["vertices"]
    faces = mesh["faces"]

    print(
        "Displaying:",
        mesh["name"]
    )

    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        alpha=0.65,
        linewidth=0.02,
        antialiased=True
    )


# ============================================================
# AXES
# ============================================================

ax.set_xlabel(
    "X (mm)"
)

ax.set_ylabel(
    "Y (mm)"
)

ax.set_zlabel(
    "Z (mm)"
)

ax.set_title(
    f"HVSMR 3D Heart Model - {PATIENT}",
    fontsize=16,
    pad=20
)


# ============================================================
# PHYSICAL ASPECT RATIO
# ============================================================

all_vertices = np.concatenate(
    [
        mesh["vertices"]
        for mesh in meshes.values()
    ],
    axis=0
)

mins = np.min(
    all_vertices,
    axis=0
)

maxs = np.max(
    all_vertices,
    axis=0
)

dimensions = (
    maxs - mins
)

print(
    "\nPhysical model dimensions:"
)

print(
    "X:",
    f"{dimensions[0]:.2f} mm"
)

print(
    "Y:",
    f"{dimensions[1]:.2f} mm"
)

print(
    "Z:",
    f"{dimensions[2]:.2f} mm"
)


try:

    ax.set_box_aspect(
        dimensions
    )

except Exception:

    pass


# ============================================================
# CAMERA
# ============================================================

ax.view_init(
    elev=20,
    azim=-60
)


# ============================================================
# SAVE IMAGE
# ============================================================

figure_file = os.path.join(
    OUTPUT_DIR,
    f"{PATIENT}_3d_heart_predicted.png"
)

plt.tight_layout()

plt.savefig(
    figure_file,
    dpi=200,
    bbox_inches="tight"
)

print(
    "\nValidation image saved:"
)

print(
    figure_file
)


# ============================================================
# DISPLAY
# ============================================================

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RECONSTRUCTION COMPLETED")
print("=" * 70)

print(
    "\nPatient:",
    PATIENT
)

print(
    "Input:",
    "U-Net predicted segmentation"
)

print(
    "Volume shape:",
    segmentation.shape
)

print(
    "Voxel spacing:",
    tuple(
        float(x)
        for x in spacing
    ),
    "mm"
)

print(
    "\nStructures reconstructed:"
)

for label, mesh in meshes.items():

    print(
        f"  Label {label}: "
        f"{mesh['name']} "
        f"({len(mesh['vertices']):,} vertices, "
        f"{len(mesh['faces']):,} faces)"
    )

print(
    "\nAll structures use the same NIfTI physical "
    "coordinate system."
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nDone!"
)

print("=" * 70)