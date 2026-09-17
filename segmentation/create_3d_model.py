import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from skimage.measure import marching_cubes
import trimesh


# =================================================
# Paths
# =================================================

PREDICTION_PATH = "outputs/pat15_prediction.npy"

ORIGINAL_MRI_PATH = (
    "../dataset/HVSMR/cropped/cropped/"
    "pat15_cropped.nii.gz"
)

OUTPUT_DIR = "outputs/3d_models"


# =================================================
# Create output folder
# =================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =================================================
# Load prediction
# =================================================

prediction = np.load(
    PREDICTION_PATH
)

print("Prediction shape:", prediction.shape)


# =================================================
# Load original MRI
# =================================================

mri_nii = nib.load(
    ORIGINAL_MRI_PATH
)

original_shape = mri_nii.shape

print(
    "Original volume shape:",
    original_shape
)

print(
    "Voxel spacing:",
    mri_nii.header.get_zooms()[:3]
)


# =================================================
# Resize prediction back to original dimensions
# =================================================

print("\nResizing prediction to original dimensions...")

prediction_tensor = torch.from_numpy(
    prediction.astype(np.float32)
).unsqueeze(0).unsqueeze(0)

prediction_resized = F.interpolate(
    prediction_tensor,
    size=original_shape,
    mode="nearest"
)

prediction_resized = (
    prediction_resized
    .squeeze()
    .numpy()
    .astype(np.int16)
)

print(
    "Resized prediction shape:",
    prediction_resized.shape
)


# =================================================
# Get voxel spacing
# =================================================

spacing = mri_nii.header.get_zooms()[:3]

print(
    "Using voxel spacing:",
    spacing
)


# =================================================
# Create meshes for classes 1-8
# =================================================

labels = sorted(
    np.unique(prediction_resized)
)

labels = [
    int(x)
    for x in labels
    if int(x) != 0
]

print(
    "\nLabels to reconstruct:",
    labels
)


for label in labels:

    print(
        f"\nCreating 3D mesh for label {label}..."
    )

    binary_mask = (
        prediction_resized == label
    )

    voxel_count = binary_mask.sum()

    print(
        "Voxel count:",
        voxel_count
    )

    if voxel_count < 10:

        print(
            "Skipping very small structure."
        )

        continue

    # ---------------------------------------------
    # Marching cubes
    # ---------------------------------------------

    vertices, faces, normals, values = (
        marching_cubes(
            binary_mask.astype(np.uint8),
            level=0.5,
            spacing=spacing
        )
    )

    print(
        "Vertices:",
        len(vertices)
    )

    print(
        "Faces:",
        len(faces)
    )

    # ---------------------------------------------
    # Create mesh
    # ---------------------------------------------

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=True
    )

    # ---------------------------------------------
    # Save STL
    # ---------------------------------------------

    stl_path = os.path.join(
        OUTPUT_DIR,
        f"pat15_label_{label}.stl"
    )

    mesh.export(
        stl_path
    )

    print(
        "Saved:",
        stl_path
    )

    # ---------------------------------------------
    # Save PLY
    # ---------------------------------------------

    ply_path = os.path.join(
        OUTPUT_DIR,
        f"pat15_label_{label}.ply"
    )

    mesh.export(
        ply_path
    )

    print(
        "Saved:",
        ply_path
    )


# =================================================
# Combined heart model
# =================================================

print("\nCreating combined heart model...")

combined_mask = (
    prediction_resized != 0
)

if combined_mask.sum() > 0:

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
        OUTPUT_DIR,
        "pat15_heart_combined.stl"
    )

    combined_ply = os.path.join(
        OUTPUT_DIR,
        "pat15_heart_combined.ply"
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


print(
    "\n===================================="
)

print(
    "3D reconstruction completed!"
)

print(
    "===================================="
)