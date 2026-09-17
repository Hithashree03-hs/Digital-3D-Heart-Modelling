import os
import nibabel as nib
import numpy as np

# ============================================================
# HVSMR DATASET INSPECTION
# ============================================================

# Patient to inspect
patient = "pat0"

# Actual location of the HVSMR files
DATASET_DIR = os.path.join(
    "dataset",
    "HVSMR",
    "cropped",
    "cropped"
)

# File paths
image_path = os.path.join(
    DATASET_DIR,
    f"{patient}_cropped.nii.gz"
)

seg_path = os.path.join(
    DATASET_DIR,
    f"{patient}_cropped_seg.nii.gz"
)

# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("HVSMR DATASET INSPECTION")
print("=" * 70)

print("\nPatient:", patient)

print("\nMRI file:")
print(image_path)

print("\nSegmentation file:")
print(seg_path)

if not os.path.exists(image_path):
    print("\nERROR: MRI file was not found.")
    print("Check the dataset folder location.")
    print("\nExpected file:")
    print(os.path.abspath(image_path))
    exit()

if not os.path.exists(seg_path):
    print("\nERROR: Segmentation file was not found.")
    print("Check the dataset folder location.")
    print("\nExpected file:")
    print(os.path.abspath(seg_path))
    exit()

print("\nFiles found successfully!")

# ============================================================
# LOAD MRI
# ============================================================

print("\n" + "=" * 70)
print("LOADING MRI")
print("=" * 70)

image = nib.load(image_path)

image_data = image.get_fdata()

# ============================================================
# LOAD SEGMENTATION
# ============================================================

print("\n" + "=" * 70)
print("LOADING SEGMENTATION")
print("=" * 70)

seg = nib.load(seg_path)

seg_data = seg.get_fdata()

# ============================================================
# MRI INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MRI INFORMATION")
print("=" * 70)

print("Shape:")
print(image_data.shape)

print("\nVoxel spacing:")
print(image.header.get_zooms())

print("\nData type:")
print(image_data.dtype)

print("\nMinimum intensity:")
print(image_data.min())

print("\nMaximum intensity:")
print(image_data.max())

print("\nMean intensity:")
print(image_data.mean())

print("\nNumber of voxels:")
print(image_data.size)

# ============================================================
# SEGMENTATION INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("SEGMENTATION INFORMATION")
print("=" * 70)

print("Shape:")
print(seg_data.shape)

print("\nVoxel spacing:")
print(seg.header.get_zooms())

# Get unique labels
labels = np.unique(seg_data)

print("\nSegmentation labels:")
print(labels)

# ============================================================
# VOXEL COUNTS
# ============================================================

print("\n" + "=" * 70)
print("SEGMENTATION VOXEL COUNTS")
print("=" * 70)

for label in labels:

    count = np.sum(seg_data == label)

    print(
        f"Label {int(label):2d} : "
        f"{count:,} voxels"
    )

# ============================================================
# PHYSICAL SIZE
# ============================================================

print("\n" + "=" * 70)
print("PHYSICAL VOLUME SIZE")
print("=" * 70)

spacing = image.header.get_zooms()[:3]

shape = image_data.shape[:3]

physical_size = np.array(shape) * np.array(spacing)

print("Physical dimensions:")
print(
    f"X = {physical_size[0]:.2f} mm"
)

print(
    f"Y = {physical_size[1]:.2f} mm"
)

print(
    f"Z = {physical_size[2]:.2f} mm"
)

# ============================================================
# AFFINE MATRIX
# ============================================================

print("\n" + "=" * 70)
print("AFFINE MATRIX")
print("=" * 70)

print(image.affine)

# ============================================================
# ORIENTATION
# ============================================================

print("\n" + "=" * 70)
print("IMAGE ORIENTATION")
print("=" * 70)

orientation = nib.aff2axcodes(image.affine)

print("Orientation:")
print(orientation)

# ============================================================
# SEGMENTATION BOUNDING BOX
# ============================================================

print("\n" + "=" * 70)
print("SEGMENTATION BOUNDING BOX")
print("=" * 70)

# Find all non-background voxels
nonzero = np.argwhere(seg_data > 0)

if len(nonzero) > 0:

    minimum = nonzero.min(axis=0)
    maximum = nonzero.max(axis=0)

    print("Minimum voxel coordinates:")
    print(minimum)

    print("\nMaximum voxel coordinates:")
    print(maximum)

    print("\nBounding box size:")
    print(maximum - minimum + 1)

else:

    print("No segmented structures found.")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("Patient:", patient)
print("MRI shape:", image_data.shape)
print("MRI spacing:", image.header.get_zooms())
print("Segmentation shape:", seg_data.shape)
print("Segmentation spacing:", seg.header.get_zooms())
print("Labels:", labels)

print("\nInspection completed successfully!")

print("=" * 70)