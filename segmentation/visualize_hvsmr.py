import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# -------------------------------------------------
# Paths
# -------------------------------------------------

IMAGE_PATH = (
    "../dataset/HVSMR/cropped/cropped/"
    "pat15_cropped.nii.gz"
)

GT_PATH = (
    "../dataset/HVSMR/cropped/cropped/"
    "pat15_cropped_seg.nii.gz"
)

PRED_PATH = "outputs/pat15_prediction.npy"


# -------------------------------------------------
# Load
# -------------------------------------------------

image = nib.load(
    IMAGE_PATH
).get_fdata()

ground_truth = nib.load(
    GT_PATH
).get_fdata()

prediction = np.load(
    PRED_PATH
)


# -------------------------------------------------
# Choose a slice containing segmentation
# -------------------------------------------------

# Find slice with largest predicted foreground area
foreground_sizes = []

for z in range(prediction.shape[2]):

    foreground = (
        prediction[:, :, z] != 0
    )

    foreground_sizes.append(
        foreground.sum()
    )

slice_index = int(
    np.argmax(foreground_sizes)
)

print(
    "Selected slice:",
    slice_index
)


# -------------------------------------------------
# Resize original MRI to prediction size
# -------------------------------------------------

from PIL import Image

mri_slice = image[:, :, slice_index]

mri_slice = (
    mri_slice - mri_slice.min()
) / (
    mri_slice.max()
    - mri_slice.min()
    + 1e-8
)

mri_slice = np.array(
    Image.fromarray(
        (mri_slice * 255).astype(np.uint8)
    ).resize(
        (256, 256)
    )
) / 255.0


# -------------------------------------------------
# Resize ground truth to 256x256
# -------------------------------------------------

import torch
import torch.nn.functional as F

gt_slice = torch.from_numpy(
    ground_truth[:, :, slice_index]
).float().unsqueeze(0).unsqueeze(0)

gt_slice = F.interpolate(
    gt_slice,
    size=(256, 256),
    mode="nearest"
)

gt_slice = (
    gt_slice
    .squeeze()
    .numpy()
)


# -------------------------------------------------
# Plot
# -------------------------------------------------

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.imshow(
    mri_slice,
    cmap="gray"
)
plt.title("MRI")
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(
    gt_slice
)
plt.title("Ground Truth")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(
    prediction[:, :, slice_index]
)
plt.title("Prediction")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "outputs/pat15_comparison.png",
    dpi=200
)

plt.show()