import os
import nibabel as nib
import matplotlib.pyplot as plt

# MRI image path
image_path = os.path.join(
    "dataset",
    "ACDC",
    "train",
    "patient003",
    "patient003_sax_ed.nii.gz"
)

# Ground truth mask path
mask_path = os.path.join(
    "dataset",
    "ACDC",
    "train",
    "patient003",
    "patient003_sax_ed_gt.nii.gz"
)

# Load MRI and mask
image = nib.load(image_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

print("MRI Shape :", image.shape)
print("Mask Shape:", mask.shape)

middle = image.shape[2] // 2

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image[:, :, middle], cmap="gray")
plt.title("MRI")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask[:, :, middle], cmap="jet")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.tight_layout()
plt.show()