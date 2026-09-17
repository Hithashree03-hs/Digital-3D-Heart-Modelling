import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2

print("Step 1")

image_path = os.path.join(
    "dataset",
    "ACDC",
    "train",
    "patient003",
    "patient003_sax_ed.nii.gz"
)

print("Step 2")

image = nib.load(image_path).get_fdata()

print("Step 3")

slice_img = image[:, :, image.shape[2]//2]

print("Step 4")

normalized = cv2.normalize(
    slice_img,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

print("Step 5")

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

enhanced = clahe.apply(normalized)

print("Step 6")

smoothed = cv2.GaussianBlur(
    enhanced,
    (5,5),
    0
)

print("Step 7")

plt.figure(figsize=(15,5))

plt.subplot(141)
plt.imshow(slice_img, cmap="gray")
plt.title("Original")

plt.subplot(142)
plt.imshow(normalized, cmap="gray")
plt.title("Normalized")

plt.subplot(143)
plt.imshow(enhanced, cmap="gray")
plt.title("CLAHE")

plt.subplot(144)
plt.imshow(smoothed, cmap="gray")
plt.title("Gaussian")

print("Step 8")

plt.show()

print("Finished")