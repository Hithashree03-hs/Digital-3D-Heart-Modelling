import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "../dataset/HVSMR/cropped/cropped/pat7_cropped.nii.gz"
PRED_PATH = "../outputs/hvsmr/pat7/predicted_segmentation.nii.gz"

image = nib.load(IMAGE_PATH).get_fdata()
prediction = nib.load(PRED_PATH).get_fdata()

print("MRI shape:", image.shape)
print("Prediction shape:", prediction.shape)

# Find the slice containing the most predicted cardiac pixels
scores = []

for z in range(prediction.shape[2]):
    scores.append(np.sum(prediction[:, :, z] > 0))

best_slice = int(np.argmax(scores))

print("Best slice:", best_slice)
print("Predicted cardiac pixels:", scores[best_slice])

# Display
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image[:, :, best_slice], cmap="gray")
plt.title(f"MRI Slice {best_slice}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image[:, :, best_slice], cmap="gray")
plt.imshow(
    prediction[:, :, best_slice],
    alpha=0.5,
    interpolation="nearest"
)
plt.title("U-Net Prediction")
plt.axis("off")

plt.tight_layout()
plt.show()