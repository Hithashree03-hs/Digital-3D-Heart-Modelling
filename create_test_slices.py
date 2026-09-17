import os
import numpy as np
import nibabel as nib
from PIL import Image

INPUT = "dataset/HVSMR/cropped/cropped/pat17_cropped.nii.gz"
OUTPUT_DIR = "test_image_slices"

os.makedirs(OUTPUT_DIR, exist_ok=True)

nii = nib.load(INPUT)
volume = nii.get_fdata()

print("Volume shape:", volume.shape)

# Create PNG images for every slice
for i in range(volume.shape[2]):

    slice_data = volume[:, :, i]

    # Normalize to 0-255
    minimum = slice_data.min()
    maximum = slice_data.max()

    if maximum > minimum:
        normalized = (
            (slice_data - minimum)
            / (maximum - minimum)
            * 255
        )
    else:
        normalized = np.zeros_like(slice_data)

    image = Image.fromarray(
        normalized.astype(np.uint8)
    )

    image.save(
        os.path.join(
            OUTPUT_DIR,
            f"slice_{i:04d}.png"
        )
    )

print("Finished creating slices.")
print("Saved to:", OUTPUT_DIR)
print("Number of slices:", volume.shape[2])