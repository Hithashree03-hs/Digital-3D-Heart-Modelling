import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from unet import UNet


# -------------------------------------------------
# Device
# -------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -------------------------------------------------
# Paths
# -------------------------------------------------

IMAGE_PATH = "../dataset/HVSMR/cropped/cropped/pat15_cropped.nii.gz"
MODEL_PATH = "../models/hvsmr_unet.pth"
OUTPUT_PATH = "outputs/pat15_prediction.npy"


# -------------------------------------------------
# Load model
# -------------------------------------------------

model = UNet(
    in_channels=1,
    out_channels=9
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model Loaded Successfully!")


# -------------------------------------------------
# Load MRI volume
# -------------------------------------------------

nii = nib.load(IMAGE_PATH)

volume = nii.get_fdata().astype(np.float32)

print("Original volume shape:", volume.shape)


# -------------------------------------------------
# Predict slice by slice
# -------------------------------------------------

predictions = []

num_slices = volume.shape[2]

print("Total slices:", num_slices)

with torch.no_grad():

    for slice_index in range(num_slices):

        # Get one MRI slice
        image = volume[:, :, slice_index]

        # -----------------------------------------
        # Same normalization as training
        # -----------------------------------------

        image_min = image.min()
        image_max = image.max()

        image = (
            image - image_min
        ) / (
            image_max - image_min + 1e-8
        )

        # -----------------------------------------
        # Convert to tensor
        # -----------------------------------------

        image = torch.from_numpy(
            image
        ).float().unsqueeze(0).unsqueeze(0)

        # -----------------------------------------
        # Same resize as training
        # -----------------------------------------

        image = F.interpolate(
            image,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        image = image.to(device)

        # -----------------------------------------
        # Model prediction
        # -----------------------------------------

        output = model(image)

        prediction = torch.argmax(
            output,
            dim=1
        )

        prediction = (
            prediction
            .squeeze(0)
            .cpu()
            .numpy()
        )

        predictions.append(prediction)

        if slice_index % 10 == 0:
            print(
                f"Processed slice "
                f"{slice_index + 1}/{num_slices}"
            )


# -------------------------------------------------
# Stack slices into 3D volume
# -------------------------------------------------

prediction_volume = np.stack(
    predictions,
    axis=2
)

print(
    "Prediction volume shape:",
    prediction_volume.shape
)

print(
    "Predicted labels:",
    np.unique(prediction_volume)
)


# -------------------------------------------------
# Save prediction
# -------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

np.save(
    OUTPUT_PATH,
    prediction_volume
)

print(
    "Prediction saved to:",
    OUTPUT_PATH
)

print("\nPrediction completed successfully!")