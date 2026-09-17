import os
import sys
import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F

# Allow importing UNet from the same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unet import UNet


# -----------------------------
# Configuration
# -----------------------------
INPUT_NIFTI = "../dataset/HVSMR/cropped/cropped/pat7_cropped.nii.gz"
MODEL_PATH = "../models/hvsmr_unet.pth"
OUTPUT_DIR = "../outputs/hvsmr/pat7"
OUTPUT_NIFTI = os.path.join(
    OUTPUT_DIR,
    "predicted_segmentation.nii.gz"
)

IMAGE_SIZE = 256
NUM_CLASSES = 9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load model
# -----------------------------
def load_model():
    print("Loading U-Net model...")
    print("Device:", DEVICE)

    model = UNet(
        in_channels=1,
        out_channels=NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return model


# -----------------------------
# Normalize MRI volume
# -----------------------------
def normalize_volume(volume):
    volume = volume.astype(np.float32)

    min_value = volume.min()
    max_value = volume.max()

    if max_value > min_value:
        volume = (volume - min_value) / (max_value - min_value)
    else:
        volume = np.zeros_like(volume)

    return volume


# -----------------------------
# Predict entire volume
# -----------------------------
def predict_volume(model, image_path):

    print("\nLoading MRI:")
    print(image_path)

    nii = nib.load(image_path)

    volume = nii.get_fdata()

    print("Original volume shape:", volume.shape)
    print("Voxel spacing:", nii.header.get_zooms()[:3])

    volume = normalize_volume(volume)

    height, width, num_slices = volume.shape

    prediction_volume = np.zeros(
        (height, width, num_slices),
        dtype=np.uint8
    )

    print("\nStarting slice-by-slice prediction...")
    print("Total slices:", num_slices)

    with torch.no_grad():

        for slice_index in range(num_slices):

            # Extract one 2D slice
            slice_2d = volume[:, :, slice_index]

            # Convert to tensor
            image_tensor = torch.from_numpy(
                slice_2d
            ).float()

            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

            # Resize to 256 x 256
            image_tensor = F.interpolate(
                image_tensor,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False
            )

            image_tensor = image_tensor.to(DEVICE)

            # U-Net prediction
            output = model(image_tensor)

            # Get predicted class
            prediction = torch.argmax(
                output,
                dim=1
            )

            # Remove batch dimension
            prediction = prediction.squeeze(0).cpu().numpy()

            # Resize prediction back to original slice size
            prediction_tensor = torch.from_numpy(
                prediction
            ).float().unsqueeze(0).unsqueeze(0)

            prediction_tensor = F.interpolate(
                prediction_tensor,
                size=(height, width),
                mode="nearest"
            )

            prediction = prediction_tensor.squeeze().numpy()

            prediction_volume[:, :, slice_index] = prediction.astype(
                np.uint8
            )

            # Progress
            if (
                slice_index == 0
                or (slice_index + 1) % 10 == 0
                or slice_index == num_slices - 1
            ):
                print(
                    f"Processed slice "
                    f"{slice_index + 1}/{num_slices}"
                )

    return prediction_volume, nii


# -----------------------------
# Save prediction
# -----------------------------
def save_prediction(prediction_volume, reference_nii):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    prediction_nii = nib.Nifti1Image(
        prediction_volume,
        reference_nii.affine,
        reference_nii.header
    )

    nib.save(
        prediction_nii,
        OUTPUT_NIFTI
    )

    print("\nPrediction saved:")
    print(OUTPUT_NIFTI)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("HVSMR 2D U-NET VOLUME PREDICTION")
    print("=" * 60)

    model = load_model()

    prediction_volume, reference_nii = predict_volume(
        model,
        INPUT_NIFTI
    )

    save_prediction(
        prediction_volume,
        reference_nii
    )

    # Show predicted labels
    unique_labels = np.unique(
        prediction_volume
    )

    print("\nPredicted labels:")
    print(unique_labels)

    print("\nPrediction complete.")
    print("=" * 60)