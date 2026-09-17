import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from segmentation.unet import UNet
from segmentation.volume_dataset import VolumeDataset


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load Trained Model
# -----------------------------
model = UNet(in_channels=1, out_channels=4)

model.load_state_dict(
    torch.load("models/unet_model.pth", map_location=device)
)

model.to(device)
model.eval()

print("Model Loaded Successfully!")


# -----------------------------
# Load One Patient Volume
# -----------------------------
dataset = VolumeDataset("dataset/ACDC/train")

print("Total Patients:", len(dataset))

patient_index = 0      # Change this to test different patients

volume = dataset.get_volume(patient_index)

print("Original Volume Shape:", volume.shape)


# -----------------------------
# Predict Every Slice
# -----------------------------
predicted_volume = []

for i in range(volume.shape[2]):

    slice_img = volume[:, :, i]

    slice_img = np.expand_dims(slice_img, axis=0)
    slice_img = np.expand_dims(slice_img, axis=0)

    slice_img = torch.tensor(
        slice_img,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        output = model(slice_img)

    prediction = torch.argmax(output, dim=1)

    predicted_volume.append(
        prediction.squeeze().cpu().numpy()
    )

    print(f"Processed Slice {i+1}/{volume.shape[2]}")


# -----------------------------
# Stack into 3D Volume
# -----------------------------
predicted_volume = np.stack(predicted_volume, axis=2)

print("\nPredicted Volume Shape:", predicted_volume.shape)


# -----------------------------
# Save Volume
# -----------------------------
os.makedirs("outputs", exist_ok=True)

np.save(
    "outputs/patient_001_volume.npy",
    predicted_volume
)

print("\n3D Volume Saved Successfully!")
