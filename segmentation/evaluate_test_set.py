import os
import csv
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from unet import UNet


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "../dataset/HVSMR/cropped/cropped"

MODEL_PATH = "../models/hvsmr_unet_split.pth"

OUTPUT_DIR = "outputs/test_results"

IMAGE_SIZE = (256, 256)

NUM_CLASSES = 9


TEST_PATIENTS = [
    "pat56",
    "pat8",
    "pat14",
    "pat15",
    "pat17",
    "pat47",
    "pat1",
    "pat7",
    "pat40"
]


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# Load model
# ============================================================

model = UNet(
    in_channels=1,
    out_channels=NUM_CLASSES
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print(
    "Model loaded:",
    MODEL_PATH
)


# ============================================================
# Dice function
# ============================================================

def dice_score(
    prediction,
    ground_truth,
    label
):

    pred = (
        prediction == label
    )

    gt = (
        ground_truth == label
    )

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    denominator = (
        pred.sum()
        +
        gt.sum()
    )

    if denominator == 0:

        return 1.0

    return (
        2.0
        * intersection
        / denominator
    )


# ============================================================
# Predict one patient
# ============================================================

def predict_patient(patient_id):

    image_path = os.path.join(
        DATASET_PATH,
        f"{patient_id}_cropped.nii.gz"
    )

    mask_path = os.path.join(
        DATASET_PATH,
        f"{patient_id}_cropped_seg.nii.gz"
    )

    print("\n" + "=" * 60)

    print(
        "Patient:",
        patient_id
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Load volumes
    # --------------------------------------------------------

    image_nii = nib.load(
        image_path
    )

    mask_nii = nib.load(
        mask_path
    )

    image_volume = image_nii.get_fdata().astype(
        np.float32
    )

    ground_truth = mask_nii.get_fdata().astype(
        np.int64
    )

    print(
        "Original shape:",
        image_volume.shape
    )

    num_slices = image_volume.shape[2]

    predictions = []

    # --------------------------------------------------------
    # Slice-by-slice prediction
    # --------------------------------------------------------

    with torch.no_grad():

        for s in range(num_slices):

            image = image_volume[:, :, s]

            # Normalize
            image_min = image.min()
            image_max = image.max()

            image = (
                image - image_min
            ) / (
                image_max
                - image_min
                + 1e-8
            )

            # Tensor
            image = torch.from_numpy(
                image
            ).unsqueeze(0).unsqueeze(0)

            # Resize exactly as training
            image = F.interpolate(
                image,
                size=IMAGE_SIZE,
                mode="bilinear",
                align_corners=False
            )

            image = image.to(device)

            # Prediction
            output = model(
                image
            )

            prediction = torch.argmax(
                output,
                dim=1
            )

            prediction = (
                prediction
                .squeeze()
                .cpu()
                .numpy()
            )

            predictions.append(
                prediction
            )

    prediction_volume = np.stack(
        predictions,
        axis=2
    )

    print(
        "Prediction shape:",
        prediction_volume.shape
    )

    # --------------------------------------------------------
    # Resize ground truth to 256x256
    # --------------------------------------------------------

    resized_gt = []

    for s in range(
        ground_truth.shape[2]
    ):

        mask = torch.from_numpy(
            ground_truth[:, :, s]
            .astype(np.float32)
        ).unsqueeze(0).unsqueeze(0)

        mask = F.interpolate(
            mask,
            size=IMAGE_SIZE,
            mode="nearest"
        )

        mask = (
            mask
            .squeeze()
            .numpy()
            .astype(np.int64)
        )

        resized_gt.append(mask)

    ground_truth_resized = np.stack(
        resized_gt,
        axis=2
    )

    print(
        "Ground truth shape:",
        ground_truth_resized.shape
    )

    # --------------------------------------------------------
    # Calculate Dice
    # --------------------------------------------------------

    scores = {}

    for label in range(
        NUM_CLASSES
    ):

        scores[label] = dice_score(
            prediction_volume,
            ground_truth_resized,
            label
        )

    # Mean Dice
    mean_dice = np.mean(
        list(scores.values())
    )

    print("\nDice Scores")

    for label, score in scores.items():

        print(
            f"Class {label}: {score:.4f}"
        )

    print(
        "\nMean Dice:",
        f"{mean_dice:.4f}"
    )

    # --------------------------------------------------------
    # Save prediction
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    prediction_path = os.path.join(
        OUTPUT_DIR,
        f"{patient_id}_prediction.npy"
    )

    np.save(
        prediction_path,
        prediction_volume
    )

    return scores, mean_dice


# ============================================================
# Evaluate all test patients
# ============================================================

all_results = []

print("\n")
print("=" * 60)
print("HVSMR TEST SET EVALUATION")
print("=" * 60)

for patient in TEST_PATIENTS:

    scores, mean_dice = predict_patient(
        patient
    )

    all_results.append({
        "patient": patient,
        "mean_dice": mean_dice,
        **{
            f"dice_class_{label}":
            scores[label]
            for label in scores
        }
    })


# ============================================================
# Overall test performance
# ============================================================

mean_test_dice = np.mean([
    result["mean_dice"]
    for result in all_results
])


print("\n")
print("=" * 60)
print("FINAL TEST SET RESULT")
print("=" * 60)

for result in all_results:

    print(
        f"{result['patient']}: "
        f"{result['mean_dice']:.4f}"
    )

print(
    "\nOverall Mean Test Dice:",
    f"{mean_test_dice:.4f}"
)


# ============================================================
# Save CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "test_set_results.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=all_results[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        all_results
    )


print(
    "\nResults saved to:",
    csv_path
)

print(
    "\nTest evaluation completed successfully!"
)