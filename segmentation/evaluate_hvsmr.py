import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F


# -------------------------------------------------
# Paths
# -------------------------------------------------

PREDICTION_PATH = "outputs/pat15_prediction.npy"

GROUND_TRUTH_PATH = (
    "../dataset/HVSMR/cropped/cropped/"
    "pat15_cropped_seg.nii.gz"
)


# -------------------------------------------------
# Load prediction
# -------------------------------------------------

prediction = np.load(
    PREDICTION_PATH
)

print("Prediction shape :", prediction.shape)


# -------------------------------------------------
# Load original ground truth
# -------------------------------------------------

ground_truth = nib.load(
    GROUND_TRUTH_PATH
).get_fdata().astype(np.int64)

print(
    "Original ground truth shape:",
    ground_truth.shape
)


# -------------------------------------------------
# Resize ground truth exactly like training
# -------------------------------------------------

resized_slices = []

for s in range(ground_truth.shape[2]):

    mask = ground_truth[:, :, s]

    mask = torch.from_numpy(
        mask
    ).float().unsqueeze(0).unsqueeze(0)

    mask = F.interpolate(
        mask,
        size=(256, 256),
        mode="nearest"
    )

    mask = (
        mask
        .squeeze(0)
        .squeeze(0)
        .numpy()
        .astype(np.int64)
    )

    resized_slices.append(mask)


ground_truth = np.stack(
    resized_slices,
    axis=2
)

print(
    "Resized ground truth shape:",
    ground_truth.shape
)


# -------------------------------------------------
# Check shapes
# -------------------------------------------------

if prediction.shape != ground_truth.shape:

    raise ValueError(
        "Prediction and ground truth still have "
        "different shapes."
    )


# -------------------------------------------------
# Dice function
# -------------------------------------------------

def dice_score(pred, target):

    pred = (pred == target)

    return pred.mean()


def class_dice(prediction, ground_truth, class_id):

    pred_class = (
        prediction == class_id
    )

    gt_class = (
        ground_truth == class_id
    )

    intersection = np.logical_and(
        pred_class,
        gt_class
    ).sum()

    denominator = (
        pred_class.sum()
        +
        gt_class.sum()
    )

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection
        / denominator
    )


# -------------------------------------------------
# Classes
# -------------------------------------------------

classes = sorted(
    set(np.unique(prediction))
    |
    set(np.unique(ground_truth))
)

print("\nClasses:")
print(classes)


# -------------------------------------------------
# Per-class Dice
# -------------------------------------------------

dice_scores = []

print("\nDice Scores")
print("========================")

for class_id in classes:

    dice = class_dice(
        prediction,
        ground_truth,
        class_id
    )

    dice_scores.append(dice)

    print(
        f"Class {class_id}: "
        f"{dice:.4f}"
    )


# -------------------------------------------------
# Mean Dice
# -------------------------------------------------

mean_dice = np.mean(
    dice_scores
)

print(
    "\nMean Dice Score:",
    f"{mean_dice:.4f}"
)

print(
    "\nEvaluation completed successfully!"
)