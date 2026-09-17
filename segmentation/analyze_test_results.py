import os
import numpy as np
import pandas as pd


RESULT_PATH = "outputs/test_results/test_set_results.csv"


# ------------------------------------------------------------
# Load results
# ------------------------------------------------------------

df = pd.read_csv(
    RESULT_PATH
)

print("=" * 65)
print("HVSMR TEST RESULT ANALYSIS")
print("=" * 65)

print("\nPatients evaluated:", len(df))


# ------------------------------------------------------------
# Mean Dice including background
# ------------------------------------------------------------

overall_mean = df["mean_dice"].mean()

print(
    "\nMean Dice including background:",
    f"{overall_mean:.4f}"
)


# ------------------------------------------------------------
# Per-class mean Dice
# ------------------------------------------------------------

print("\nPer-class mean Dice")
print("=" * 65)

class_means = {}

for label in range(9):

    column = f"dice_class_{label}"

    mean_value = df[column].mean()

    class_means[label] = mean_value

    print(
        f"Class {label}: {mean_value:.4f}"
    )


# ------------------------------------------------------------
# Foreground Dice
# Exclude background class 0
# ------------------------------------------------------------

foreground_columns = [
    f"dice_class_{label}"
    for label in range(1, 9)
]

foreground_values = df[
    foreground_columns
].values

mean_foreground_dice = (
    foreground_values.mean()
)

print(
    "\nMean Foreground Dice:",
    f"{mean_foreground_dice:.4f}"
)


# ------------------------------------------------------------
# Structure names
# ------------------------------------------------------------

names = {
    1: "Left Ventricle (LV)",
    2: "Right Ventricle (RV)",
    3: "Left Atrium (LA)",
    4: "Right Atrium (RA)",
    5: "Aorta (AO)",
    6: "Pulmonary Artery (PA)",
    7: "Superior Vena Cava (SVC)",
    8: "Inferior Vena Cava (IVC)"
}


print("\nCardiac Structure Performance")
print("=" * 65)

for label in range(1, 9):

    print(
        f"{names[label]:30s} "
        f"{class_means[label]:.4f}"
    )


# ------------------------------------------------------------
# Best and worst test patients
# ------------------------------------------------------------

best_patient = df.loc[
    df["mean_dice"].idxmax()
]

worst_patient = df.loc[
    df["mean_dice"].idxmin()
]

print("\nBest test patient:")
print(
    best_patient["patient"],
    f"({best_patient['mean_dice']:.4f})"
)

print("\nLowest test patient:")
print(
    worst_patient["patient"],
    f"({worst_patient['mean_dice']:.4f})"
)


# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------

os.makedirs(
    "outputs/test_results",
    exist_ok=True
)

summary_path = (
    "outputs/test_results/"
    "test_metrics_summary.csv"
)

summary = {
    "overall_mean_dice": overall_mean,
    "mean_foreground_dice": mean_foreground_dice
}

for label in range(1, 9):

    summary[
        f"class_{label}_mean_dice"
    ] = class_means[label]


pd.DataFrame(
    [summary]
).to_csv(
    summary_path,
    index=False
)


print(
    "\nSummary saved to:",
    summary_path
)

print(
    "\nAnalysis completed successfully!"
)