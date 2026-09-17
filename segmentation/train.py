import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import HVSMR2DDataset
from unet import UNet


# =====================================================
# Configuration
# =====================================================

BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 0.0001

NUM_CLASSES = 9

# Paths are relative to the segmentation folder
DATASET_PATH = "../dataset/HVSMR/cropped/cropped"
MODEL_PATH = "../models/hvsmr_unet.pth"


# =====================================================
# Dice Loss
# =====================================================

class DiceLoss(nn.Module):

    def __init__(self, num_classes=9):

        super().__init__()

        self.num_classes = num_classes

    def forward(self, outputs, targets):

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        targets_one_hot = F.one_hot(
            targets,
            num_classes=self.num_classes
        )

        targets_one_hot = targets_one_hot.permute(
            0,
            3,
            1,
            2
        ).float()

        smooth = 1e-6

        intersection = (
            probabilities * targets_one_hot
        ).sum(dim=(0, 2, 3))

        denominator = (
            probabilities.sum(dim=(0, 2, 3))
            +
            targets_one_hot.sum(dim=(0, 2, 3))
        )

        dice = (
            (2.0 * intersection + smooth)
            /
            (denominator + smooth)
        )

        return 1.0 - dice.mean()


# =====================================================
# Combined Loss
# =====================================================

class CombinedLoss(nn.Module):

    def __init__(self, num_classes=9):

        super().__init__()

        self.cross_entropy = nn.CrossEntropyLoss()

        self.dice = DiceLoss(
            num_classes=num_classes
        )

    def forward(self, outputs, targets):

        ce_loss = self.cross_entropy(
            outputs,
            targets
        )

        dice_loss = self.dice(
            outputs,
            targets
        )

        total_loss = (
            ce_loss + dice_loss
        ) / 2.0

        return total_loss


# =====================================================
# Training
# =====================================================

def train():

    print("\n====================================")
    print("Loading HVSMR Dataset")
    print("====================================\n")

    # -------------------------------------------------
    # Check dataset path
    # -------------------------------------------------

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            f"\nHVSMR dataset not found:\n"
            f"{os.path.abspath(DATASET_PATH)}\n"
        )

    print(
        "Dataset path:",
        os.path.abspath(DATASET_PATH)
    )

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    train_dataset = HVSMR2DDataset(
        DATASET_PATH
    )

    if len(train_dataset) == 0:

        raise RuntimeError(
            "No HVSMR training slices were found."
        )

    # -------------------------------------------------
    # DataLoader
    # -------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nTraining Configuration")
    print("======================")

    print(
        "Device:",
        device
    )

    print(
        "Training slices:",
        len(train_dataset)
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Total batches per epoch:",
        len(train_loader)
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    print(
        "Number of classes:",
        NUM_CLASSES
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = UNet(
        in_channels=1,
        out_channels=NUM_CLASSES
    ).to(device)

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------

    criterion = CombinedLoss(
        num_classes=NUM_CLASSES
    )

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -------------------------------------------------
    # Create model directory
    # -------------------------------------------------

    os.makedirs(
        "../models",
        exist_ok=True
    )

    best_loss = float("inf")

    # =================================================
    # Start Training
    # =================================================

    print("\n====================================")
    print("HVSMR 2D U-Net Training Started")
    print("====================================")

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        print(
            f"\n========== Epoch "
            f"{epoch + 1}/{EPOCHS} =========="
        )

        # -------------------------------------------------
        # Batches
        # -------------------------------------------------

        for batch_idx, (images, masks) in enumerate(
            train_loader
        ):

            images = images.to(device)

            masks = masks.long().to(device)

            # -------------------------------------------------
            # Clear gradients
            # -------------------------------------------------

            optimizer.zero_grad()

            # -------------------------------------------------
            # Forward pass
            # -------------------------------------------------

            outputs = model(
                images
            )

            # -------------------------------------------------
            # Calculate loss
            # -------------------------------------------------

            loss = criterion(
                outputs,
                masks
            )

            # -------------------------------------------------
            # Backpropagation
            # -------------------------------------------------

            loss.backward()

            # -------------------------------------------------
            # Update model
            # -------------------------------------------------

            optimizer.step()

            # -------------------------------------------------
            # Track loss
            # -------------------------------------------------

            running_loss += loss.item()

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if (
                (batch_idx + 1) % 20 == 0
                or
                (batch_idx + 1) == len(train_loader)
            ):

                print(
                    f"Batch "
                    f"{batch_idx + 1}/"
                    f"{len(train_loader)} "
                    f"| Loss: "
                    f"{loss.item():.4f}"
                )

        # -------------------------------------------------
        # Average epoch loss
        # -------------------------------------------------

        average_loss = (
            running_loss
            /
            len(train_loader)
        )

        print(
            f"\nEpoch {epoch + 1} "
            f"Average Loss: "
            f"{average_loss:.4f}"
        )

        # -------------------------------------------------
        # Save best model
        # -------------------------------------------------

        if average_loss < best_loss:

            best_loss = average_loss

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "\nBest HVSMR model saved:"
            )

            print(
                os.path.abspath(MODEL_PATH)
            )

    # =================================================
    # Training Complete
    # =================================================

    print("\n====================================")
    print("Training Completed Successfully!")
    print("====================================")

    print(
        "Best Loss:",
        round(best_loss, 4)
    )

    print(
        "Model:",
        os.path.abspath(MODEL_PATH)
    )


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    train()