import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset import HVSMR2DDataset
from unet import UNet


# =====================================================
# Configuration
# =====================================================

DATASET_PATH = "../dataset/HVSMR/cropped/cropped"

MODEL_PATH = "../models/hvsmr_unet_split.pth"

BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 0.0001

NUM_CLASSES = 9

SEED = 42


# =====================================================
# Reproducibility
# =====================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =====================================================
# Patient discovery
# =====================================================

def get_patient_ids(root_dir):

    patient_ids = []

    for filename in os.listdir(root_dir):

        if (
            filename.startswith("pat")
            and filename.endswith("_cropped.nii.gz")
            and "_seg" not in filename
        ):

            patient_name = filename.replace(
                "_cropped.nii.gz",
                ""
            )

            patient_ids.append(patient_name)

    def patient_number(name):
        return int(name.replace("pat", ""))

    patient_ids.sort(
        key=patient_number
    )

    return patient_ids


# =====================================================
# Dataset restricted to specific patients
# =====================================================

class HVSMRPatientDataset(Dataset):

    def __init__(
        self,
        root_dir,
        patient_ids,
        image_size=(256, 256)
    ):

        self.root_dir = root_dir
        self.image_size = image_size

        self.samples = []

        print(
            f"\nBuilding dataset for "
            f"{len(patient_ids)} patients..."
        )

        for patient_id in patient_ids:

            image_path = os.path.join(
                root_dir,
                f"{patient_id}_cropped.nii.gz"
            )

            mask_path = os.path.join(
                root_dir,
                f"{patient_id}_cropped_seg.nii.gz"
            )

            if not os.path.exists(image_path):

                print(
                    "Missing MRI:",
                    patient_id
                )

                continue

            if not os.path.exists(mask_path):

                print(
                    "Missing segmentation:",
                    patient_id
                )

                continue

            import nibabel as nib

            image_nii = nib.load(
                image_path
            )

            mask_nii = nib.load(
                mask_path
            )

            if image_nii.shape != mask_nii.shape:

                print(
                    "Shape mismatch:",
                    patient_id
                )

                continue

            for slice_index in range(
                image_nii.shape[2]
            ):

                self.samples.append(
                    (
                        image_path,
                        mask_path,
                        slice_index
                    )
                )

        print(
            "Registered slices:",
            len(self.samples)
        )

    # -------------------------------------------------
    # Length
    # -------------------------------------------------

    def __len__(self):

        return len(self.samples)

    # -------------------------------------------------
    # Get sample
    # -------------------------------------------------

    def __getitem__(self, idx):

        image_path, mask_path, slice_index = (
            self.samples[idx]
        )

        import nibabel as nib

        image_nii = nib.load(
            image_path
        )

        mask_nii = nib.load(
            mask_path
        )

        image = np.asarray(
            image_nii.dataobj[:, :, slice_index],
            dtype=np.float32
        )

        mask = np.asarray(
            mask_nii.dataobj[:, :, slice_index],
            dtype=np.int64
        )

        # ---------------------------------------------
        # Normalize image
        # ---------------------------------------------

        image_min = image.min()
        image_max = image.max()

        image = (
            image - image_min
        ) / (
            image_max - image_min + 1e-8
        )

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        mask = torch.from_numpy(
            mask
        )

        # ---------------------------------------------
        # Resize
        # ---------------------------------------------

        image = F.interpolate(
            image.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False
        )

        mask = F.interpolate(
            mask.unsqueeze(0)
                 .unsqueeze(0)
                 .float(),
            size=self.image_size,
            mode="nearest"
        )

        image = image.squeeze(0)

        mask = (
            mask
            .squeeze(0)
            .squeeze(0)
            .long()
        )

        return image, mask


# =====================================================
# Dice Loss
# =====================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        num_classes=9
    ):

        super().__init__()

        self.num_classes = num_classes

    def forward(
        self,
        outputs,
        targets
    ):

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        targets_one_hot = F.one_hot(
            targets,
            num_classes=self.num_classes
        )

        targets_one_hot = (
            targets_one_hot
            .permute(0, 3, 1, 2)
            .float()
        )

        smooth = 1e-6

        intersection = (
            probabilities
            * targets_one_hot
        ).sum(
            dim=(0, 2, 3)
        )

        denominator = (
            probabilities.sum(
                dim=(0, 2, 3)
            )
            +
            targets_one_hot.sum(
                dim=(0, 2, 3)
            )
        )

        dice = (
            2.0 * intersection
            + smooth
        ) / (
            denominator
            + smooth
        )

        return 1.0 - dice.mean()


# =====================================================
# Combined loss
# =====================================================

class CombinedLoss(nn.Module):

    def __init__(
        self,
        num_classes=9
    ):

        super().__init__()

        self.cross_entropy = (
            nn.CrossEntropyLoss()
        )

        self.dice = DiceLoss(
            num_classes=num_classes
        )

    def forward(
        self,
        outputs,
        targets
    ):

        ce = self.cross_entropy(
            outputs,
            targets
        )

        dice = self.dice(
            outputs,
            targets
        )

        return (
            ce + dice
        ) / 2.0


# =====================================================
# Main training function
# =====================================================

def train():

    print("\n====================================")
    print("HVSMR PATIENT-LEVEL TRAINING")
    print("====================================")

    # -------------------------------------------------
    # Find all patients
    # -------------------------------------------------

    patient_ids = get_patient_ids(
        DATASET_PATH
    )

    print(
        "\nTotal patients:",
        len(patient_ids)
    )

    if len(patient_ids) != 60:

        print(
            "WARNING: Expected 60 patients."
        )

    # -------------------------------------------------
    # Shuffle patients
    # -------------------------------------------------

    rng = random.Random(SEED)

    shuffled = patient_ids.copy()

    rng.shuffle(
        shuffled
    )

    # -------------------------------------------------
    # Patient split
    # -------------------------------------------------

    train_count = int(
        0.70 * len(shuffled)
    )

    val_count = int(
        0.15 * len(shuffled)
    )

    train_patients = shuffled[
        :train_count
    ]

    val_patients = shuffled[
        train_count:
        train_count + val_count
    ]

    test_patients = shuffled[
        train_count + val_count:
    ]

    print(
        "\nTRAIN PATIENTS:",
        len(train_patients)
    )

    print(
        train_patients
    )

    print(
        "\nVALIDATION PATIENTS:",
        len(val_patients)
    )

    print(
        val_patients
    )

    print(
        "\nTEST PATIENTS:",
        len(test_patients)
    )

    print(
        test_patients
    )

    # -------------------------------------------------
    # Datasets
    # -------------------------------------------------

    train_dataset = HVSMRPatientDataset(
        DATASET_PATH,
        train_patients
    )

    val_dataset = HVSMRPatientDataset(
        DATASET_PATH,
        val_patients
    )

    # Test dataset is created later for evaluation.
    # -------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = UNet(
        in_channels=1,
        out_channels=NUM_CLASSES
    ).to(device)

    criterion = CombinedLoss(
        num_classes=NUM_CLASSES
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    os.makedirs(
        "../models",
        exist_ok=True
    )

    best_val_loss = float(
        "inf"
    )

    # =================================================
    # Training
    # =================================================

    for epoch in range(EPOCHS):

        model.train()

        training_loss = 0.0

        for images, masks in train_loader:

            images = images.to(device)

            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                masks
            )

            loss.backward()

            optimizer.step()

            training_loss += loss.item()

        training_loss /= len(
            train_loader
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)

                masks = masks.to(device)

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    masks
                )

                validation_loss += (
                    loss.item()
                )

        validation_loss /= len(
            val_loader
        )

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Training Loss:   "
            f"{training_loss:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        # -------------------------------------------------
        # Save best validation model
        # -------------------------------------------------

        if validation_loss < best_val_loss:

            best_val_loss = (
                validation_loss
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "Best model saved."
            )

    print(
        "\n===================================="
    )

    print(
        "PATIENT-LEVEL TRAINING COMPLETED"
    )

    print(
        "===================================="
    )

    print(
        "Best validation loss:",
        round(
            best_val_loss,
            4
        )
    )

    print(
        "Model:",
        os.path.abspath(
            MODEL_PATH
        )
    )


# =====================================================
# Run
# =====================================================

if __name__ == "__main__":

    train()