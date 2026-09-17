import os
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# MAKE backend able to import the copied U-Net architecture
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from model.unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    BACKEND_DIR,
    "model",
    "hvsmr_unet_split.pth"
)

IMAGE_SIZE = (256, 256)
NUM_CLASSES = 9

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained HVSMR U-Net.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = UNet(
        in_channels=1,
        out_channels=NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Support common PyTorch checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Remove possible "module." prefix from DataParallel
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# NORMALIZE SLICE
# ============================================================

def normalize_slice(image):
    """
    Same min-max normalization used during training.
    """

    image = image.astype(np.float32)

    image_min = image.min()
    image_max = image.max()

    image = (
        image - image_min
    ) / (
        image_max - image_min + 1e-8
    )

    return image


# ============================================================
# PREPARE SLICE
# ============================================================

def prepare_slice(image):
    """
    Normalize and resize an MRI slice exactly according
    to the training pipeline.
    """

    image = normalize_slice(image)

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0)

    tensor = F.interpolate(
        tensor,
        size=IMAGE_SIZE,
        mode="bilinear",
        align_corners=False
    )

    return tensor


# ============================================================
# PREDICT ONE SLICE
# ============================================================

def predict_slice(model, image):
    """
    Segment one MRI slice.
    """

    input_tensor = prepare_slice(image)

    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():

        output = model(
            input_tensor
        )

        prediction = torch.argmax(
            output,
            dim=1
        )

    prediction = (
        prediction
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.int16)
    )

    return prediction


# ============================================================
# PREDICT FULL MRI VOLUME
# ============================================================

def predict_volume(
    input_mri_path,
    output_prediction_path=None
):
    """
    Segment a complete MRI volume slice-by-slice.

    Input:
        .nii or .nii.gz MRI volume

    Output:
        Numpy segmentation volume with the
        same X,Y,Z dimensions as the input MRI.
    """

    if not os.path.exists(input_mri_path):

        raise FileNotFoundError(
            f"MRI file not found: {input_mri_path}"
        )

    # --------------------------------------------------------
    # Load MRI
    # --------------------------------------------------------

    mri = nib.load(
        input_mri_path
    )

    image_volume = np.asarray(
        mri.dataobj,
        dtype=np.float32
    )

    original_shape = image_volume.shape

    if len(original_shape) != 3:

        raise ValueError(
            "Input MRI must be a 3D volume."
        )

    height, width, num_slices = original_shape

    print(
        f"Input MRI shape: {original_shape}"
    )

    # --------------------------------------------------------
    # Load model once
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Allocate output
    # --------------------------------------------------------

    prediction_volume = np.zeros(
        original_shape,
        dtype=np.int16
    )

    # --------------------------------------------------------
    # Process every axial slice
    # --------------------------------------------------------

    for slice_index in range(num_slices):

        if slice_index % 10 == 0:

            print(
                f"Processing slice "
                f"{slice_index + 1}/{num_slices}"
            )

        image = image_volume[
            :, :,
            slice_index
        ]

        # --------------------------------------------
        # Predict at 256x256
        # --------------------------------------------

        prediction_256 = predict_slice(
            model,
            image
        )

        # --------------------------------------------
        # Restore prediction to original slice size
        # --------------------------------------------

        prediction_tensor = torch.from_numpy(
            prediction_256.astype(np.float32)
        ).unsqueeze(0).unsqueeze(0)

        prediction_original = F.interpolate(
            prediction_tensor,
            size=(height, width),
            mode="nearest"
        )

        prediction_original = (
            prediction_original
            .squeeze()
            .numpy()
            .astype(np.int16)
        )

        prediction_volume[
            :, :,
            slice_index
        ] = prediction_original

    # --------------------------------------------------------
    # Save .npy prediction
    # --------------------------------------------------------

    if output_prediction_path is not None:

        output_dir = os.path.dirname(
            output_prediction_path
        )

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True
            )

        np.save(
            output_prediction_path,
            prediction_volume
        )

        print(
            f"Prediction saved to: "
            f"{output_prediction_path}"
        )

    return prediction_volume, mri