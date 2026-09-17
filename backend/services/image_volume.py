import os
import re
import tempfile

import numpy as np
import nibabel as nib
from PIL import Image


# ============================================================
# SUPPORTED IMAGE FORMATS
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg"
}


# ============================================================
# NATURAL SORT
# ============================================================

def natural_sort_key(filename):
    """
    Sort names such as:

        slice1.png
        slice2.png
        slice10.png

    correctly.
    """

    name = os.path.basename(filename).lower()

    return [
        int(text) if text.isdigit() else text
        for text in re.split(
            r"(\d+)",
            name
        )
    ]


# ============================================================
# VALIDATE IMAGE FILES
# ============================================================

def validate_image_files(image_paths):

    if not image_paths:
        raise ValueError(
            "No image slices were supplied."
        )

    for path in image_paths:

        extension = os.path.splitext(
            path
        )[1].lower()

        if extension not in SUPPORTED_EXTENSIONS:

            raise ValueError(
                f"Unsupported image format: "
                f"{path}"
            )


# ============================================================
# LOAD IMAGE SLICES
# ============================================================

def load_image_slices(image_paths):

    validate_image_files(
        image_paths
    )

    sorted_paths = sorted(
        image_paths,
        key=natural_sort_key
    )

    slices = []

    reference_shape = None

    for path in sorted_paths:

        try:

            image = Image.open(
                path
            ).convert("L")

            array = np.asarray(
                image,
                dtype=np.float32
            )

        except Exception as exc:

            raise ValueError(
                f"Could not read image "
                f"{os.path.basename(path)}: "
                f"{exc}"
            )

        if reference_shape is None:

            reference_shape = array.shape

        elif array.shape != reference_shape:

            raise ValueError(
                "All MRI slices must have "
                "the same image dimensions."
            )

        slices.append(
            array
        )

    if not slices:

        raise ValueError(
            "No valid MRI slices found."
        )

    volume = np.stack(
        slices,
        axis=2
    )

    return (
        volume,
        sorted_paths
    )


# ============================================================
# CREATE NIFTI VOLUME
# ============================================================

def create_nifti_from_images(
    image_paths,
    pixel_spacing_x,
    pixel_spacing_y,
    slice_spacing,
    output_path
):
    """
    Convert multiple 2D MRI image slices into
    a 3D NIfTI volume.

    Parameters
    ----------
    image_paths : list[str]
        Paths to PNG/JPG/JPEG slices.

    pixel_spacing_x : float
        Pixel spacing in mm along X.

    pixel_spacing_y : float
        Pixel spacing in mm along Y.

    slice_spacing : float
        Distance between slices in mm.

    output_path : str
        Output .nii.gz path.
    """

    # --------------------------------------------------------
    # Validate spacing
    # --------------------------------------------------------

    try:

        pixel_spacing_x = float(
            pixel_spacing_x
        )

        pixel_spacing_y = float(
            pixel_spacing_y
        )

        slice_spacing = float(
            slice_spacing
        )

    except (
        ValueError,
        TypeError
    ):

        raise ValueError(
            "Spacing values must be numeric."
        )

    if (
        pixel_spacing_x <= 0
        or pixel_spacing_y <= 0
        or slice_spacing <= 0
    ):

        raise ValueError(
            "Spacing values must be greater than zero."
        )

    # --------------------------------------------------------
    # Load slices
    # --------------------------------------------------------

    volume, sorted_paths = (
        load_image_slices(
            image_paths
        )
    )

    # --------------------------------------------------------
    # Create affine
    # --------------------------------------------------------

    affine = np.eye(
        4,
        dtype=np.float32
    )

    affine[0, 0] = pixel_spacing_x
    affine[1, 1] = pixel_spacing_y
    affine[2, 2] = slice_spacing

    # --------------------------------------------------------
    # Create NIfTI
    # --------------------------------------------------------

    nifti_image = nib.Nifti1Image(
        volume,
        affine
    )

    nifti_image.header.set_zooms(
        (
            pixel_spacing_x,
            pixel_spacing_y,
            slice_spacing
        )
    )

    # --------------------------------------------------------
    # Ensure output folder exists
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    nib.save(
        nifti_image,
        output_path
    )

    return {
        "nifti_path": output_path,
        "shape": list(volume.shape),
        "spacing_mm": [
            pixel_spacing_x,
            pixel_spacing_y,
            slice_spacing
        ],
        "slice_count": len(sorted_paths),
        "ordered_slices": [
            os.path.basename(path)
            for path in sorted_paths
        ]
    }