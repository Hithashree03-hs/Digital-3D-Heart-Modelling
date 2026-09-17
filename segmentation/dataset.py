import os
import glob

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


class HVSMR2DDataset(Dataset):

    def __init__(
        self,
        root_dir,
        image_size=(256, 256)
    ):

        self.root_dir = root_dir
        self.image_size = image_size

        self.samples = []

        # -------------------------------------------------
        # Find MRI volumes
        # -------------------------------------------------

        image_files = sorted(
            glob.glob(
                os.path.join(
                    root_dir,
                    "*_cropped.nii.gz"
                )
            )
        )

        print("Loading HVSMR volumes...")

        if len(image_files) == 0:

            raise FileNotFoundError(
                f"No HVSMR MRI files found in: {root_dir}"
            )

        # -------------------------------------------------
        # Register patients and slices
        # -------------------------------------------------

        for image_path in image_files:

            filename = os.path.basename(
                image_path
            )

            # Ignore segmentation files
            if "_cropped_seg" in filename:
                continue

            base_name = filename.replace(
                "_cropped.nii.gz",
                ""
            )

            mask_path = os.path.join(
                root_dir,
                f"{base_name}_cropped_seg.nii.gz"
            )

            if not os.path.exists(mask_path):

                print(
                    f"Segmentation not found: "
                    f"{base_name}"
                )

                continue

            # Read header only to determine
            # number of slices.
            image_nii = nib.load(
                image_path
            )

            mask_nii = nib.load(
                mask_path
            )

            image_shape = image_nii.shape
            mask_shape = mask_nii.shape

            if image_shape != mask_shape:

                print(
                    f"Shape mismatch: {base_name}"
                )

                print(
                    "MRI:",
                    image_shape
                )

                print(
                    "Mask:",
                    mask_shape
                )

                continue

            print(
                f"Loading: {base_name}"
            )

            num_slices = image_shape[2]

            # -------------------------------------------------
            # Store references instead of loading everything
            # -------------------------------------------------

            for slice_index in range(num_slices):

                self.samples.append(
                    (
                        image_path,
                        mask_path,
                        slice_index
                    )
                )

        # -------------------------------------------------
        # Dataset information
        # -------------------------------------------------

        print(
            f"\nRegistered {len(self.samples)} "
            f"HVSMR 2D slices successfully."
        )

        if len(self.samples) == 0:

            raise RuntimeError(
                "No valid HVSMR slices were found."
            )

    # =====================================================
    # Dataset Length
    # =====================================================

    def __len__(self):

        return len(self.samples)

    # =====================================================
    # Get Sample
    # =====================================================

    def __getitem__(self, idx):

        image_path, mask_path, slice_index = (
            self.samples[idx]
        )

        # -------------------------------------------------
        # Load only the required slice
        # -------------------------------------------------

        image_nii = nib.load(
            image_path
        )

        mask_nii = nib.load(
            mask_path
        )

        image_volume = image_nii.dataobj
        mask_volume = mask_nii.dataobj

        image = np.asarray(
            image_volume[:, :, slice_index],
            dtype=np.float32
        )

        mask = np.asarray(
            mask_volume[:, :, slice_index],
            dtype=np.int64
        )

        # -------------------------------------------------
        # Normalize MRI slice
        # -------------------------------------------------

        image_min = image.min()
        image_max = image.max()

        image = (
            image - image_min
        ) / (
            image_max - image_min + 1e-8
        )

        # -------------------------------------------------
        # Convert to tensors
        # -------------------------------------------------

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        mask = torch.from_numpy(
            mask
        )

        # -------------------------------------------------
        # Resize MRI
        # -------------------------------------------------

        image = F.interpolate(
            image.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False
        )

        # -------------------------------------------------
        # Resize segmentation
        # -------------------------------------------------

        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=self.image_size,
            mode="nearest"
        )

        # -------------------------------------------------
        # Remove extra dimensions
        # -------------------------------------------------

        image = image.squeeze(0)

        mask = (
            mask
            .squeeze(0)
            .squeeze(0)
            .long()
        )

        return image, mask