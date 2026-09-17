import os
import glob
import nibabel as nib
import numpy as np


class VolumeDataset:

    def __init__(self, root_dir):

        self.patient_dirs = sorted(
            glob.glob(os.path.join(root_dir, "patient*"))
        )

    def __len__(self):
        return len(self.patient_dirs)

    def get_volume(self, index):

        patient = self.patient_dirs[index]

        image_path = os.path.join(
            patient,
            os.path.basename(patient) + "_sax_ed.nii.gz"
        )

        volume = nib.load(image_path).get_fdata()

        # Normalize
        volume = (volume - volume.min()) / (volume.max() - volume.min())

        return volume