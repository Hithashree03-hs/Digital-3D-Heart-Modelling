# Digital 3D Heart Analysis Backend

## Overview

This backend processes a 3D cardiac MRI volume and performs:

1. MRI preprocessing
2. Cardiac structure segmentation using a trained U-Net
3. Patient-specific 3D reconstruction
4. Cardiac feature extraction
5. Morphological reference-cohort comparison
6. Patient-wise morphological abnormality screening
7. Generation of STL and PLY 3D models
8. JSON result generation

## Input

The API accepts:

- `.nii`
- `.nii.gz`

The input must be a compatible 3D cardiac MRI NIfTI volume.

## Start the Backend

From the project root:

```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000