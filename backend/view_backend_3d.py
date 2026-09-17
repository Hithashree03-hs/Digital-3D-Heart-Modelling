import os
from pathlib import Path

import trimesh


# ============================================================
# PATIENT TO VISUALIZE
# ============================================================

PATIENT_ID = "api_test_01"


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = (
    BASE_DIR
    / "outputs"
    / PATIENT_ID
    / "3d_models"
)


# ============================================================
# STRUCTURE COLORS
# RGB + Alpha
# ============================================================

STRUCTURE_COLORS = {
    "left_ventricle": [220, 40, 40, 255],
    "right_ventricle": [40, 90, 220, 255],
    "left_atrium": [240, 150, 40, 255],
    "right_atrium": [170, 70, 220, 255],
    "aorta": [230, 70, 70, 255],
    "pulmonary_artery": [50, 180, 180, 255],
    "superior_vena_cava": [80, 190, 90, 255],
    "inferior_vena_cava": [240, 200, 50, 255]
}


# ============================================================
# CHECK DIRECTORY
# ============================================================

if not MODEL_DIR.exists():

    raise FileNotFoundError(
        f"3D model directory not found:\n"
        f"{MODEL_DIR}"
    )


print("=" * 70)
print("BACKEND 3D HEART VISUALIZATION")
print("=" * 70)

print(
    f"\nPatient: {PATIENT_ID}"
)

print(
    f"Model directory:\n{MODEL_DIR}"
)


# ============================================================
# CREATE SCENE
# ============================================================

scene = trimesh.Scene()

loaded_count = 0


# ============================================================
# LOAD INDIVIDUAL STRUCTURES
# ============================================================

for structure, color in STRUCTURE_COLORS.items():

    model_path = (
        MODEL_DIR /
        f"{PATIENT_ID}_{structure}.stl"
    )

    if not model_path.exists():

        print(
            f"\n[SKIP] Not found: "
            f"{model_path.name}"
        )

        continue

    try:

        mesh = trimesh.load(
            model_path,
            force="mesh"
        )

        # Apply structure-specific color
        mesh.visual.face_colors = color

        scene.add_geometry(
            mesh,
            node_name=structure
        )

        print(
            f"[LOADED] "
            f"{structure.replace('_', ' ').title()}"
        )

        loaded_count += 1

    except Exception as exc:

        print(
            f"[ERROR] "
            f"{model_path.name}: {exc}"
        )


# ============================================================
# CHECK LOADED STRUCTURES
# ============================================================

if loaded_count == 0:

    raise RuntimeError(
        "No individual 3D structure models "
        "could be loaded."
    )


print(
    f"\nStructures loaded: "
    f"{loaded_count}/8"
)


# ============================================================
# VISUALIZE
# ============================================================

print("\nOpening 3D viewer...")
print(
    "Close the viewer window when finished."
)

scene.show()