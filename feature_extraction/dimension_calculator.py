import os
import numpy as np


# ==========================================================
# HVSMR HEART DIMENSION CALCULATOR
# ==========================================================

STRUCTURE_FILES = [
    "pat7_lv.npz",
    "pat7_rv.npz",
    "pat7_la.npz",
    "pat7_ra.npz",
    "pat7_aorta.npz",
    "pat7_pulmonary_artery.npz",
    "pat7_svc.npz",
    "pat7_ivc.npz",
]


# ==========================================================
# Calculate Heart Dimensions
# ==========================================================

def calculate_dimensions(
    output_folder="outputs/hvsmr/pat7"
):

    """
    Calculate the overall dimensions of the reconstructed
    cardiac structures using their physical mesh coordinates.

    Returns
    -------
    dict
        Width, height, depth and bounding-box coordinates
        in millimetres.
    """

    all_vertices = []

    # ------------------------------------------------------
    # Load all available cardiac meshes
    # ------------------------------------------------------

    for filename in STRUCTURE_FILES:

        mesh_file = os.path.join(
            output_folder,
            filename
        )

        if not os.path.exists(mesh_file):

            print(
                "Mesh not found, skipping:"
            )

            print(
                mesh_file
            )

            continue

        data = np.load(
            mesh_file
        )

        vertices = data["vertices"]

        all_vertices.append(
            vertices
        )

    # ------------------------------------------------------
    # Check
    # ------------------------------------------------------

    if len(all_vertices) == 0:

        raise FileNotFoundError(
            "No cardiac mesh files were found."
        )

    # ------------------------------------------------------
    # Combine vertices
    # ------------------------------------------------------

    vertices = np.vstack(
        all_vertices
    )

    # ------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------

    min_x, min_y, min_z = (
        vertices.min(axis=0)
    )

    max_x, max_y, max_z = (
        vertices.max(axis=0)
    )

    width = max_x - min_x

    height = max_y - min_y

    depth = max_z - min_z

    return {

        "Width": float(width),

        "Height": float(height),

        "Depth": float(depth),

        "Min Coordinates": (
            float(min_x),
            float(min_y),
            float(min_z)
        ),

        "Max Coordinates": (
            float(max_x),
            float(max_y),
            float(max_z)
        )

    }


# ==========================================================
# Standalone Testing
# ==========================================================

if __name__ == "__main__":

    dimensions = calculate_dimensions()

    print(
        "\n========== HEART DIMENSIONS ==========\n"
    )

    print(
        f"Width  : "
        f"{dimensions['Width']:.2f} mm"
    )

    print(
        f"Height : "
        f"{dimensions['Height']:.2f} mm"
    )

    print(
        f"Depth  : "
        f"{dimensions['Depth']:.2f} mm"
    )

    print(
        "\nBounding Box"
    )

    print(
        "Min:",
        dimensions["Min Coordinates"]
    )

    print(
        "Max:",
        dimensions["Max Coordinates"]
    )