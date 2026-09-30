import os
import numpy as np


# ==========================================================
# HVSMR HEART DIMENSION CALCULATOR
# ==========================================================

STRUCTURES = [
    "lv",
    "rv",
    "la",
    "ra",
    "aorta",
    "pulmonary_artery",
    "svc",
    "ivc",
]


def calculate_dimensions(output_folder, patient):
    """
    Calculate overall heart dimensions for the selected patient.

    Expected mesh files:
        <patient>_lv.npz
        <patient>_rv.npz
        <patient>_la.npz
        <patient>_ra.npz
        <patient>_aorta.npz
        <patient>_pulmonary_artery.npz
        <patient>_svc.npz
        <patient>_ivc.npz

    The patient ID is supplied at runtime; nothing is hardcoded to pat7.
    """

    if not patient:
        raise ValueError("Patient ID is required.")

    all_vertices = []

    for structure in STRUCTURES:

        filename = f"{patient}_{structure}.npz"
        mesh_file = os.path.join(output_folder, filename)

        if not os.path.exists(mesh_file):

            print("Mesh not found, skipping:")
            print(mesh_file)

            continue

        data = np.load(mesh_file)

        if "vertices" not in data:
            print("No 'vertices' array found, skipping:")
            print(mesh_file)
            continue

        vertices = data["vertices"]

        if vertices.size == 0:
            print("Empty vertex array, skipping:")
            print(mesh_file)
            continue

        all_vertices.append(vertices)

    if not all_vertices:
        raise FileNotFoundError(
            f"No cardiac NPZ meshes were found for patient '{patient}' "
            f"in:\n{output_folder}"
        )

    vertices = np.vstack(all_vertices)

    min_x, min_y, min_z = vertices.min(axis=0)
    max_x, max_y, max_z = vertices.max(axis=0)

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
            float(min_z),
        ),
        "Max Coordinates": (
            float(max_x),
            float(max_y),
            float(max_z),
        ),
    }


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate overall cardiac dimensions."
    )

    parser.add_argument(
        "patient",
        help="Patient ID, for example pat7, pat9, pat15"
    )

    parser.add_argument(
        "--output-folder",
        default=None,
        help="Patient mesh folder. Defaults to outputs/hvsmr/<patient>."
    )

    args = parser.parse_args()

    if args.output_folder is None:
        args.output_folder = os.path.join(
            "outputs",
            "hvsmr",
            args.patient,
        )

    dimensions = calculate_dimensions(
        args.output_folder,
        args.patient,
    )

    print("\n========== HEART DIMENSIONS ==========\n")
    print(f"Patient: {args.patient}")
    print(f"Width  : {dimensions['Width']:.2f} mm")
    print(f"Height : {dimensions['Height']:.2f} mm")
    print(f"Depth  : {dimensions['Depth']:.2f} mm")

    print("\nBounding Box")
    print("Min:", dimensions["Min Coordinates"])
    print("Max:", dimensions["Max Coordinates"])

