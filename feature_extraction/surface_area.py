import os
import numpy as np


# ==========================================================
# HVSMR SURFACE AREA CALCULATOR
# ==========================================================

STRUCTURES = {
    "Left Ventricle": "lv",
    "Right Ventricle": "rv",
    "Left Atrium": "la",
    "Right Atrium": "ra",
    "Aorta": "aorta",
    "Pulmonary Artery": "pulmonary_artery",
    "Superior Vena Cava": "svc",
    "Inferior Vena Cava": "ivc",
}


def triangle_area(v1, v2, v3):
    """
    Calculate triangle area using the cross product.
    """

    return 0.5 * np.linalg.norm(
        np.cross(
            v2 - v1,
            v3 - v1,
        )
    )


def calculate_surface_area(output_folder, patient):
    """
    Calculate surface area for the selected patient.

    Expected files:
        <patient>_lv.npz
        <patient>_rv.npz
        ...
        <patient>_ivc.npz

    No patient number is hardcoded.
    """

    if not patient:
        raise ValueError("Patient ID is required.")

    results = {}

    for name, suffix in STRUCTURES.items():

        filename = f"{patient}_{suffix}.npz"
        mesh_file = os.path.join(output_folder, filename)

        print(f"\nProcessing {name}...")

        if not os.path.exists(mesh_file):

            print("Mesh not found:")
            print(mesh_file)

            continue

        data = np.load(mesh_file)

        if "vertices" not in data or "faces" not in data:
            print("NPZ must contain 'vertices' and 'faces', skipping:")
            print(mesh_file)
            continue

        vertices = data["vertices"]
        faces = data["faces"]

        total_area = 0.0

        for face in faces:

            v1 = vertices[face[0]]
            v2 = vertices[face[1]]
            v3 = vertices[face[2]]

            total_area += triangle_area(v1, v2, v3)

        results[name] = float(total_area)

        print(
            f"Surface Area: {total_area:.2f} mm²"
        )

    return results


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate cardiac surface areas."
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

    surface_areas = calculate_surface_area(
        args.output_folder,
        args.patient,
    )

    print("\n========== SURFACE AREA REPORT ==========\n")

    for structure, area in surface_areas.items():

        print(
            f"{structure}: {area:.2f} mm²"
        )

