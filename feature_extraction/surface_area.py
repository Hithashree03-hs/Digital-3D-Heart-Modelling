import os
import numpy as np


# ==========================================================
# HVSMR SURFACE AREA CALCULATOR
# ==========================================================

STRUCTURES = {
    "Left Ventricle": "pat7_lv.npz",
    "Right Ventricle": "pat7_rv.npz",
    "Left Atrium": "pat7_la.npz",
    "Right Atrium": "pat7_ra.npz",
    "Aorta": "pat7_aorta.npz",
    "Pulmonary Artery": "pat7_pulmonary_artery.npz",
    "Superior Vena Cava": "pat7_svc.npz",
    "Inferior Vena Cava": "pat7_ivc.npz",
}


# ==========================================================
# Triangle Area
# ==========================================================

def triangle_area(v1, v2, v3):

    """
    Calculate triangle area using the cross product.
    """

    return 0.5 * np.linalg.norm(
        np.cross(
            v2 - v1,
            v3 - v1
        )
    )


# ==========================================================
# Surface Area Calculation
# ==========================================================

def calculate_surface_area(
    output_folder="outputs/hvsmr/pat7"
):

    """
    Calculate surface area of each reconstructed
    cardiac structure.

    Mesh coordinates are in millimetres, therefore
    surface area is returned in mm².
    """

    results = {}

    for name, filename in STRUCTURES.items():

        mesh_file = os.path.join(
            output_folder,
            filename
        )

        print(
            f"\nProcessing {name}..."
        )

        if not os.path.exists(mesh_file):

            print(
                "Mesh not found:"
            )

            print(
                mesh_file
            )

            continue

        data = np.load(
            mesh_file
        )

        vertices = data["vertices"]

        faces = data["faces"]

        total_area = 0.0

        # --------------------------------------------------
        # Calculate area of every triangular face
        # --------------------------------------------------

        for face in faces:

            v1 = vertices[
                face[0]
            ]

            v2 = vertices[
                face[1]
            ]

            v3 = vertices[
                face[2]
            ]

            total_area += triangle_area(
                v1,
                v2,
                v3
            )

        results[name] = float(
            total_area
        )

        print(
            f"Surface Area: "
            f"{total_area:.2f} mm²"
        )

    return results


# ==========================================================
# Standalone Testing
# ==========================================================

if __name__ == "__main__":

    surface_areas = calculate_surface_area()

    print(
        "\n========== SURFACE AREA REPORT ==========\n"
    )

    for structure, area in surface_areas.items():

        print(
            f"{structure}: "
            f"{area:.2f} mm²"
        )