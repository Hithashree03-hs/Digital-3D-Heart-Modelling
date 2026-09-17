import os

from volume_calculator import calculate_volumes
from dimension_calculator import calculate_dimensions
from surface_area import calculate_surface_area


# ==========================================================
# HVSMR COMPLETE FEATURE REPORT
# ==========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


PATIENT = "pat7"


VOLUME_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "hvsmr",
    PATIENT,
    "predicted_segmentation.nii.gz"
)


MESH_FOLDER = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "hvsmr",
    PATIENT
)


REPORT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "hvsmr",
    PATIENT,
    "feature_report.txt"
)


# ==========================================================
# Generate Report
# ==========================================================

def generate_report():

    print("=" * 70)
    print("HVSMR CARDIAC FEATURE EXTRACTION")
    print("=" * 70)

    print(
        "\nPatient:",
        PATIENT
    )

    # ------------------------------------------------------
    # Check input
    # ------------------------------------------------------

    if not os.path.exists(VOLUME_PATH):

        raise FileNotFoundError(
            "\nPredicted segmentation not found:\n"
            + VOLUME_PATH
        )

    if not os.path.exists(MESH_FOLDER):

        raise FileNotFoundError(
            "\nMesh folder not found:\n"
            + MESH_FOLDER
        )

    # ------------------------------------------------------
    # Calculate volumes
    # ------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CALCULATING VOLUMES"
    )

    print(
        "=" * 70
    )

    volumes = calculate_volumes(
        VOLUME_PATH
    )

    # ------------------------------------------------------
    # Calculate dimensions
    # ------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CALCULATING HEART DIMENSIONS"
    )

    print(
        "=" * 70
    )

    dimensions = calculate_dimensions(
        MESH_FOLDER
    )

    # ------------------------------------------------------
    # Calculate surface areas
    # ------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CALCULATING SURFACE AREAS"
    )

    print(
        "=" * 70
    )

    surface_areas = calculate_surface_area(
        MESH_FOLDER
    )

    # ------------------------------------------------------
    # Build report
    # ------------------------------------------------------

    report = []

    report.append(
        "=" * 70
    )

    report.append(
        "           HVSMR CARDIAC FEATURE REPORT"
    )

    report.append(
        "=" * 70
    )

    report.append(
        f"Patient: {PATIENT}"
    )

    report.append(
        "\nNOTE:"
    )

    report.append(
        "Measurements are calculated from the "
        "U-Net predicted segmentation and reconstructed "
        "3D cardiac meshes."
    )

    report.append(
        "\nHVSMR does not provide a separate myocardium label; "
        "therefore myocardium measurements are not included."
    )


    # ======================================================
    # STRUCTURE FEATURES
    # ======================================================

    for structure, values in volumes.items():

        report.append(
            "\n" + structure.upper()
        )

        report.append(
            "-" * 45
        )

        report.append(
            f"Label         : "
            f"{values['label']}"
        )

        report.append(
            f"Voxel Count   : "
            f"{values['voxels']:,}"
        )

        report.append(
            f"Volume        : "
            f"{values['volume_ml']:.2f} mL"
        )

        report.append(
            f"Volume        : "
            f"{values['volume_mm3']:.2f} mm³"
        )

        if structure in surface_areas:

            report.append(
                f"Surface Area  : "
                f"{surface_areas[structure]:.2f} mm²"
            )

        else:

            report.append(
                "Surface Area  : Not available"
            )


    # ======================================================
    # OVERALL HEART DIMENSIONS
    # ======================================================

    report.append(
        "\nOVERALL HEART DIMENSIONS"
    )

    report.append(
        "-" * 45
    )

    report.append(
        f"Width         : "
        f"{dimensions['Width']:.2f} mm"
    )

    report.append(
        f"Height        : "
        f"{dimensions['Height']:.2f} mm"
    )

    report.append(
        f"Depth         : "
        f"{dimensions['Depth']:.2f} mm"
    )

    report.append(
        "\nBounding Box"
    )

    report.append(
        f"Minimum       : "
        f"{dimensions['Min Coordinates']}"
    )

    report.append(
        f"Maximum       : "
        f"{dimensions['Max Coordinates']}"
    )


    report.append(
        "\n" + "=" * 70
    )


    # ------------------------------------------------------
    # Convert to text
    # ------------------------------------------------------

    report_text = "\n".join(
        report
    )


    # ------------------------------------------------------
    # Print report
    # ------------------------------------------------------

    print(
        "\n" + report_text
    )


    # ------------------------------------------------------
    # Save report
    # ------------------------------------------------------

    os.makedirs(
        os.path.dirname(REPORT_PATH),
        exist_ok=True
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report_text
        )


    print(
        "\nFeature report saved successfully."
    )

    print(
        "Location:",
        REPORT_PATH
    )


    # ------------------------------------------------------
    # Return results
    # ------------------------------------------------------

    return {

        "patient": PATIENT,

        "volumes": volumes,

        "dimensions": dimensions,

        "surface_areas": surface_areas,

        "report": report_text

    }


# ==========================================================
# Standalone Execution
# ==========================================================

if __name__ == "__main__":

    generate_report()