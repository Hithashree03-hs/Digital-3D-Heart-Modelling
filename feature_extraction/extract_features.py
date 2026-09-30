import os
import argparse

from volume_calculator import calculate_volumes
from dimension_calculator import calculate_dimensions
from surface_area import calculate_surface_area


# ==========================================================
# HVSMR COMPLETE FEATURE REPORT
# ==========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)


def generate_report(patient):
    """
    Generate a cardiac feature report for ANY selected patient.

    Example:
        generate_report("pat7")
        generate_report("pat9")
        generate_report("pat15")

    The patient ID is supplied by the caller. It is NOT hardcoded.
    """

    if not patient:
        raise ValueError("Patient ID is required.")

    patient = str(patient).strip()

    if not patient:
        raise ValueError("Patient ID cannot be empty.")

    patient_folder = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "hvsmr",
        patient,
    )

    volume_path = os.path.join(
        patient_folder,
        "predicted_segmentation.nii.gz",
    )

    mesh_folder = patient_folder

    report_path = os.path.join(
        patient_folder,
        "feature_report.txt",
    )

    print("=" * 70)
    print("HVSMR CARDIAC FEATURE EXTRACTION")
    print("=" * 70)

    print("\nPatient:", patient)

    # ------------------------------------------------------
    # Check input
    # ------------------------------------------------------

    if not os.path.exists(volume_path):

        raise FileNotFoundError(
            "\nPredicted segmentation not found:\n"
            + volume_path
        )

    if not os.path.exists(mesh_folder):

        raise FileNotFoundError(
            "\nPatient output folder not found:\n"
            + mesh_folder
        )

    # ------------------------------------------------------
    # Calculate volumes
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("CALCULATING VOLUMES")
    print("=" * 70)

    volumes = calculate_volumes(volume_path)

    # ------------------------------------------------------
    # Calculate dimensions
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("CALCULATING HEART DIMENSIONS")
    print("=" * 70)

    dimensions = calculate_dimensions(
        mesh_folder,
        patient,
    )

    # ------------------------------------------------------
    # Calculate surface areas
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("CALCULATING SURFACE AREAS")
    print("=" * 70)

    surface_areas = calculate_surface_area(
        mesh_folder,
        patient,
    )

    # ------------------------------------------------------
    # Build report
    # ------------------------------------------------------

    report = []

    report.append("=" * 70)
    report.append("           HVSMR CARDIAC FEATURE REPORT")
    report.append("=" * 70)
    report.append(f"Patient: {patient}")

    report.append("\nNOTE:")

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

        report.append("\n" + structure.upper())
        report.append("-" * 45)

        report.append(
            f"Label         : {values['label']}"
        )

        report.append(
            f"Voxel Count   : {values['voxels']:,}"
        )

        report.append(
            f"Volume        : {values['volume_ml']:.2f} mL"
        )

        report.append(
            f"Volume        : {values['volume_mm3']:.2f} mm³"
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

    report.append("\nOVERALL HEART DIMENSIONS")
    report.append("-" * 45)

    report.append(
        f"Width         : {dimensions['Width']:.2f} mm"
    )

    report.append(
        f"Height        : {dimensions['Height']:.2f} mm"
    )

    report.append(
        f"Depth         : {dimensions['Depth']:.2f} mm"
    )

    report.append("\nBounding Box")

    report.append(
        f"Minimum       : {dimensions['Min Coordinates']}"
    )

    report.append(
        f"Maximum       : {dimensions['Max Coordinates']}"
    )

    report.append("\n" + "=" * 70)

    report_text = "\n".join(report)

    print("\n" + report_text)

    os.makedirs(
        os.path.dirname(report_path),
        exist_ok=True,
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(report_text)

    print("\nFeature report saved successfully.")
    print("Location:", report_path)

    return {
        "patient": patient,
        "volumes": volumes,
        "dimensions": dimensions,
        "surface_areas": surface_areas,
        "report": report_text,
    }


# ==========================================================
# Standalone Execution
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Generate HVSMR cardiac features for the selected patient."
        )
    )

    parser.add_argument(
        "patient",
        help="Patient ID, e.g. pat7, pat9, pat15",
    )

    args = parser.parse_args()

    generate_report(args.patient)

