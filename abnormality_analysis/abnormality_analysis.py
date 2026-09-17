import os
import re


# ==========================================================
# HVSMR ABNORMALITY ANALYSIS
# ==========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

PATIENT = "pat7"

FEATURE_REPORT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "hvsmr",
    PATIENT,
    "feature_report.txt"
)

ANALYSIS_REPORT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "hvsmr",
    PATIENT,
    "analysis_report.txt"
)


# ==========================================================
# READ FEATURE REPORT
# ==========================================================

def read_feature_report():

    """
    Read the HVSMR feature report and extract
    cardiac structure volume and surface-area values.
    """

    if not os.path.exists(FEATURE_REPORT_PATH):

        print(
            "Feature report not found!"
        )

        print(
            FEATURE_REPORT_PATH
        )

        return None


    with open(
        FEATURE_REPORT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        lines = file.readlines()


    structures = {}

    current_structure = None


    # ------------------------------------------------------
    # Structure names
    # ------------------------------------------------------

    structure_names = {
        "LEFT VENTRICLE": "Left Ventricle",
        "RIGHT VENTRICLE": "Right Ventricle",
        "LEFT ATRIUM": "Left Atrium",
        "RIGHT ATRIUM": "Right Atrium",
        "AORTA": "Aorta",
        "PULMONARY ARTERY": "Pulmonary Artery",
        "SUPERIOR VENA CAVA": "Superior Vena Cava",
        "INFERIOR VENA CAVA": "Inferior Vena Cava",
    }


    # ------------------------------------------------------
    # Parse report
    # ------------------------------------------------------

    for line in lines:

        line = line.strip()


        if line in structure_names:

            current_structure = structure_names[
                line
            ]

            structures[current_structure] = {}


            continue


        if current_structure is None:

            continue


        # --------------------------------------------------
        # Volume in mL
        # --------------------------------------------------

        if line.startswith("Volume") and "mL" in line:

            try:

                value = float(
                    line.split(":")[1]
                    .replace("mL", "")
                    .strip()
                )

                structures[
                    current_structure
                ]["volume_ml"] = value

            except ValueError:

                pass


        # --------------------------------------------------
        # Surface area
        # --------------------------------------------------

        elif line.startswith("Surface Area"):

            try:

                value = float(
                    line.split(":")[1]
                    .replace("mm²", "")
                    .strip()
                )

                structures[
                    current_structure
                ]["surface_area_mm2"] = value

            except ValueError:

                pass


    return structures


# ==========================================================
# PARAMETER SCREENING
# ==========================================================

def screen_parameter(value):

    """
    Classify a parameter for project-level screening.

    IMPORTANT:
    These categories are not clinical diagnostic thresholds.

    The purpose is to identify parameters that may deserve
    further inspection or comparison.

    Because HVSMR does not provide validated patient-specific
    clinical reference ranges in this pipeline, the safest
    classification is based on whether a measurable value
    exists rather than claiming a medical diagnosis.
    """

    if value is None:

        return "Unavailable"

    if value <= 0:

        return "Requires Review"

    return "Measured"


# ==========================================================
# ANALYZE
# ==========================================================

def analyze():

    print("=" * 70)
    print("HVSMR HEART ABNORMALITY ANALYSIS")
    print("=" * 70)

    print(
        "\nPatient:",
        PATIENT
    )

    print(
        "\nFeature report:"
    )

    print(
        FEATURE_REPORT_PATH
    )


    # ------------------------------------------------------
    # Read features
    # ------------------------------------------------------

    structures = read_feature_report()


    if structures is None:

        return


    # ------------------------------------------------------
    # Create report
    # ------------------------------------------------------

    report = []

    report.append(
        "=" * 70
    )

    report.append(
        "           HEART ABNORMALITY ANALYSIS"
    )

    report.append(
        "=" * 70
    )

    report.append(
        f"Patient: {PATIENT}"
    )

    report.append(
        "\nAnalysis Basis"
    )

    report.append(
        "-" * 45
    )

    report.append(
        "Analysis is based on cardiac parameters "
        "calculated from the U-Net predicted segmentation."
    )

    report.append(
        "The system performs parameter screening and "
        "does not provide a clinical diagnosis."
    )

    report.append(
        "Clinical interpretation requires comparison "
        "with appropriate patient-specific reference "
        "ranges by a qualified medical professional."
    )


    # ======================================================
    # STRUCTURE ANALYSIS
    # ======================================================

    report.append(
        "\nCARDIAC STRUCTURE ANALYSIS"
    )

    report.append(
        "-" * 45
    )


    for structure, values in structures.items():

        volume = values.get(
            "volume_ml"
        )

        surface_area = values.get(
            "surface_area_mm2"
        )


        volume_status = screen_parameter(
            volume
        )

        surface_status = screen_parameter(
            surface_area
        )


        report.append(
            f"\n{structure}"
        )

        report.append(
            "-" * 30
        )


        if volume is not None:

            report.append(
                f"Volume        : "
                f"{volume:.2f} mL"
            )

            report.append(
                f"Volume Status : "
                f"{volume_status}"
            )

        else:

            report.append(
                "Volume        : Unavailable"
            )


        if surface_area is not None:

            report.append(
                f"Surface Area  : "
                f"{surface_area:.2f} mm²"
            )

            report.append(
                f"Surface Status: "
                f"{surface_status}"
            )

        else:

            report.append(
                "Surface Area  : Unavailable"
            )


    # ======================================================
    # OVERALL ASSESSMENT
    # ======================================================

    report.append(
        "\nOVERALL ASSESSMENT"
    )

    report.append(
        "-" * 45
    )

    measured_count = 0

    review_count = 0


    for structure, values in structures.items():

        volume = values.get(
            "volume_ml"
        )

        if volume is not None:

            if volume > 0:

                measured_count += 1

            else:

                review_count += 1


    if review_count > 0:

        overall = (
            "One or more parameters require review. "
            "Further inspection of the segmentation "
            "and clinical evaluation may be required."
        )

    elif measured_count > 0:

        overall = (
            "Cardiac structures were successfully "
            "segmented and measurable parameters were "
            "obtained. No clinical abnormality should "
            "be inferred from these measurements alone."
        )

    else:

        overall = (
            "No usable cardiac measurements were obtained. "
            "The segmentation should be reviewed."
        )


    report.append(
        overall
    )


    # ======================================================
    # DATASET LIMITATION
    # ======================================================

    report.append(
        "\nDATASET LIMITATION"
    )

    report.append(
        "-" * 45
    )

    report.append(
        "HVSMR provides separate labels for the cardiac "
        "chambers and major vessels but does not provide "
        "a separate myocardium label."
    )

    report.append(
        "Therefore myocardium volume and wall-thickness "
        "abnormality are not evaluated in this analysis."
    )


    report.append(
        "\n" + "=" * 70
    )


    # ------------------------------------------------------
    # Convert report to text
    # ------------------------------------------------------

    report_text = "\n".join(
        report
    )


    # ------------------------------------------------------
    # Print
    # ------------------------------------------------------

    print(
        "\n" + report_text
    )


    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            ANALYSIS_REPORT_PATH
        ),
        exist_ok=True
    )


    with open(
        ANALYSIS_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report_text
        )


    print(
        "\nAnalysis report saved successfully."
    )

    print(
        "Location:",
        ANALYSIS_REPORT_PATH
    )


    return {
        "patient": PATIENT,
        "structures": structures,
        "report": report_text
    }


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    analyze()