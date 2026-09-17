import os
import pyvista as pv


# ---------------------------------------------------------
# Patient
# ---------------------------------------------------------

PATIENT_ID = "pat17"

MODEL_DIR = (
    f"outputs/test_3d_models/{PATIENT_ID}"
)


# ---------------------------------------------------------
# Structure names and colours
# ---------------------------------------------------------

STRUCTURES = [
    ("Left Ventricle", "red"),
    ("Right Ventricle", "blue"),
    ("Left Atrium", "green"),
    ("Right Atrium", "orange"),
    ("Aorta", "yellow"),
    ("Pulmonary Artery", "purple"),
    ("Superior Vena Cava", "cyan"),
    ("Inferior Vena Cava", "magenta"),
]


# ---------------------------------------------------------
# Create viewer
# ---------------------------------------------------------

plotter = pv.Plotter(
    window_size=(1300, 900)
)

plotter.set_background("white")

legend_entries = []


# ---------------------------------------------------------
# Load structures
# ---------------------------------------------------------

for structure_name, color in STRUCTURES:

    filename = (
        structure_name
        .lower()
        .replace(" ", "_")
    )

    path = os.path.join(
        MODEL_DIR,
        f"{PATIENT_ID}_{filename}.stl"
    )

    if not os.path.exists(path):

        print(
            "Missing:",
            path
        )

        continue

    mesh = pv.read(path)

    print(
        f"{structure_name}: "
        f"{mesh.n_points} points, "
        f"{mesh.n_cells} faces"
    )

    plotter.add_mesh(
        mesh,
        color=color,
        smooth_shading=True,
        name=structure_name
    )

    legend_entries.append(
        [structure_name, color]
    )


# ---------------------------------------------------------
# Legend
# ---------------------------------------------------------

plotter.add_legend(
    labels=legend_entries,
    bcolor="white",
    border=True,
    size=(0.30, 0.35)
)


# ---------------------------------------------------------
# Axes
# ---------------------------------------------------------

plotter.add_axes()

plotter.show_grid()


# ---------------------------------------------------------
# Show
# ---------------------------------------------------------

plotter.show(
    title="PAT17 – Patient-Specific 3D Heart Model"
)