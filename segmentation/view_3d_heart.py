import os
import pyvista as pv


# -------------------------------------------------
# Settings
# -------------------------------------------------

MODEL_DIR = "outputs/3d_models"

# Different colour for each segmentation label
COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "cyan",
    "magenta",
]


# -------------------------------------------------
# Create 3D viewer
# -------------------------------------------------

plotter = pv.Plotter(
    window_size=(1200, 900)
)

plotter.set_background("white")

legend_entries = []


# -------------------------------------------------
# Load each anatomical label separately
# -------------------------------------------------

for label in range(1, 9):

    stl_path = os.path.join(
        MODEL_DIR,
        f"pat15_label_{label}.stl"
    )

    if not os.path.exists(stl_path):

        print(
            f"Warning: {stl_path} not found"
        )

        continue

    mesh = pv.read(stl_path)

    print(
        f"Label {label}: "
        f"{mesh.n_points} points, "
        f"{mesh.n_cells} faces"
    )

    color = COLORS[label - 1]

    plotter.add_mesh(
        mesh,
        color=color,
        smooth_shading=True,
        opacity=1.0,
        name=f"Label {label}"
    )

    legend_entries.append(
        [
            f"Label {label}",
            color
        ]
    )


# -------------------------------------------------
# Add legend
# -------------------------------------------------

plotter.add_legend(
    labels=legend_entries,
    bcolor="white",
    border=True,
    size=(0.25, 0.30)
)


# -------------------------------------------------
# Axes and grid
# -------------------------------------------------

plotter.add_axes()

plotter.show_grid()


# -------------------------------------------------
# Display
# -------------------------------------------------

plotter.show(
    title="3D Heart Reconstruction - PAT15"
)