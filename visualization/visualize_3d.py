import os
import numpy as np
import pyvista as pv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)


# ============================================================
# HEART STRUCTURES
# ============================================================

STRUCTURES = {
    "lv": {
        "label": "Left Ventricle",
        "color": "#C62828",
        "opacity": 1.0,
    },

    "myo": {
        "label": "Myocardium",
        "color": "#FBC02D",
        "opacity": 0.45,
    },

    "rv": {
        "label": "Right Ventricle",
        "color": "#1E88E5",
        "opacity": 0.95,
    },
}


# ============================================================
# LOAD MESH
# ============================================================

def load_mesh(prefix):

    vertex_path = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_vertices.npy"
    )

    face_path = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_faces.npy"
    )

    if not os.path.exists(vertex_path):

        print(
            f"WARNING: {vertex_path} not found."
        )

        return None

    if not os.path.exists(face_path):

        print(
            f"WARNING: {face_path} not found."
        )

        return None

    vertices = np.load(
        vertex_path
    )

    faces = np.load(
        face_path
    )

    print()
    print("=" * 60)
    print(f"Loading {prefix.upper()}")
    print("=" * 60)

    print(
        "Vertices :",
        vertices.shape
    )

    print(
        "Faces    :",
        faces.shape
    )

    # --------------------------------------------------------
    # PyVista requires:
    #
    # [3, v1, v2, v3]
    # [3, v1, v2, v3]
    # ...
    # --------------------------------------------------------

    faces_pv = np.hstack(
        (
            np.full(
                (faces.shape[0], 1),
                3,
                dtype=np.int64
            ),

            faces.astype(
                np.int64
            )
        )
    ).ravel()

    mesh = pv.PolyData(
        vertices,
        faces_pv
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    mesh = mesh.clean()

    # --------------------------------------------------------
    # Compute normals
    # --------------------------------------------------------

    mesh.compute_normals(
        inplace=True,
        auto_orient_normals=True,
        consistent_normals=True
    )

    return mesh


# ============================================================
# MAIN VISUALIZATION
# ============================================================

def visualize():

    print()
    print("=" * 70)
    print("          DIGITAL 3D HEART MODEL")
    print("=" * 70)

    plotter = pv.Plotter(
        window_size=(1400, 900)
    )

    plotter.set_background(
        "white"
    )

    loaded = []

    # --------------------------------------------------------
    # Load structures
    # --------------------------------------------------------

    for prefix, info in STRUCTURES.items():

        mesh = load_mesh(
            prefix
        )

        if mesh is None:
            continue

        loaded.append(
            prefix
        )

        # ----------------------------------------------------
        # Smooth only for visualization
        # ----------------------------------------------------

        try:

            mesh = mesh.smooth(
                n_iter=30,
                relaxation_factor=0.01,
                boundary_smoothing=False,
                feature_smoothing=False
            )

        except Exception as error:

            print(
                "Smoothing skipped:",
                error
            )

        # ----------------------------------------------------
        # Add mesh
        # ----------------------------------------------------

        plotter.add_mesh(
            mesh,

            color=info["color"],

            opacity=info["opacity"],

            smooth_shading=True,

            specular=0.35,

            specular_power=20,

            ambient=0.25,

            diffuse=0.75,

            label=info["label"]
        )

    # --------------------------------------------------------
    # Check whether anything loaded
    # --------------------------------------------------------

    if len(loaded) == 0:

        raise FileNotFoundError(
            "\nNo heart meshes were found "
            "inside the outputs folder."
        )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    plotter.view_isometric()

    plotter.camera.zoom(
        1.4
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    plotter.show_axes()

    # --------------------------------------------------------
    # Orientation cube
    # --------------------------------------------------------

    try:

        plotter.add_orientation_cube()

    except Exception:

        pass

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend = []

    if "lv" in loaded:

        legend.append(
            ("Left Ventricle", "#C62828")
        )

    if "myo" in loaded:

        legend.append(
            ("Myocardium", "#FBC02D")
        )

    if "rv" in loaded:

        legend.append(
            ("Right Ventricle", "#1E88E5")
        )

    if legend:

        plotter.add_legend(
            labels=legend,
            bcolor="white",
            border=True,
            size=(0.20, 0.15)
        )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    plotter.add_text(
        "Digital 3D Heart Model",
        position="upper_left",
        font_size=18,
        color="black"
    )

    plotter.add_text(
        "MRI-based cardiac reconstruction",
        position="upper_right",
        font_size=11,
        color="black"
    )

    # --------------------------------------------------------
    # Save screenshot
    # --------------------------------------------------------

    screenshot_path = os.path.join(
        OUTPUT_DIR,
        "heart_3d_model.png"
    )

    try:

        plotter.show(
            title="Digital 3D Heart Modelling",
            auto_close=False
        )

        plotter.screenshot(
            screenshot_path
        )

        print()
        print(
            "Screenshot saved:"
        )

        print(
            screenshot_path
        )

    finally:

        plotter.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    visualize()