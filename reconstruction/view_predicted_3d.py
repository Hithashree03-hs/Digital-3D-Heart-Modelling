import os
import numpy as np
import plotly.graph_objects as go


# ============================================================
# INTERACTIVE 3D HEART VIEWER
# PATIENT-SPECIFIC MODEL FROM U-NET PREDICTION
# ============================================================


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


# ------------------------------------------------------------
# PATIENT
# ------------------------------------------------------------

PATIENT = "pat7"


# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "hvsmr",
    PATIENT
)


# ============================================================
# CARDIAC STRUCTURES
# ============================================================

STRUCTURES = {
    1: {
        "name": "Left Ventricle (LV)",
        "file": "pat7_lv.npz",
        "color": "red",
    },

    2: {
        "name": "Right Ventricle (RV)",
        "file": "pat7_rv.npz",
        "color": "blue",
    },

    3: {
        "name": "Left Atrium (LA)",
        "file": "pat7_la.npz",
        "color": "green",
    },

    4: {
        "name": "Right Atrium (RA)",
        "file": "pat7_ra.npz",
        "color": "orange",
    },

    5: {
        "name": "Aorta",
        "file": "pat7_aorta.npz",
        "color": "purple",
    },

    6: {
        "name": "Pulmonary Artery",
        "file": "pat7_pulmonary_artery.npz",
        "color": "cyan",
    },

    7: {
        "name": "Superior Vena Cava (SVC)",
        "file": "pat7_svc.npz",
        "color": "magenta",
    },

    8: {
        "name": "Inferior Vena Cava (IVC)",
        "file": "pat7_ivc.npz",
        "color": "gray",
    },
}


# ============================================================
# START
# ============================================================

print("=" * 70)
print("PATIENT-SPECIFIC INTERACTIVE 3D HEART VIEWER")
print("=" * 70)

print("\nPatient:", PATIENT)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# CREATE FIGURE
# ============================================================

fig = go.Figure()

structures_loaded = 0


# ============================================================
# LOAD EACH 3D MESH
# ============================================================

for label, structure in STRUCTURES.items():

    mesh_file = os.path.join(
        OUTPUT_DIR,
        structure["file"]
    )

    print(
        f"\nLoading: {structure['name']}"
    )

    if not os.path.exists(mesh_file):

        print(
            "WARNING: Mesh file not found:"
        )

        print(
            mesh_file
        )

        continue


    # --------------------------------------------------------
    # LOAD NPZ
    # --------------------------------------------------------

    data = np.load(
        mesh_file
    )

    vertices = data["vertices"]
    faces = data["faces"]


    print(
        "Vertices:",
        f"{len(vertices):,}"
    )

    print(
        "Faces:",
        f"{len(faces):,}"
    )


    # --------------------------------------------------------
    # CREATE INTERACTIVE MESH
    # --------------------------------------------------------

    fig.add_trace(
        go.Mesh3d(

            # Coordinates
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],

            # Triangle indices
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],

            # Structure information
            name=structure["name"],

            legendgroup=structure["name"],

            # Appearance
            color=structure["color"],
            opacity=0.70,

            flatshading=False,

            lighting=dict(
                ambient=0.55,
                diffuse=0.75,
                specular=0.35,
                roughness=0.45,
                fresnel=0.15,
            ),

            lightposition=dict(
                x=100,
                y=100,
                z=200,
            ),

            # Hover information
            hovertemplate=(
                "<b>"
                + structure["name"]
                + "</b><br><br>"
                "X: %{x:.1f} mm<br>"
                "Y: %{y:.1f} mm<br>"
                "Z: %{z:.1f} mm"
                "<extra></extra>"
            ),

            showlegend=True,
        )
    )

    structures_loaded += 1


# ============================================================
# CHECK
# ============================================================

if structures_loaded == 0:

    raise RuntimeError(
        "\nNo 3D mesh files were found."
        "\nRun reconstruct_hvsmr.py first."
    )


# ============================================================
# FIGURE LAYOUT
# ============================================================

fig.update_layout(

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    title={
        "text": (
            "Patient-Specific 3D Heart Model — "
            + PATIENT
        ),

        "x": 0.5,

        "xanchor": "center",

        "font": {
            "size": 22
        },
    },


    # --------------------------------------------------------
    # 3D SCENE
    # --------------------------------------------------------

    scene=dict(

        # X axis
        xaxis=dict(
            title="X (mm)",

            showgrid=False,

            zeroline=False,

            showbackground=False,

            showline=False,

            showticklabels=False,
        ),

        # Y axis
        yaxis=dict(
            title="Y (mm)",

            showgrid=False,

            zeroline=False,

            showbackground=False,

            showline=False,

            showticklabels=False,
        ),

        # Z axis
        zaxis=dict(
            title="Z (mm)",

            showgrid=False,

            zeroline=False,

            showbackground=False,

            showline=False,

            showticklabels=False,
        ),


        # Preserve physical proportions
        aspectmode="data",


        # Initial camera position
        camera=dict(

            eye=dict(
                x=1.6,
                y=1.6,
                z=1.2,
            ),

            up=dict(
                x=0,
                y=0,
                z=1,
            ),
        ),
    ),


    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    legend=dict(

        title={
            "text": "Cardiac Structures"
        },

        x=0.01,

        y=0.99,

        xanchor="left",

        yanchor="top",

        bgcolor="rgba(255,255,255,0.90)",

        bordercolor="rgba(0,0,0,0.15)",

        borderwidth=1,

        font=dict(
            size=13
        ),
    ),


    # --------------------------------------------------------
    # GENERAL APPEARANCE
    # --------------------------------------------------------

    paper_bgcolor="white",

    plot_bgcolor="white",

    margin=dict(
        l=0,
        r=0,
        t=65,
        b=0,
    ),


    # --------------------------------------------------------
    # MODEBAR
    # --------------------------------------------------------

    modebar=dict(
        orientation="v"
    ),
)


# ============================================================
# SAVE INTERACTIVE HTML
# ============================================================

html_file = os.path.join(
    OUTPUT_DIR,
    f"{PATIENT}_interactive_3d_heart.html"
)


fig.write_html(
    html_file,
    include_plotlyjs=True
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("INTERACTIVE MODEL CREATED")
print("=" * 70)

print(
    "\nPatient:",
    PATIENT
)

print(
    "Structures loaded:",
    structures_loaded
)

print(
    "\nHTML file:"
)

print(
    html_file
)

print("\nOpening browser...")


# ============================================================
# OPEN BROWSER
# ============================================================

fig.show()


# ============================================================
# FINISHED
# ============================================================

print("\nDone!")

print("=" * 70)