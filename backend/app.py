from pathlib import Path
from typing import List
import json
import shutil
import uuid

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import FileResponse

from backend.pipeline import analyze_patient


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent

UPLOADS_DIR = BACKEND_DIR / "uploads"

OUTPUTS_DIR = BACKEND_DIR / "outputs"


UPLOADS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Digital 3D Heart Analysis API",

    description=(
        "Backend API for Digital 3D Heart Modelling "
        "for Cardiac Structure and Abnormality Analysis"
    ),

    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# CARDIAC STRUCTURE ORDER
# ============================================================

STRUCTURE_ORDER = {
    "Left Ventricle": 1,
    "Right Ventricle": 2,
    "Left Atrium": 3,
    "Right Atrium": 4,
    "Aorta": 5,
    "Pulmonary Artery": 6,
    "Superior Vena Cava": 7,
    "Inferior Vena Cava": 8,
}


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",

        "service":
            "Digital 3D Heart Analysis API",

        "version":
            "1.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",

        "service":
            "Digital 3D Heart Analysis API",
    }


# ============================================================
# SAVE UPLOAD
# ============================================================

def save_upload_file(
    upload_file: UploadFile,
    destination: Path
):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with destination.open("wb") as buffer:

        shutil.copyfileobj(
            upload_file.file,
            buffer
        )


# ============================================================
# PATIENT OUTPUT DIRECTORY
# ============================================================

def get_patient_output_dir(
    patient_id: str
) -> Path:

    # Prevent path traversal.
    safe_patient_id = Path(
        patient_id
    ).name

    return (
        OUTPUTS_DIR /
        safe_patient_id
    )


# ============================================================
# STRUCTURE NAME DETECTION
# ============================================================

def identify_structure(
    filename: str
):

    """
    Identify which cardiac structure a generated
    model file belongs to.
    """

    name = (
        filename
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    # Check specific names first.

    if "left_ventricle" in name:
        return "Left Ventricle"

    if "right_ventricle" in name:
        return "Right Ventricle"

    if "left_atrium" in name:
        return "Left Atrium"

    if "right_atrium" in name:
        return "Right Atrium"

    if "pulmonary_artery" in name:
        return "Pulmonary Artery"

    if "superior_vena_cava" in name:
        return "Superior Vena Cava"

    if "inferior_vena_cava" in name:
        return "Inferior Vena Cava"

    if "aorta" in name:
        return "Aorta"

    return None


# ============================================================
# FIND MODEL FILE
# ============================================================

def find_model_file(
    patient_id: str,
    filename: str
):

    """
    Safely locate a generated 3D model belonging
    to a specific patient.
    """

    safe_patient_id = Path(
        patient_id
    ).name

    safe_filename = Path(
        filename
    ).name

    patient_dir = (
        OUTPUTS_DIR /
        safe_patient_id
    )

    if not patient_dir.exists():
        return None

    allowed_extensions = {
        ".stl",
        ".ply",
        ".obj",
        ".glb",
        ".gltf",
    }

    extension = (
        Path(safe_filename)
        .suffix
        .lower()
    )

    if extension not in allowed_extensions:
        return None

    # Search recursively for the exact filename.

    matches = [
        path
        for path in patient_dir.rglob(
            safe_filename
        )
        if path.is_file()
    ]

    if matches:
        return matches[0]

    return None


# ============================================================
# ANALYZE NIFTI MRI VOLUME
# ============================================================

@app.post("/analyze")
async def analyze_mri(
    file: UploadFile = File(...)
):

    """
    Analyze a 3D NIfTI MRI volume.

    Pipeline:

        MRI
         ↓
        Preprocessing
         ↓
        2D U-Net Segmentation
         ↓
        3D Segmentation
         ↓
        3D Reconstruction
         ↓
        Feature Extraction
         ↓
        Morphological Abnormality Screening
    """

    # --------------------------------------------------------
    # VALIDATE FILE
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,

            detail=
                "No MRI file was selected."
        )

    filename = Path(
        file.filename
    ).name

    lower_filename = (
        filename.lower()
    )

    if not (
        lower_filename.endswith(".nii")
        or
        lower_filename.endswith(".nii.gz")
    ):

        raise HTTPException(
            status_code=400,

            detail=(
                "Invalid MRI format. "
                "Please upload a .nii or .nii.gz file."
            )
        )

    # --------------------------------------------------------
    # CREATE PATIENT ID
    # --------------------------------------------------------

    patient_id = (
        f"patient_{uuid.uuid4().hex[:8]}"
    )

    patient_upload_dir = (
        UPLOADS_DIR /
        patient_id
    )

    patient_upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    input_path = (
        patient_upload_dir /
        filename
    )

    # --------------------------------------------------------
    # SAVE MRI
    # --------------------------------------------------------

    try:

        save_upload_file(
            file,
            input_path
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,

            detail=(
                "Failed to save uploaded MRI: "
                f"{exc}"
            )
        )

    # --------------------------------------------------------
    # RUN PROJECT PIPELINE
    # --------------------------------------------------------

    try:

        result = analyze_patient(
            str(input_path),

            patient_id=patient_id
        )

    except Exception as exc:

        print(
            f"MRI analysis failed for "
            f"{patient_id}: {exc}"
        )

        raise HTTPException(
            status_code=500,

            detail=(
                f"MRI analysis failed: {exc}"
            )
        )

    # --------------------------------------------------------
    # VALIDATE RESULT
    # --------------------------------------------------------

    if result is None:

        raise HTTPException(
            status_code=500,

            detail=(
                "Analysis completed without "
                "returning a result."
            )
        )

    if not isinstance(result, dict):

        raise HTTPException(
            status_code=500,

            detail=(
                "Analysis returned an invalid "
                "result format."
            )
        )

    # --------------------------------------------------------
    # ATTACH GENERATED 3D MODELS
    # --------------------------------------------------------

    try:

        patient_output_dir = (
            get_patient_output_dir(
                patient_id
            )
        )

        allowed_extensions = {
            ".stl",
            ".ply",
            ".obj",
            ".glb",
            ".gltf",
        }

        generated_models = []

        if patient_output_dir.exists():

            for model_path in patient_output_dir.rglob("*"):

                if not model_path.is_file():
                    continue

                if (
                    model_path.suffix.lower()
                    not in allowed_extensions
                ):
                    continue

                structure = identify_structure(
                    model_path.name
                )

                if structure is None:
                    continue

                generated_models.append({
                    "filename":
                        model_path.name,

                    "structure":
                        structure,

                    "url":
                        (
                            f"/models/"
                            f"{patient_id}/"
                            f"{model_path.name}"
                        ),
                })

        # ----------------------------------------------------
        # REMOVE DUPLICATE STRUCTURES
        # ----------------------------------------------------

        unique_models = []

        seen_structures = set()

        for model in generated_models:

            structure = (
                model["structure"]
            )

            if structure in seen_structures:
                continue

            seen_structures.add(
                structure
            )

            unique_models.append(
                model
            )

        # ----------------------------------------------------
        # SORT CARDIAC STRUCTURES
        # ----------------------------------------------------

        unique_models.sort(
            key=lambda model:
                STRUCTURE_ORDER.get(
                    model["structure"],
                    99
                )
        )

        # ----------------------------------------------------
        # CREATE RECONSTRUCTION SECTION
        # ----------------------------------------------------

        if (
            "3d_reconstruction"
            not in result
        ):

            result[
                "3d_reconstruction"
            ] = {}

        result[
            "3d_reconstruction"
        ][
            "individual_models"
        ] = unique_models

        result[
            "3d_reconstruction"
        ][
            "model_count"
        ] = len(
            unique_models
        )

        print(
            f"3D models found for "
            f"{patient_id}: "
            f"{len(unique_models)}"
        )

        for model in unique_models:

            print(
                f"  - "
                f"{model['structure']}: "
                f"{model['filename']}"
            )

    except Exception as exc:

        print(
            "Warning: Could not attach "
            "3D models to analysis result: "
            f"{exc}"
        )

        if (
            "3d_reconstruction"
            not in result
        ):

            result[
                "3d_reconstruction"
            ] = {}

        result[
            "3d_reconstruction"
        ][
            "individual_models"
        ] = []

        result[
            "3d_reconstruction"
        ][
            "model_count"
        ] = 0

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return result


# ============================================================
# ANALYZE INDIVIDUAL MRI SLICES
# ============================================================

@app.post("/analyze-images")
async def analyze_images(
    files: List[UploadFile] = File(...)
):

    """
    Individual MRI slice endpoint.

    The current project pipeline operates on
    3D NIfTI MRI volumes.
    """

    if not files:

        raise HTTPException(
            status_code=400,

            detail=
                "No MRI slice files were selected."
        )

    raise HTTPException(
        status_code=501,

        detail=(
            "Individual MRI slice processing is not "
            "implemented in the current pipeline. "
            "Please upload a 3D .nii or .nii.gz MRI volume."
        )
    )


# ============================================================
# GET COMPLETE PATIENT RESULT
# ============================================================

@app.get("/result/{patient_id}")
def get_result(
    patient_id: str
):

    """
    Return the saved JSON analysis result.
    """

    patient_dir = (
        get_patient_output_dir(
            patient_id
        )
    )

    if not patient_dir.exists():

        raise HTTPException(
            status_code=404,

            detail=
                "Patient result not found."
        )

    result_file = (
        patient_dir /
        "result.json"
    )

    if not result_file.exists():

        json_files = list(
            patient_dir.glob("*.json")
        )

        if not json_files:

            raise HTTPException(
                status_code=404,

                detail=(
                    "No analysis result was "
                    "found for this patient."
                )
            )

        result_file = json_files[0]

    try:

        with result_file.open(
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception as exc:

        raise HTTPException(
            status_code=500,

            detail=(
                "Unable to read analysis result: "
                f"{exc}"
            )
        )


# ============================================================
# SERVE SEGMENTATION
# ============================================================

@app.get("/segmentation/{patient_id}")
def get_segmentation(
    patient_id: str
):

    """
    Return the generated segmentation file.
    """

    patient_dir = (
        get_patient_output_dir(
            patient_id
        )
    )

    if not patient_dir.exists():

        raise HTTPException(
            status_code=404,

            detail=(
                "Patient output directory "
                "not found."
            )
        )

    # --------------------------------------------------------
    # NIFTI
    # --------------------------------------------------------

    nii_files = list(
        patient_dir.rglob("*.nii")
    )

    nii_gz_files = list(
        patient_dir.rglob("*.nii.gz")
    )

    nii_candidates = (
        nii_files +
        nii_gz_files
    )

    if nii_candidates:

        preferred = [

            path

            for path in nii_candidates

            if (
                "predicted_segmentation"
                in path.name.lower()
            )
        ]

        segmentation_file = (
            preferred[0]
            if preferred
            else nii_candidates[0]
        )

        return FileResponse(

            path=str(
                segmentation_file
            ),

            media_type=
                "application/octet-stream",

            filename=
                segmentation_file.name
        )

    # --------------------------------------------------------
    # NPY FALLBACK
    # --------------------------------------------------------

    npy_files = list(
        patient_dir.rglob(
            "segmentation.npy"
        )
    )

    if npy_files:

        segmentation_file = (
            npy_files[0]
        )

        return FileResponse(

            path=str(
                segmentation_file
            ),

            media_type=
                "application/octet-stream",

            filename=
                segmentation_file.name
        )

    raise HTTPException(
        status_code=404,

        detail=
            "Segmentation file not found."
    )


# ============================================================
# SERVE INDIVIDUAL 3D MODEL
# ============================================================

@app.get("/models/{patient_id}/{filename}")
def get_model(
    patient_id: str,
    filename: str
):

    """
    Serve an actual generated 3D cardiac model.

    Supported formats:

        STL
        PLY
        OBJ
        GLB
        GLTF
    """

    # --------------------------------------------------------
    # BASIC FILENAME SECURITY
    # --------------------------------------------------------

    safe_filename = Path(
        filename
    ).name

    if safe_filename != filename:

        raise HTTPException(
            status_code=400,

            detail=
                "Invalid model filename."
        )

    # --------------------------------------------------------
    # FIND MODEL
    # --------------------------------------------------------

    model_path = find_model_file(
        patient_id,
        safe_filename
    )

    if model_path is None:

        raise HTTPException(
            status_code=404,

            detail=(
                f"3D model not found: "
                f"{safe_filename}"
            )
        )

    # --------------------------------------------------------
    # MEDIA TYPE
    # --------------------------------------------------------

    extension = (
        model_path
        .suffix
        .lower()
    )

    media_types = {

        ".stl":
            "application/octet-stream",

        ".ply":
            "application/octet-stream",

        ".obj":
            "text/plain",

        ".glb":
            "model/gltf-binary",

        ".gltf":
            "model/gltf+json",
    }

    # --------------------------------------------------------
    # RETURN ACTUAL FILE
    # --------------------------------------------------------

    return FileResponse(

        path=str(
            model_path
        ),

        media_type=
            media_types.get(
                extension,
                "application/octet-stream"
            ),

        filename=
            model_path.name
    )


# ============================================================
# LIST GENERATED 3D CARDIAC MODELS
# ============================================================

@app.get("/models/{patient_id}")
def list_models(
    patient_id: str
):

    """
    Return the individual cardiac structures
    generated for the patient.

    Combined heart files are excluded.
    """

    safe_patient_id = Path(
        patient_id
    ).name

    patient_dir = (
        OUTPUTS_DIR /
        safe_patient_id
    )

    if not patient_dir.exists():

        raise HTTPException(
            status_code=404,

            detail=(
                "Patient output directory "
                "not found."
            )
        )

    allowed_extensions = {
        ".stl",
        ".ply",
        ".obj",
        ".glb",
        ".gltf",
    }

    model_files = []

    # --------------------------------------------------------
    # FIND ALL MODEL FILES
    # --------------------------------------------------------

    for path in patient_dir.rglob("*"):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in allowed_extensions
        ):
            continue

        structure = identify_structure(
            path.name
        )

        if structure is None:
            continue

        model_files.append(
            (
                path,
                structure
            )
        )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    model_files.sort(
        key=lambda item:
            STRUCTURE_ORDER.get(
                item[1],
                99
            )
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    models = []

    added_structures = set()

    for path, structure in model_files:

        if structure in added_structures:
            continue

        added_structures.add(
            structure
        )

        models.append({

            "filename":
                path.name,

            "structure":
                structure,

            "url":
                (
                    f"/models/"
                    f"{safe_patient_id}/"
                    f"{path.name}"
                ),
        })

    return {

        "patient_id":
            safe_patient_id,

        "count":
            len(models),

        "models":
            models,
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "backend.app:app",

        host="127.0.0.1",

        port=8000,

        reload=True
    )