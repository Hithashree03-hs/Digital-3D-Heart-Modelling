from pathlib import Path
from typing import List
import json
import shutil
import uuid
import re
import traceback

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.pipeline import analyze_patient


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent

UPLOADS_DIR = BACKEND_DIR / "uploads"
OUTPUTS_DIR = BACKEND_DIR / "outputs"

# Local HVSMR dataset used by the patient selector.
# Expected MRI files include:
# pat0_cropped.nii.gz ... pat59_cropped.nii.gz
PROJECT_ROOT = BACKEND_DIR.parent
HVSMR_DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "HVSMR"
    / "cropped"
    / "cropped"
)

UPLOADS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Digital 3D Heart Analysis API",
    description=(
        "Backend API for Digital 3D Heart Modelling "
        "for Cardiac Structure and Abnormality Analysis"
    ),
    version="2.0.0",
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
# ALLOWED 3D MODEL TYPES
# ============================================================

ALLOWED_MODEL_EXTENSIONS = {
    ".stl",
    ".ply",
    ".obj",
    ".glb",
    ".gltf",
}


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "Digital 3D Heart Analysis API",
        "version": "2.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "Digital 3D Heart Analysis API",
    }


# ============================================================
# SANITIZE PATIENT ID
# ============================================================

def sanitize_patient_id(
    patient_id: str
) -> str:

    patient_id = str(
        patient_id or ""
    ).strip()

    patient_id = Path(
        patient_id
    ).name

    patient_id = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        patient_id
    )

    patient_id = patient_id.strip("_")

    if not patient_id:
        patient_id = (
            f"patient_"
            f"{uuid.uuid4().hex[:8]}"
        )

    return patient_id


# ============================================================
# EXTRACT DATASET PATIENT ID FROM FILE NAME
# ============================================================
#
# Examples:
#
# pat9_cropped.nii.gz
#       -> pat9
#
# pat9.nii.gz
#       -> pat9
#
# patient_scan.nii.gz
#       -> generated patient ID
#
# ============================================================

def extract_dataset_patient_id(
    filename: str
):

    name = Path(
        filename
    ).name.lower()

    # Remove NIfTI extensions
    if name.endswith(".nii.gz"):
        name = name[:-7]

    elif name.endswith(".nii"):
        name = name[:-4]

    # Look specifically for HVSMR style IDs.
    #
    # pat0
    # pat1
    # pat9
    # pat59
    #
    match = re.search(
        r"(pat\d+)",
        name,
        re.IGNORECASE
    )

    if match:

        return match.group(1).lower()

    return None


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

    with destination.open(
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            upload_file.file,
            buffer
        )


# ============================================================
# GET PATIENT OUTPUT DIRECTORY
# ============================================================

def get_patient_output_dir(
    patient_id: str
) -> Path:

    safe_patient_id = sanitize_patient_id(
        patient_id
    )

    return (
        OUTPUTS_DIR
        / safe_patient_id
    )


# ============================================================
# IDENTIFY CARDIAC STRUCTURE
# ============================================================

def identify_structure(
    filename: str
):

    name = (
        filename
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    # Specific structures first.

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
# FIND GENERATED 3D MODELS
# ============================================================

def collect_patient_models(
    patient_id: str
):

    patient_dir = (
        get_patient_output_dir(
            patient_id
        )
    )

    if not patient_dir.exists():
        return []

    generated_models = []

    for model_path in patient_dir.rglob("*"):

        if not model_path.is_file():
            continue

        extension = (
            model_path.suffix.lower()
        )

        if extension not in ALLOWED_MODEL_EXTENSIONS:
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


    # --------------------------------------------------------
    # REMOVE DUPLICATE STRUCTURES
    # --------------------------------------------------------

    unique_models = []

    seen_structures = set()

    for model in generated_models:

        structure = model[
            "structure"
        ]

        if structure in seen_structures:
            continue

        seen_structures.add(
            structure
        )

        unique_models.append(
            model
        )


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    unique_models.sort(
        key=lambda model:
            STRUCTURE_ORDER.get(
                model["structure"],
                99
            )
    )

    return unique_models


# ============================================================
# FIND RESULT PATIENT DIRECTORY
# ============================================================

def find_existing_patient_output(
    patient_id: str
):

    patient_dir = (
        get_patient_output_dir(
            patient_id
        )
    )

    if patient_dir.exists():
        return patient_dir

    return None


# ============================================================
# DATASET PATIENT SELECTION
# ============================================================

class PatientSelectionRequest(BaseModel):
    patient_id: str


def normalize_dataset_patient_id(patient_id: str) -> str:
    """
    Normalize:
        15 -> pat15
        pat15 -> pat15
        PAT15 -> pat15
    """
    value = str(patient_id or "").strip().lower()

    match = re.fullmatch(r"pat(\d+)", value)
    if match:
        return f"pat{int(match.group(1))}"

    match = re.fullmatch(r"(\d+)", value)
    if match:
        return f"pat{int(match.group(1))}"

    raise HTTPException(
        status_code=400,
        detail=(
            "Invalid patient ID. Use an HVSMR ID such as "
            "pat0, pat15, or pat59."
        ),
    )


def find_dataset_mri(patient_id: str) -> Path:
    """Find the MRI volume for the selected HVSMR patient."""
    patient_id = normalize_dataset_patient_id(patient_id)

    if not HVSMR_DATASET_DIR.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "HVSMR dataset directory was not found: "
                f"{HVSMR_DATASET_DIR}"
            ),
        )

    preferred_names = [
        f"{patient_id}_cropped.nii.gz",
        f"{patient_id}_cropped.nii",
        f"{patient_id}.nii.gz",
        f"{patient_id}.nii",
    ]

    preferred_lower = {name.lower() for name in preferred_names}

    for name in preferred_names:
        candidate = HVSMR_DATASET_DIR / name
        if candidate.is_file():
            return candidate

    for candidate in sorted(HVSMR_DATASET_DIR.rglob("*")):
        if (
            candidate.is_file()
            and candidate.name.lower() in preferred_lower
        ):
            return candidate

    raise HTTPException(
        status_code=404,
        detail=(
            f"MRI volume for {patient_id} was not found in "
            f"{HVSMR_DATASET_DIR}."
        ),
    )


def list_dataset_patients():
    """
    Discover patients from the dataset itself.
    No pat7/pat9 hardcoding.
    """
    if not HVSMR_DATASET_DIR.exists():
        return []

    patient_ids = set()

    for path in HVSMR_DATASET_DIR.rglob("*.nii*"):
        name = path.name.lower()

        # Only use MRI volumes, not segmentation files.
        if not (
            name.endswith("_cropped.nii.gz")
            or name.endswith("_cropped.nii")
        ):
            continue

        patient_id = extract_dataset_patient_id(path.name)
        if patient_id:
            patient_ids.add(patient_id)

    def patient_number(value):
        match = re.search(r"pat(\d+)$", value)
        return int(match.group(1)) if match else 10**9

    return sorted(patient_ids, key=patient_number)


@app.get("/patients")
def get_available_patients():
    """Return all HVSMR MRI patients actually present in the dataset."""
    patients = list_dataset_patients()

    if not patients:
        raise HTTPException(
            status_code=404,
            detail=(
                "No HVSMR patient MRI volumes were found in "
                f"{HVSMR_DATASET_DIR}."
            ),
        )

    return {
        "count": len(patients),
        "patients": patients,
    }


@app.post("/analyze-patient")
async def analyze_selected_patient(
    request: PatientSelectionRequest,
):
    """
    Analyze an HVSMR patient selected by ID.

    Example:
        {"patient_id": "pat15"}
    """
    patient_id = normalize_dataset_patient_id(
        request.patient_id
    )

    input_path = find_dataset_mri(patient_id)

    print()
    print("=" * 70)
    print("DATASET PATIENT ANALYSIS")
    print("=" * 70)
    print(f"Selected patient : {patient_id}")
    print(f"MRI volume       : {input_path}")
    print("=" * 70)

    try:
        result = analyze_patient(
            str(input_path),
            patient_id=patient_id,
        )
    except Exception as exc:
        print()
        print("Dataset patient analysis failed:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Analysis failed for {patient_id}: "
                f"{str(exc)}"
            ),
        )

    if result is None or not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail="Analysis completed but returned an invalid result.",
        )

    result["patient_id"] = patient_id
    result["input_mri"] = input_path.name

    generated_models = collect_patient_models(patient_id)

    if "3d_reconstruction" not in result:
        result["3d_reconstruction"] = {}

    result["3d_reconstruction"]["individual_models"] = generated_models
    result["3d_reconstruction"]["model_count"] = len(generated_models)

    result["api"] = {
        "patient_id": patient_id,
        "result_url": f"/result/{patient_id}",
        "segmentation_url": f"/segmentation/{patient_id}",
        "models_url": f"/models/{patient_id}",
    }

    print()
    print("=" * 70)
    print(f"PATIENT {patient_id} COMPLETE")
    print(f"3D models found: {len(generated_models)}/8")
    print("=" * 70)

    return result


# ============================================================
# ANALYZE NIFTI MRI
# ============================================================

@app.post("/analyze")
async def analyze_mri(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # VALIDATE FILE
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No MRI file was selected."
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
    # IMPORTANT PATIENT ID FIX
    # --------------------------------------------------------
    #
    # If user uploads:
    #
    #     pat9_cropped.nii.gz
    #
    # we use:
    #
    #     pat9
    #
    # instead of:
    #
    #     patient_d0796aff
    #
    # This keeps:
    #
    # MRI
    #   ↓
    # pipeline
    #   ↓
    # outputs/pat9
    #   ↓
    # API
    #   ↓
    # React
    #
    # synchronized.
    # --------------------------------------------------------

    dataset_patient_id = (
        extract_dataset_patient_id(
            filename
        )
    )


    if dataset_patient_id:

        patient_id = (
            dataset_patient_id
        )

    else:

        patient_id = (
            f"patient_"
            f"{uuid.uuid4().hex[:8]}"
        )


    patient_id = sanitize_patient_id(
        patient_id
    )


    print()
    print("=" * 70)
    print("MRI UPLOAD")
    print("=" * 70)

    print(
        f"Original filename : {filename}"
    )

    print(
        f"Patient ID        : {patient_id}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # CREATE UPLOAD DIRECTORY
    # --------------------------------------------------------

    patient_upload_dir = (
        UPLOADS_DIR
        / patient_id
    )

    patient_upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    input_path = (
        patient_upload_dir
        / filename
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
    # RUN COMPLETE PIPELINE
    # --------------------------------------------------------

    try:

        print()
        print("=" * 70)
        print(
            f"STARTING PIPELINE FOR: "
            f"{patient_id}"
        )
        print("=" * 70)

        result = analyze_patient(
            str(input_path),
            patient_id=patient_id
        )

    except Exception as exc:

        print()
        print(
            "MRI analysis failed:"
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "MRI analysis failed: "
                f"{str(exc)}"
            )
        )


    # --------------------------------------------------------
    # VALIDATE PIPELINE RESULT
    # --------------------------------------------------------

    if result is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Analysis completed but "
                "returned no result."
            )
        )


    if not isinstance(
        result,
        dict
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "Analysis returned an "
                "invalid result format."
            )
        )


    # --------------------------------------------------------
    # FIND GENERATED OUTPUT
    # --------------------------------------------------------

    patient_output_dir = (
        get_patient_output_dir(
            patient_id
        )
    )


    # --------------------------------------------------------
    # IMPORTANT FALLBACK
    # --------------------------------------------------------
    #
    # If an older pipeline still created the
    # output under another ID, try to identify
    # the output directory from result paths.
    # --------------------------------------------------------

    if not patient_output_dir.exists():

        print(
            "WARNING: Expected patient "
            "directory was not found:"
        )

        print(
            patient_output_dir
        )

        # Search for a result.json whose
        # parent directory contains models.

        possible_dirs = []

        for candidate in (
            OUTPUTS_DIR.glob("*")
        ):

            if not candidate.is_dir():
                continue

            result_file = (
                candidate
                / "result.json"
            )

            models_dir = (
                candidate
                / "3d_models"
            )

            if (
                result_file.exists()
                and
                models_dir.exists()
            ):

                possible_dirs.append(
                    candidate
                )


        # If exactly one valid candidate
        # exists, use it.

        if len(
            possible_dirs
        ) == 1:

            patient_output_dir = (
                possible_dirs[0]
            )

            patient_id = (
                patient_output_dir.name
            )

            print(
                "Recovered actual "
                "pipeline patient ID:"
            )

            print(
                patient_id
            )


    # --------------------------------------------------------
    # COLLECT 3D MODELS
    # --------------------------------------------------------

    generated_models = (
        collect_patient_models(
            patient_id
        )
    )


    # --------------------------------------------------------
    # CREATE / UPDATE RECONSTRUCTION
    # --------------------------------------------------------

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
    ] = generated_models


    result[
        "3d_reconstruction"
    ][
        "model_count"
    ] = len(
        generated_models
    )


    # --------------------------------------------------------
    # API URLS
    # --------------------------------------------------------

    result["api"] = {

        "patient_id":
            patient_id,

        "result_url":
            f"/result/{patient_id}",

        "segmentation_url":
            f"/segmentation/{patient_id}",

        "models_url":
            f"/models/{patient_id}",
    }


    # --------------------------------------------------------
    # ALSO STORE PATIENT ID DIRECTLY
    # --------------------------------------------------------

    result[
        "patient_id"
    ] = patient_id


    # --------------------------------------------------------
    # DEBUG OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("3D MODEL CHECK")
    print("=" * 70)

    print(
        f"Patient ID: {patient_id}"
    )

    print(
        f"Output directory:"
    )

    print(
        patient_output_dir
    )

    print(
        f"3D models found for "
        f"{patient_id}: "
        f"{len(generated_models)}"
    )


    for model in generated_models:

        print(
            "  - "
            f"{model['structure']}: "
            f"{model['filename']}"
        )


    print("=" * 70)


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return result


# ============================================================
# ANALYZE IMAGE SLICES
# ============================================================

@app.post("/analyze-images")
async def analyze_images(
    files: List[UploadFile] = File(...)
):

    if not files:

        raise HTTPException(
            status_code=400,
            detail=(
                "No MRI image slices "
                "were selected."
            )
        )


    raise HTTPException(
        status_code=501,
        detail=(
            "Individual MRI slice processing "
            "is not implemented in the current "
            "pipeline. Please upload a 3D "
            ".nii or .nii.gz MRI volume."
        )
    )


# ============================================================
# GET COMPLETE RESULT
# ============================================================

@app.get(
    "/result/{patient_id}"
)
def get_result(
    patient_id: str
):

    patient_id = (
        sanitize_patient_id(
            patient_id
        )
    )

    patient_dir = (
        get_patient_output_dir(
            patient_id
        )
    )

    if not patient_dir.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                f"Patient output "
                f"not found: {patient_id}"
            )
        )


    result_file = (
        patient_dir
        / "result.json"
    )


    if not result_file.exists():

        json_files = list(
            patient_dir.glob(
                "*.json"
            )
        )

        if not json_files:

            raise HTTPException(
                status_code=404,
                detail=(
                    "No analysis "
                    "result was found."
                )
            )

        result_file = (
            json_files[0]
        )


    try:

        with result_file.open(
            "r",
            encoding="utf-8"
        ) as file:

            result = json.load(
                file
            )


        # Ensure patient ID is available
        # even if older result.json does not
        # contain it.

        if isinstance(
            result,
            dict
        ):

            result[
                "patient_id"
            ] = patient_id


        return result


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to read "
                "analysis result: "
                f"{exc}"
            )
        )


# ============================================================
# GET SEGMENTATION
# ============================================================

@app.get(
    "/segmentation/{patient_id}"
)
def get_segmentation(
    patient_id: str
):

    patient_id = (
        sanitize_patient_id(
            patient_id
        )
    )


    patient_dir = (
        get_patient_output_dir(
            patient_id
        )
    )


    if not patient_dir.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Patient output "
                "directory not found."
            )
        )


    # --------------------------------------------------------
    # NIFTI
    # --------------------------------------------------------

    nii_files = list(
        patient_dir.rglob(
            "*.nii"
        )
    )

    nii_gz_files = list(
        patient_dir.rglob(
            "*.nii.gz"
        )
    )


    candidates = (
        nii_files
        +
        nii_gz_files
    )


    if candidates:

        preferred = [

            path

            for path in candidates

            if (
                "predicted_segmentation"
                in path.name.lower()
            )
        ]


        segmentation_file = (

            preferred[0]

            if preferred

            else candidates[0]
        )


        return FileResponse(

            path=str(
                segmentation_file
            ),

            media_type=(
                "application/octet-stream"
            ),

            filename=(
                segmentation_file.name
            )
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

            media_type=(
                "application/octet-stream"
            ),

            filename="segmentation.npy"
        )


    raise HTTPException(
        status_code=404,
        detail=(
            "Segmentation file "
            "not found."
        )
    )


# ============================================================
# LIST 3D MODELS
# ============================================================

@app.get(
    "/models/{patient_id}"
)
def list_models(
    patient_id: str
):

    patient_id = (
        sanitize_patient_id(
            patient_id
        )
    )


    models = (
        collect_patient_models(
            patient_id
        )
    )


    if not models:

        raise HTTPException(
            status_code=404,
            detail=(
                "No 3D models found "
                f"for patient {patient_id}."
            )
        )


    return {

        "patient_id":
            patient_id,

        "model_count":
            len(models),

        "models":
            models,
    }


# ============================================================
# SERVE INDIVIDUAL 3D MODEL
# ============================================================

@app.get(
    "/models/{patient_id}/{filename}"
)
def get_model(
    patient_id: str,
    filename: str
):

    patient_id = (
        sanitize_patient_id(
            patient_id
        )
    )


    safe_filename = Path(
        filename
    ).name


    models_dir = (
        get_patient_output_dir(
            patient_id
        )
        / "3d_models"
    )


    requested_file = (
        models_dir
        / safe_filename
    )


    # --------------------------------------------------------
    # SECURITY CHECK
    # --------------------------------------------------------

    try:

        requested_file.resolve().relative_to(
            models_dir.resolve()
        )

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail="Invalid file path."
        )


    # --------------------------------------------------------
    # FILE EXISTS
    # --------------------------------------------------------

    if not requested_file.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "3D model not found: "
                f"{safe_filename}"
            )
        )


    # --------------------------------------------------------
    # EXTENSION
    # --------------------------------------------------------

    extension = (
        requested_file
        .suffix
        .lower()
    )


    if extension not in (
        ALLOWED_MODEL_EXTENSIONS
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported 3D "
                "model format."
            )
        )


    # --------------------------------------------------------
    # MEDIA TYPE
    # --------------------------------------------------------

    if extension == ".stl":

        media_type = (
            "application/sla"
        )

    elif extension == ".ply":

        media_type = (
            "application/octet-stream"
        )

    elif extension == ".obj":

        media_type = (
            "text/plain"
        )

    elif extension == ".glb":

        media_type = (
            "model/gltf-binary"
        )

    elif extension == ".gltf":

        media_type = (
            "model/gltf+json"
        )

    else:

        media_type = (
            "application/octet-stream"
        )


    return FileResponse(

        path=str(
            requested_file
        ),

        media_type=media_type,

        filename=(
            requested_file.name
        )
    )


# ============================================================
# DEBUG: CHECK PATIENT MODELS
# ============================================================

@app.get(
    "/debug/models/{patient_id}"
)
def debug_models(
    patient_id: str
):

    patient_id = (
        sanitize_patient_id(
            patient_id
        )
    )


    patient_dir = (
        get_patient_output_dir(
            patient_id
        )
    )


    models_dir = (
        patient_dir
        / "3d_models"
    )


    models = []


    if models_dir.exists():

        for path in sorted(
            models_dir.iterdir()
        ):

            if not path.is_file():
                continue

            models.append({
                "filename":
                    path.name,

                "extension":
                    path.suffix.lower(),

                "exists":
                    path.exists(),

                "url":
                    (
                        f"/models/"
                        f"{patient_id}/"
                        f"{path.name}"
                    ),
            })


    return {

        "patient_id":
            patient_id,

        "patient_directory":
            str(patient_dir),

        "patient_directory_exists":
            patient_dir.exists(),

        "models_directory":
            str(models_dir),

        "models_directory_exists":
            models_dir.exists(),

        "model_count":
            len(models),

        "models":
            models,
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )