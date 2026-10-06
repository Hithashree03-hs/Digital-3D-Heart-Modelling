import React, {
  useEffect,
  useMemo,
  useState,
} from "react";

import Heart3DViewer from "./Heart3DViewer";

import "./App.css";
import "./digitalHeartTheme.css";

const API_BASE = "https://digital-3d-heart-modelling.onrender.com";

const STRUCTURE_NAMES = [
  "Left Ventricle",
  "Right Ventricle",
  "Left Atrium",
  "Right Atrium",
  "Aorta",
  "Pulmonary Artery",
  "Superior Vena Cava",
  "Inferior Vena Cava",
];

const STRUCTURE_KEYS = {
  "Left Ventricle": "left_ventricle",
  "Right Ventricle": "right_ventricle",
  "Left Atrium": "left_atrium",
  "Right Atrium": "right_atrium",
  Aorta: "aorta",
  "Pulmonary Artery": "pulmonary_artery",
  "Superior Vena Cava": "superior_vena_cava",
  "Inferior Vena Cava": "inferior_vena_cava",
};

function getFileNameFromPath(path) {
  if (!path) return "";

  const normalized = String(path).replaceAll("\\", "/");

  return normalized.split("/").pop();
}

function getModelUrl(patientId, model) {
  if (!model) return "";

  if (typeof model === "string") {
    const filename = getFileNameFromPath(model);

    if (!filename) return "";

    return `${API_BASE}/models/${encodeURIComponent(
      patientId
    )}/${encodeURIComponent(filename)}`;
  }

  if (model.url) {
    return model.url.startsWith("http")
      ? model.url
      : `${API_BASE}${model.url}`;
  }

  if (model.file_url) {
    return model.file_url.startsWith("http")
      ? model.file_url
      : `${API_BASE}${model.file_url}`;
  }

  if (model.path) {
    const filename = getFileNameFromPath(model.path);

    if (!filename) return "";

    return `${API_BASE}/models/${encodeURIComponent(
      patientId
    )}/${encodeURIComponent(filename)}`;
  }

  if (model.filename) {
    return `${API_BASE}/models/${encodeURIComponent(
      patientId
    )}/${encodeURIComponent(model.filename)}`;
  }

  return "";
}

function formatNumber(value, digits = 2) {
  if (
    value === null ||
    value === undefined ||
    value === ""
  ) {
    return "—";
  }

  const number = Number(value);

  if (Number.isNaN(number)) {
    return String(value);
  }

  return number.toFixed(digits);
}

function getFeatureName(feature) {
  return (
    feature?.parameter ||
    feature?.name ||
    feature?.feature_name ||
    feature?.feature ||
    feature?.label ||
    ""
  );
}

/*
 * Backend abnormality_analysis.py returns
 * the patient measurement as "patient_value".
 */
function getFeatureValue(feature) {
  return (
    feature?.patient_value ??
    feature?.measured ??
    feature?.value ??
    feature?.measurement ??
    feature?.measured_value ??
    null
  );
}

/*
 * Backend returns:
 * reference_lower
 * reference_upper
 *
 * Convert them into an object that the
 * formatting function can display.
 */
function getFeatureReference(feature) {
  if (
    feature?.reference_lower !== undefined &&
    feature?.reference_upper !== undefined
  ) {
    return {
      min: feature.reference_lower,
      max: feature.reference_upper,
    };
  }

  return (
    feature?.reference ??
    feature?.reference_range ??
    feature?.normal_range ??
    null
  );
}

/*
 * Backend returns "severity":
 * Normal range
 * Mild deviation
 * Marked deviation
 */
function getFeatureAssessment(feature) {
  return (
    feature?.severity ||
    feature?.assessment ||
    feature?.status ||
    feature?.classification ||
    feature?.deviation ||
    "Assessment unavailable"
  );
}

function getNumericFeatureValue(feature) {
  const value =
    feature?.patient_value ??
    feature?.measured ??
    feature?.value ??
    feature?.measurement ??
    feature?.measured_value ??
    feature?.measuredValue ??
    feature?.actual ??
    null;

  if (
    value !== null &&
    value !== undefined &&
    value !== ""
  ) {
    return value;
  }

  if (feature?.parameter_value !== undefined) {
    return feature.parameter_value;
  }

  return null;
}

/*
 * Display the actual reference interval
 * calculated by the backend.
 *
 * Example:
 * 62.40 – 128.70
 */
function formatReference(reference, feature) {
  if (
    reference &&
    reference.min !== undefined &&
    reference.max !== undefined
  ) {
    return `${formatNumber(
      reference.min
    )} – ${formatNumber(reference.max)}`;
  }

  return "—";
}

function normalizeFeatureText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
}

function findMeasuredFromCardiacRows(
  feature,
  rows
) {
  const featureName =
    normalizeFeatureText(
      getFeatureName(feature)
    );

  if (!featureName) {
    return null;
  }

  const matches = [
    ["leftventriclevolume", 0, "volume"],
    ["rightventriclevolume", 1, "volume"],
    ["leftatriumvolume", 2, "volume"],
    ["rightatriumvolume", 3, "volume"],
    ["aortavolume", 4, "volume"],
    ["pulmonaryarteryvolume", 5, "volume"],
    ["superiorvenacavavolume", 6, "volume"],
    ["inferiorvenacavavolume", 7, "volume"],

    ["leftventriclelength", 0, "length"],
    ["leftventriclewidth", 0, "width"],
    ["leftventricledepth", 0, "depth"],

    ["rightventriclelength", 1, "length"],
    ["rightventriclewidth", 1, "width"],
    ["rightventricledepth", 1, "depth"],

    ["leftatriumlength", 2, "length"],
    ["leftatriumwidth", 2, "width"],
    ["leftatriumdepth", 2, "depth"],

    ["rightatriumlength", 3, "length"],
    ["rightatriumwidth", 3, "width"],
    ["rightatriumdepth", 3, "depth"],
  ];

  for (const [
    key,
    rowIndex,
    field,
  ] of matches) {
    if (featureName.includes(key)) {
      return rows[rowIndex]?.[field] ?? null;
    }
  }

  return null;
}

export default function App() {
  // ---------------------------------------------------------
  // STATE
  // ---------------------------------------------------------

  const [
    inputType,
    setInputType,
  ] = useState("volume");

  const [
    selectedFiles,
    setSelectedFiles,
  ] = useState([]);

  const [
    loading,
    setLoading,
  ] = useState(false);

  const [
    error,
    setError,
  ] = useState("");

  const [
    result,
    setResult,
  ] = useState(null);

  // ---------------------------------------------------------
  // FILE SELECTION
  // ---------------------------------------------------------

  const handleFileChange = (
    event
  ) => {
    const files = Array.from(
      event.target.files || []
    );

    setSelectedFiles(files);
    setError("");
    setResult(null);
  };

  // ---------------------------------------------------------
  // ANALYZE MRI
  // ---------------------------------------------------------

  const handleAnalyze = async () => {
    if (selectedFiles.length === 0) {
      setError(
        inputType === "volume"
          ? "Please select an MRI volume first."
          : "Please select MRI slices first."
      );
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData =
        new FormData();

      if (inputType === "volume") {
        formData.append(
          "file",
          selectedFiles[0]
        );

        const response =
          await fetch(
            `${API_BASE}/analyze`,
            {
              method: "POST",
              body: formData,
            }
          );

        if (!response.ok) {
          const text =
            await response.text();

          throw new Error(
            text ||
              `Backend error: ${response.status}`
          );
        }

        const data =
          await response.json();

        setResult(data);
      } else {
        selectedFiles.forEach(
          (file) => {
            formData.append(
              "files",
              file
            );
          }
        );

        const response =
          await fetch(
            `${API_BASE}/analyze-images`,
            {
              method: "POST",
              body: formData,
            }
          );

        if (!response.ok) {
          const text =
            await response.text();

          throw new Error(
            text ||
              `Backend error: ${response.status}`
          );
        }

        const data =
          await response.json();

        setResult(data);
      }
    } catch (err) {
      console.error(err);

      if (
        err.message ===
        "Failed to fetch"
      ) {
        setError(
          "Failed to connect to the backend. Make sure the FastAPI server is running at https://digital-3d-heart-modelling.onrender.com."
        );
      } else {
        setError(
          err.message ||
            "Analysis failed."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  // ---------------------------------------------------------
  // EXTRACT BACKEND DATA
  // ---------------------------------------------------------

  const patientId =
    result?.patient_id ||
    result?.patientId ||
    "";

  const segmentation =
    result?.segmentation || {};

  const featureExtraction =
    result?.feature_extraction ||
    {};

  const abnormality =
    result?.abnormality_analysis ||
    {};

  /*
   * Septal defect screening added
   * by backend/services/septal_defect.py
   */
  const septalDefect =
    result?.septal_defect_screening ||
    null;

  const reconstruction =
    result?.["3d_reconstruction"] ||
    {};

  const limitations =
    result?.limitations ||
    [];

  // ---------------------------------------------------------
  // VOXEL SPACING
  // ---------------------------------------------------------

  const voxelSpacing =
    segmentation?.voxel_spacing_mm ||
    segmentation?.voxel_spacing ||
    result?.voxel_spacing_mm ||
    result?.voxel_spacing ||
    result?.preprocessing
      ?.voxel_spacing_mm ||
    result?.metadata
      ?.voxel_spacing_mm ||
    result?.input_mri
      ?.voxel_spacing_mm ||
    null;

  const voxelSpacingText =
    Array.isArray(voxelSpacing)
      ? voxelSpacing
          .map((v) =>
            Number(v).toFixed(3)
          )
          .join(" × ") + " mm"
      : "Unavailable";

  // ---------------------------------------------------------
  // VOLUME SHAPE
  // ---------------------------------------------------------

  const volumeShape =
    segmentation?.shape ||
    segmentation?.volume_shape ||
    segmentation?.volumeShape ||
    null;

  const volumeShapeText =
    Array.isArray(volumeShape)
      ? volumeShape.join(" × ")
      : volumeShape ||
        "Unavailable";

  // ---------------------------------------------------------
  // CARDIAC STRUCTURES
  // ---------------------------------------------------------

  const cardiacStructureCount =
    segmentation?.num_structures ||
    segmentation?.structures_count ||
    segmentation?.number_of_structures ||
    8;

  // ---------------------------------------------------------
  // CARDIAC PARAMETER TABLE
  // ---------------------------------------------------------

  const cardiacRows =
    STRUCTURE_NAMES.map(
      (name) => {
        const key =
          STRUCTURE_KEYS[name];

        const data =
          featureExtraction?.[key] ||
          featureExtraction?.[
            `${key}_features`
          ] ||
          {};

        return {
          name,

          volume:
            data.volume_ml ??
            data.volume ??
            featureExtraction?.[
              `${key}_volume_ml`
            ],

          length:
            data.length_mm ??
            data.length ??
            featureExtraction?.[
              `${key}_length_mm`
            ],

          width:
            data.width_mm ??
            data.width ??
            featureExtraction?.[
              `${key}_width_mm`
            ],

          depth:
            data.depth_mm ??
            data.depth ??
            featureExtraction?.[
              `${key}_depth_mm`
            ],
        };
      }
    );

  // ---------------------------------------------------------
  // LV / RV
  // ---------------------------------------------------------

  const lvVolume =
    featureExtraction
      ?.left_ventricle_volume_ml ??
    featureExtraction
      ?.left_ventricle
      ?.volume_ml ??
    cardiacRows[0]?.volume;

  const rvVolume =
    featureExtraction
      ?.right_ventricle_volume_ml ??
    featureExtraction
      ?.right_ventricle
      ?.volume_ml ??
    cardiacRows[1]?.volume;

  const lvRvRatio =
    featureExtraction
      ?.lv_rv_volume_ratio ??
    featureExtraction
      ?.lv_rv_ratio ??
    (lvVolume && rvVolume
      ? Number(lvVolume) /
        Number(rvVolume)
      : null);

  const lvRvDifference =
    featureExtraction
      ?.lv_rv_volume_difference_percent ??
    featureExtraction
      ?.lv_rv_difference_percent ??
    null;

  // ---------------------------------------------------------
  // ABNORMALITY FEATURES
  // ---------------------------------------------------------

  const abnormalityFeatures =
    Array.isArray(
      abnormality?.features
    )
      ? abnormality.features
      : [];

  const mildDeviations =
    abnormality
      ?.mild_deviations ??
    abnormality
      ?.mildDeviations ??
    0;

  const markedDeviations =
    abnormality
      ?.marked_deviations ??
    abnormality
      ?.markedDeviations ??
    0;

  const normalCount = Math.max(
    0,
    abnormalityFeatures.length -
      Number(mildDeviations) -
      Number(markedDeviations)
  );

  const backendOverallAssessment =
    abnormality
      ?.overall_assessment ||
    abnormality
      ?.overallAssessment ||
    "";

  const overallAssessment =
    backendOverallAssessment ||
    (
      Number(markedDeviations) > 0
        ? "Marked morphological deviation detected"
        : Number(mildDeviations) > 0
        ? "Mild morphological deviation detected"
        : "No morphological deviation detected"
    );

  // ---------------------------------------------------------
  // 3D MODELS
  // ---------------------------------------------------------

  const heartModels =
    useMemo(() => {
      const raw =
        reconstruction
          ?.individual_models ||
        reconstruction
          ?.individualModels ||
        reconstruction
          ?.models ||
        [];

      if (!Array.isArray(raw)) {
        return [];
      }

      return raw
        .map((model) => {
          const originalPath =
            typeof model === "string"
              ? model
              : model?.filename ||
                model?.url ||
                model?.model_url ||
                model?.path ||
                "";

          if (!originalPath) {
            return null;
          }

          const filename =
            getFileNameFromPath(
              originalPath
            );

          /*
           * Backend generates STL models.
           */
          const stlFilename =
            filename.replace(
              /\.(ply|obj|glb|gltf|stl)$/i,
              ""
            ) + ".stl";

          const lowerName =
            filename.toLowerCase();

          const structureName =
            STRUCTURE_NAMES.find(
              (name) =>
                lowerName.includes(
                  name
                    .toLowerCase()
                    .replaceAll(
                      " ",
                      "_"
                    )
                )
            ) ||
            (typeof model ===
            "object"
              ? model.name ||
                model.structure ||
                model.label
              : null) ||
            filename;

          const stlUrl =
            `${API_BASE}/models/${encodeURIComponent(
              patientId
            )}/${encodeURIComponent(
              stlFilename
            )}`;

          return {
            ...(typeof model ===
            "object"
              ? model
              : {}),

            name:
              structureName,

            filename:
              stlFilename,

            url:
              stlUrl,
          };
        })
        .filter(
          (model) =>
            model &&
            model.url
        );
    }, [
      reconstruction,
      patientId,
    ]);

  // ---------------------------------------------------------
  // MODEL COUNT
  // ---------------------------------------------------------

  const generatedModelCount =
    reconstruction
      ?.individual_models
      ?.length ||
    reconstruction
      ?.individualModels
      ?.length ||
    reconstruction
      ?.models
      ?.length ||
    heartModels.length;

  // ---------------------------------------------------------
  // RETURN
  // ---------------------------------------------------------

  return (
    <div className="app">

      {/* =====================================================
          HEADER
      ===================================================== */}

      <header className="site-header">

        <div className="header-left">

          <h1>
            Digital 3D Heart
          </h1>

          <p>
            Cardiac Structure and
            Abnormality Analysis
          </p>

        </div>

        <div className="backend-status">

          <span className="status-dot" />

          <div>
            <span>
              Backend
            </span>

            <strong>
              Connected
            </strong>
          </div>

        </div>

      </header>

      {/* =====================================================
          HERO / UPLOAD
      ===================================================== */}

      <main>

        
        <section className="dh-hero">
          <div className="dh-hero-glow dh-hero-glow-one" />
          <div className="dh-hero-glow dh-hero-glow-two" />

          <div className="dh-hero-content">
            <div className="dh-eyebrow">
              <span className="dh-pulse-dot" />
              PATIENT-SPECIFIC CARDIAC VISUALIZATION
            </div>

            <h1 className="dh-title">
              Digital <span>3D Heart</span>
            </h1>

            <p className="dh-subtitle">
              Interactive reconstruction and abnormality analysis from
              segmented cardiac MRI.
            </p>

            <div className="dh-hero-pills">
              <span>3D Reconstruction</span>
              <span>MRI Analysis</span>
              <span>Defect Screening</span>
            </div>
          </div>
        </section>

<section className="hero-section">

          <div className="section-number">
            01
          </div>

          <h2>
            Transform cardiac MRI into a
            patient-specific 3D heart model
          </h2>

          <p>
            Upload a cardiac MRI volume
            or individual MRI slices to
            perform segmentation,
            3D reconstruction and
            cardiac morphological analysis.
          </p>

        </section>

        <section className="upload-section">

          <div className="input-tabs">

            <button
              className={
                inputType === "volume"
                  ? "active"
                  : ""
              }
              onClick={() =>
                setInputType("volume")
              }
            >
              3D MRI Volume

              <span>
                NIfTI (.nii / .nii.gz)
              </span>
            </button>

            <button
              className={
                inputType === "slices"
                  ? "active"
                  : ""
              }
              onClick={() =>
                setInputType("slices")
              }
            >
              Individual MRI Slices

              <span>
                Image slices
              </span>
            </button>

          </div>

          <label className="upload-box">

            <input
              type="file"
              accept={
                inputType ===
                "volume"
                  ? ".nii,.nii.gz"
                  : "image/*"
              }
              multiple={
                inputType ===
                "slices"
              }
              onChange={
                handleFileChange
              }
            />

            <div className="upload-icon">
              +
            </div>

            <h3>
              {inputType ===
              "volume"
                ? "Select MRI Volume"
                : "Select MRI Slices"}
            </h3>

            <p>
              {inputType ===
              "volume"
                ? "Choose a .nii or .nii.gz cardiac MRI volume"
                : "Choose individual cardiac MRI image slices"}
            </p>

          </label>

          {selectedFiles.length >
            0 && (

            <div className="selected-files">

              <strong>
                Selected file
                {selectedFiles.length >
                1
                  ? "s"
                  : ""}
                :
              </strong>

              <div>
                {selectedFiles
                  .map(
                    (file) =>
                      file.name
                  )
                  .join(", ")}
              </div>

            </div>

          )}

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          <button
            className="analyze-button"
            onClick={
              handleAnalyze
            }
            disabled={loading}
          >
            {loading
              ? "Analyzing MRI..."
              : "Analyze MRI"}
          </button>

        </section>

        {/* =================================================
            RESULT
        ================================================= */}

        {result && (

          <div className="results">

            {/* =============================================
                ANALYSIS COMPLETE
            ============================================= */}

            <section className="complete-banner">

              <span>
                ANALYSIS COMPLETE
              </span>

            </section>

            <section className="patient-header">

              <div className="section-number">
                —
              </div>

              <h2>
                Patient-Specific
                Cardiac Analysis
              </h2>

              <p>
                Patient ID:{" "}
                <strong>
                  {patientId ||
                    "Unavailable"}
                </strong>
              </p>

            </section>

            {/* =============================================
                PIPELINE
            ============================================= */}

            <section className="content-section">

              <div className="section-number">
                02
              </div>

              <h2>
                Analysis Pipeline
              </h2>

              <p>
                Automated processing of
                the selected cardiac MRI.
              </p>

              <div className="pipeline-grid">

                <div className="pipeline-card">
                  <span>01</span>

                  <strong>
                    Preprocessing
                  </strong>

                  <small>
                    Completed
                  </small>
                </div>

                <div className="pipeline-card">
                  <span>02</span>

                  <strong>
                    U-Net Segmentation
                  </strong>

                  <small>
                    Completed
                  </small>
                </div>

                <div className="pipeline-card">
                  <span>03</span>

                  <strong>
                    3D Reconstruction
                  </strong>

                  <small>
                    Completed
                  </small>
                </div>

                <div className="pipeline-card">
                  <span>04</span>

                  <strong>
                    Cardiac Analysis
                  </strong>

                  <small>
                    Completed
                  </small>
                </div>

              </div>

            </section>

            {/* =============================================
                SEGMENTATION
            ============================================= */}

            <section className="content-section">

              <div className="section-number">
                03
              </div>

              <h2>
                Segmentation
              </h2>

              <p>
                Cardiac structures extracted
                from the MRI volume.
              </p>

              <div className="info-grid">

                <div className="info-card">

                  <span>
                    Volume Shape
                  </span>

                  <strong>
                    {volumeShapeText}
                  </strong>

                </div>

                <div className="info-card">

                  <span>
                    Voxel Spacing
                  </span>

                  <strong>
                    {voxelSpacingText}
                  </strong>

                </div>

                <div className="info-card">

                  <span>
                    Structures
                  </span>

                  <strong>
                    {cardiacStructureCount}
                    {" "}
                    Cardiac Structures
                  </strong>

                </div>

              </div>

            </section>

            {/* =============================================
                CARDIAC PARAMETERS
            ============================================= */}

            <section className="content-section">

              <div className="section-number">
                04
              </div>

              <h2>
                Cardiac Parameters
              </h2>

              <p>
                Quantitative measurements
                extracted from the reconstructed
                cardiac structures.
              </p>

              <div className="table-container">

                <table>

                  <thead>

                    <tr>

                      <th>
                        Structure
                      </th>

                      <th>
                        Volume (mL)
                      </th>

                      <th>
                        Length (mm)
                      </th>

                      <th>
                        Width (mm)
                      </th>

                      <th>
                        Depth (mm)
                      </th>

                    </tr>

                  </thead>

                  <tbody>

                    {cardiacRows.map(
                      (row) => (

                        <tr
                          key={
                            row.name
                          }
                        >

                          <td>
                            <strong>
                              {row.name}
                            </strong>
                          </td>

                          <td>
                            {formatNumber(
                              row.volume
                            )}
                          </td>

                          <td>
                            {formatNumber(
                              row.length
                            )}
                          </td>

                          <td>
                            {formatNumber(
                              row.width
                            )}
                          </td>

                          <td>
                            {formatNumber(
                              row.depth
                            )}
                          </td>

                        </tr>

                      )
                    )}

                  </tbody>

                </table>

              </div>

            </section>

            {/* =============================================
                VENTRICULAR COMPARISON
            ============================================= */}

            <section className="content-section">

              <div className="section-number">
                05
              </div>

              <h2>
                Ventricular Comparison
              </h2>

              <p>
                Comparison of left and
                right ventricular volumes.
              </p>

              <div className="comparison-grid">

                <div className="comparison-card">

                  <span>
                    Left Ventricle
                  </span>

                  <strong>
                    {formatNumber(
                      lvVolume
                    )}
                    {" "}
                    mL
                  </strong>

                </div>

                <div className="comparison-card">

                  <span>
                    Right Ventricle
                  </span>

                  <strong>
                    {formatNumber(
                      rvVolume
                    )}
                    {" "}
                    mL
                  </strong>

                </div>

                <div className="comparison-card">

                  <span>
                    LV / RV Volume Ratio
                  </span>

                  <strong>
                    {formatNumber(
                      lvRvRatio
                    )}
                  </strong>

                </div>

                <div className="comparison-card">

                  <span>
                    Volume Difference
                  </span>

                  <strong>
                    {formatNumber(
                      lvRvDifference
                    )}
                    %
                  </strong>

                </div>

              </div>

            </section>

            {/* =============================================
                SEPTAL DEFECT SCREENING
            ============================================= */}

            <section className="septal-defect-section">

              <div className="section-number">
                06
              </div>

              <h2>
                Septal Defect Screening
              </h2>

              <p>
                Automated MRI-based screening
                for ventricular and atrial
                septal defects.
              </p>

              {septalDefect ? (

                <div className="septal-defect-card">

                  <div className="septal-defect-grid">

                    <div className="septal-defect-item">

                      <span>
                        Ventricular Septal Defect
                      </span>

                      <strong>
                        {septalDefect?.vsd?.detected === true
                          ? "VSD defect suspected"
                          : septalDefect?.vsd?.detected === false
                          ? "No VSD defect"
                          : "Result unavailable"}
                      </strong>

                    </div>

                    <div className="septal-defect-item">

                      <span>
                        Atrial Septal Defect
                      </span>

                      <strong>
                        {septalDefect?.asd?.detected === true
                          ? "ASD defect suspected"
                          : septalDefect?.asd?.detected === false
                          ? "No ASD defect"
                          : "Result unavailable"}
                      </strong>

                    </div>

                  </div>

                  <div className="septal-defect-overall">

                    <span>
                      Overall Screening Result
                    </span>

                    <strong>
                      {
                        septalDefect
                          ?.overall ||
                        "Result unavailable"
                      }
                    </strong>

                  </div>

                  

                </div>

              ) : (

                <div className="septal-defect-card unavailable">

                  Septal defect screening
                  result is unavailable.

                </div>

              )}

            </section>

            {/* =============================================
                MORPHOLOGICAL ABNORMALITY ANALYSIS
            ============================================= */}

            <section className="abnormality-section">

              <div className="section-number">
                07
              </div>

              <h2>
                Morphological Abnormality Analysis
              </h2>

              <p>
                Automated screening based on
                extracted cardiac morphology.
              </p>

              <div className="abnormality-summary">

                <div className="abnormality-item normal">

                  <span className="status-circle" />

                  <strong>
                    Normal
                  </strong>

                  <b>
                    {normalCount}
                  </b>

                </div>

                <div className="abnormality-item mild">

                  <span className="status-circle" />

                  <strong>
                    Mild deviation
                  </strong>

                  <b>
                    {mildDeviations}
                  </b>

                </div>

                <div className="abnormality-item marked">

                  <span className="status-circle" />

                  <strong>
                    Marked deviation
                  </strong>

                  <b>
                    {markedDeviations}
                  </b>

                </div>

              </div>

              <div className="overall-assessment">

                <span>
                  Overall Assessment
                </span>

                <strong>
                  {overallAssessment}
                </strong>

              </div>

              {abnormalityFeatures.length >
                0 && (

                <div className="table-container abnormality-table">

                  <table>

                    <thead>

                      <tr>

                        <th>
                          Parameter
                        </th>

                        <th>
                          Measured
                        </th>

                        <th>
                          Reference
                        </th>

                        <th>
                          Assessment
                        </th>

                      </tr>

                    </thead>

                    <tbody>

                      {abnormalityFeatures.map(
                        (
                          feature,
                          index
                        ) => {

                          const name =
                            getFeatureName(
                              feature
                            );

                          const rawMeasured =
                            getNumericFeatureValue(
                              feature
                            );

                          const measured =
                            rawMeasured ??
                            findMeasuredFromCardiacRows(
                              feature,
                              cardiacRows
                            );

                          const reference =
                            getFeatureReference(
                              feature
                            );

                          const referenceText =
                            formatReference(
                              reference,
                              feature
                            );

                          const assessment =
                            getFeatureAssessment(
                              feature
                            );

                          const assessmentLower =
                            String(
                              assessment
                            ).toLowerCase();

                          return (
                            <tr
                              key={
                                index
                              }
                            >

                              <td>
                                <strong>
                                  {name ||
                                    `Cardiac parameter ${
                                      index +
                                      1
                                    }`}
                                </strong>
                              </td>

                              <td>
                                {typeof measured ===
                                "number"
                                  ? formatNumber(
                                      measured
                                    )
                                  : measured ??
                                    "—"}
                              </td>

                              <td>
                                {referenceText}
                              </td>

                              <td>

                                <span
                                  className={
                                    assessmentLower.includes(
                                      "marked"
                                    )
                                      ? "assessment marked"
                                      : assessmentLower.includes(
                                          "mild"
                                        )
                                      ? "assessment mild"
                                      : "assessment normal"
                                  }
                                >
                                  {assessment}
                                </span>

                              </td>

                            </tr>
                          );
                        }
                      )}

                    </tbody>

                  </table>

                </div>

              )}
</section>

            {/* =============================================
                3D HEART
            ============================================= */}

            <section className="heart-section">

              <div className="section-number">
                08
              </div>

              <h2>
                Patient-Specific 3D Heart Model
              </h2>

              <p>
                Interactive visualization of
                reconstructed cardiac structures.
              </p>

              <div className="model-count">

                <strong>
                  {generatedModelCount ||
                    heartModels.length}
                  /8
                </strong>

                {" "}
                3D cardiac structure
                models generated

              </div>

              <div className="model-status">

                {heartModels.length}/8
                {" "}
                3D models loaded

              </div>

              {abnormalityFeatures.length >
                0 && (

                <div className="heart-abnormality-status">

                  {Number(
                    markedDeviations
                  ) > 0 ? (

                    <>
                      <strong>
                        Marked morphological
                        deviation detected
                      </strong>

                      {" "}
                      in the current
                      screening.
                    </>

                  ) : Number(
                      mildDeviations
                    ) > 0 ? (

                    <>
                      <strong>
                        Mild morphological
                        deviation detected
                      </strong>

                      {" "}
                      in the current
                      screening.
                    </>

                  ) : (

                    <>
                      <strong>
                        No morphological
                        deviation detected
                      </strong>

                      {" "}
                      in the current
                      screening.
                    </>

                  )}

                </div>

              )}

              <Heart3DViewer
                models={
                  heartModels
                }
                patientId={
                  patientId
                }
                septalDefect={
                  septalDefect
                }
                reconstruction={
                  reconstruction
                }
              />

            </section>

          </div>

        )}

      </main>

      {/* =====================================================
          FOOTER
      ===================================================== */}

      <footer className="site-footer">

        <div>

          <strong>
            Digital 3D Heart
          </strong>

          <span>
            Patient-specific cardiac
            structure modelling and analysis
          </span>

        </div>

        <div>
          Cardiac Structure &
          Abnormality Analysis
        </div>

      </footer>

    </div>
  );
}
