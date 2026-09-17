import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const BACKEND_URL = "http://127.0.0.1:8000";

const STRUCTURE_COLORS = {
  "Left Ventricle": 0xff0000,
  "Right Ventricle": 0x1677ff,
  "Left Atrium": 0x00c853,
  "Right Atrium": 0xff9800,
  Aorta: 0x8a00ff,
  "Pulmonary Artery": 0x18c5d8,
  "Superior Vena Cava": 0xff00a8,
  "Inferior Vena Cava": 0xb0b0b0,
};

const STRUCTURE_ORDER = [
  "Left Ventricle",
  "Right Ventricle",
  "Left Atrium",
  "Right Atrium",
  "Aorta",
  "Pulmonary Artery",
  "Superior Vena Cava",
  "Inferior Vena Cava",
];

function getStructureName(model) {
  if (!model) {
    return "Unknown Structure";
  }

  if (typeof model === "string") {
    return model;
  }

  return (
    model.structure_name ||
    model.structure ||
    model.name ||
    model.label ||
    model.title ||
    "Unknown Structure"
  );
}

function getModelUrl(model) {
  if (!model) {
    return "";
  }

  if (typeof model === "string") {
    return model;
  }

  return (
    model.url ||
    model.model_url ||
    model.file_url ||
    model.path ||
    model.stl_url ||
    ""
  );
}

function makeMaterial(structureName) {
  const color =
    STRUCTURE_COLORS[structureName] !== undefined
      ? STRUCTURE_COLORS[structureName]
      : 0xffffff;

  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.58,
    metalness: 0.05,
    side: THREE.FrontSide,
  });
}

function normalizeUrl(url) {
  if (!url) {
    return "";
  }

  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }

  if (url.startsWith("/")) {
    return `${BACKEND_URL}${url}`;
  }

  return `${BACKEND_URL}/${url.replace(/^\/+/, "")}`;
}

function disposeObject(object) {
  object.traverse((child) => {
    if (child.geometry) {
      child.geometry.dispose();
    }

    if (child.material) {
      if (Array.isArray(child.material)) {
        child.material.forEach((material) => material.dispose());
      } else {
        child.material.dispose();
      }
    }
  });
}

export default function Heart3DViewer({
  models = [],
  patientId = "",
}) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const groupRef = useRef(null);
  const animationRef = useRef(null);

  const cameraDistanceRef = useRef(400);

  const [loading, setLoading] = useState(true);
  const [loadedModels, setLoadedModels] = useState([]);
  const [errors, setErrors] = useState([]);
  const [showLegend, setShowLegend] = useState(true);
  const [selectedStructure, setSelectedStructure] = useState("");

  /*
   * ------------------------------------------------------------
   * THREE.JS SCENE SETUP
   * ------------------------------------------------------------
   */
  useEffect(() => {
    const mount = mountRef.current;

    if (!mount) {
      return undefined;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x030711);

    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      45,
      1,
      0.1,
      10000
    );

    camera.position.set(300, 200, 350);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });

    renderer.setPixelRatio(
      Math.min(window.devicePixelRatio || 1, 2)
    );

    renderer.setSize(
      mount.clientWidth || 900,
      mount.clientHeight || 600,
      false
    );

    renderer.outputColorSpace = THREE.SRGBColorSpace;

    rendererRef.current = renderer;

    mount.appendChild(renderer.domElement);

    /*
     * Lighting
     */
    const ambientLight = new THREE.AmbientLight(
      0xffffff,
      1.8
    );

    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(
      0xffffff,
      2.4
    );

    keyLight.position.set(300, 400, 500);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(
      0xffffff,
      1.2
    );

    fillLight.position.set(-300, 100, 200);
    scene.add(fillLight);

    const backLight = new THREE.DirectionalLight(
      0xffffff,
      1.0
    );

    backLight.position.set(100, -200, -400);
    scene.add(backLight);

    /*
     * Group containing all eight cardiac structures.
     *
     * IMPORTANT:
     * We keep all STL meshes in their original common
     * coordinate system. We do NOT center each mesh.
     */
    const group = new THREE.Group();

    group.name = "PatientSpecificHeart";
    groupRef.current = group;

    scene.add(group);

    /*
     * Orbit controls:
     * Left drag  = rotate
     * Wheel      = zoom
     * Right drag = pan
     */
    const controls = new OrbitControls(
      camera,
      renderer.domElement
    );

    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    controls.enableZoom = true;
    controls.enablePan = true;
    controls.enableRotate = true;

    controls.screenSpacePanning = true;

    controls.minDistance = 10;
    controls.maxDistance = 5000;

    controls.target.set(0, 0, 0);

    controlsRef.current = controls;

    /*
     * Resize handling
     */
    const resize = () => {
      if (!mount || !camera || !renderer) {
        return;
      }

      const width = mount.clientWidth || 900;
      const height = mount.clientHeight || 600;

      camera.aspect = width / height;
      camera.updateProjectionMatrix();

      renderer.setSize(width, height, false);
    };

    resize();

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(mount);

    /*
     * Animation loop
     */
    const animate = () => {
      animationRef.current =
        requestAnimationFrame(animate);

      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    /*
     * Cleanup
     */
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }

      resizeObserver.disconnect();

      controls.dispose();

      if (groupRef.current) {
        disposeObject(groupRef.current);
      }

      renderer.dispose();

      if (
        renderer.domElement &&
        renderer.domElement.parentNode === mount
      ) {
        mount.removeChild(renderer.domElement);
      }

      sceneRef.current = null;
      cameraRef.current = null;
      rendererRef.current = null;
      controlsRef.current = null;
      groupRef.current = null;
    };
  }, []);

  /*
   * ------------------------------------------------------------
   * LOAD ALL STL MODELS
   * ------------------------------------------------------------
   */
  useEffect(() => {
    let cancelled = false;

    async function loadAllModels() {
      const group = groupRef.current;

      if (!group) {
        return;
      }

      setLoading(true);
      setLoadedModels([]);
      setErrors([]);
      setSelectedStructure("");

      /*
       * Clear previous models.
       *
       * This is important during Vite/React development reloads.
       */
      while (group.children.length > 0) {
        const child = group.children[0];

        disposeObject(child);
        group.remove(child);
      }

      if (!Array.isArray(models) || models.length === 0) {
        setLoading(false);
        return;
      }

      const successful = [];
      const failed = [];

      const loader = new STLLoader();

      /*
       * Load one model at a time.
       *
       * This avoids sending eight large STL requests
       * simultaneously.
       */
      for (const model of models) {
        if (cancelled) {
          return;
        }

        const structureName =
          getStructureName(model);

        let url = normalizeUrl(
          getModelUrl(model)
        );

        if (!url) {
          failed.push(
            `${structureName}: model URL is missing`
          );

          setErrors([...failed]);

          continue;
        }

        try {
          console.log(
            `Loading ${structureName}: ${url}`
          );

          const geometry =
            await new Promise((resolve, reject) => {
              loader.load(
                url,
                resolve,
                undefined,
                reject
              );
            });

          if (cancelled) {
            geometry.dispose();
            return;
          }

          /*
           * Recalculate normals so the reconstructed
           * surfaces appear smoother.
           */
          if (geometry.index) {
            geometry.computeVertexNormals();
          } else {
            geometry.computeVertexNormals();
          }

          geometry.computeBoundingBox();
          geometry.computeBoundingSphere();

          const material =
            makeMaterial(structureName);

          const mesh = new THREE.Mesh(
            geometry,
            material
          );

          mesh.name = structureName;

          /*
           * DO NOT change mesh.position here.
           *
           * All structures already belong to the same
           * patient coordinate system.
           */
          mesh.position.set(0, 0, 0);

          group.add(mesh);

          successful.push(structureName);

          setLoadedModels([...successful]);

          console.log(
            `Loaded ${structureName}`
          );
        } catch (error) {
          console.error(
            `Failed to load ${structureName}:`,
            error
          );

          failed.push(
            `${structureName}: ${
              error?.message ||
              "unable to load 3D model"
            }`
          );

          setErrors([...failed]);
        }
      }

      /*
       * --------------------------------------------------------
       * CENTER THE COMPLETE HEART ONLY ONCE
       * --------------------------------------------------------
       */
      if (!cancelled && group.children.length > 0) {
        /*
         * First make sure the group itself has no old offset.
         */
        group.position.set(0, 0, 0);

        /*
         * Calculate the bounding box of the COMPLETE heart.
         */
        const box =
          new THREE.Box3().setFromObject(group);

        const center =
          box.getCenter(new THREE.Vector3());

        const size =
          box.getSize(new THREE.Vector3());

        const maxDimension =
          Math.max(
            size.x,
            size.y,
            size.z
          );

        if (maxDimension > 0) {
          /*
           * Move the entire heart so its center
           * is exactly at the origin.
           */
          group.position.set(
            -center.x,
            -center.y,
            -center.z
          );

          /*
           * Camera distance based on actual heart size.
           */
          const distance =
            Math.max(
              maxDimension * 2.4,
              100
            );

          cameraDistanceRef.current =
            distance;

          const camera =
            cameraRef.current;

          const controls =
            controlsRef.current;

          if (camera && controls) {
            camera.position.set(
              distance,
              distance * 0.65,
              distance
            );

            camera.up.set(0, 1, 0);

            controls.target.set(
              0,
              0,
              0
            );

            camera.near =
              Math.max(
                0.1,
                maxDimension / 100
              );

            camera.far =
              Math.max(
                10000,
                maxDimension * 20
              );

            camera.lookAt(0, 0, 0);

            camera.updateProjectionMatrix();

            controls.update();
          }
        }
      }

      if (!cancelled) {
        setLoading(false);
      }
    }

    loadAllModels();

    return () => {
      cancelled = true;
    };
  }, [models]);

  /*
   * ------------------------------------------------------------
   * CAMERA VIEWS
   * ------------------------------------------------------------
   */
  const setView = (view) => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;

    if (!camera || !controls) {
      return;
    }

    const distance =
      cameraDistanceRef.current || 400;

    controls.target.set(0, 0, 0);

    if (view === "front") {
      camera.up.set(0, 1, 0);

      camera.position.set(
        0,
        0,
        distance
      );
    }

    if (view === "back") {
      camera.up.set(0, 1, 0);

      camera.position.set(
        0,
        0,
        -distance
      );
    }

    if (view === "left") {
      camera.up.set(0, 1, 0);

      camera.position.set(
        -distance,
        0,
        0
      );
    }

    if (view === "right") {
      camera.up.set(0, 1, 0);

      camera.position.set(
        distance,
        0,
        0
      );
    }

    if (view === "top") {
      /*
       * Top view.
       *
       * Change camera up vector so the orientation
       * remains stable instead of twisting.
       */
      camera.up.set(0, 0, -1);

      camera.position.set(
        0,
        distance,
        0
      );
    }

    if (view === "reset") {
      camera.up.set(0, 1, 0);

      camera.position.set(
        distance,
        distance * 0.65,
        distance
      );
    }

    camera.lookAt(0, 0, 0);

    controls.target.set(
      0,
      0,
      0
    );

    controls.update();
  };

  /*
   * ------------------------------------------------------------
   * SHOW / HIDE STRUCTURES
   * ------------------------------------------------------------
   */
  const toggleStructure = (structureName) => {
    const group = groupRef.current;

    if (!group) {
      return;
    }

    /*
     * Clicking a structure must NEVER hide/remove it.
     * It selects/highlights the structure instead.
     */
    const nextSelected =
      selectedStructure === structureName
        ? ""
        : structureName;

    setSelectedStructure(nextSelected);

    group.children.forEach((child) => {
      if (!child.material) {
        return;
      }

      const materials = Array.isArray(child.material)
        ? child.material
        : [child.material];

      materials.forEach((material) => {
        if ("emissive" in material) {
          const structureColor =
            STRUCTURE_COLORS[child.name] ?? 0xffffff;

          if (nextSelected === child.name) {
            material.emissive.setHex(0xffffff);
            material.emissiveIntensity = 0.38;
          } else {
            material.emissive.setHex(structureColor);
            material.emissiveIntensity = 0.0;
          }
        }

        material.transparent = false;
        material.opacity = 1.0;
      });
    });
  };

  /*
   * ------------------------------------------------------------
   * RENDER
   * ------------------------------------------------------------
   */
  const generatedCount = Array.isArray(models)
    ? models.length
    : 0;

  return (
    <section className="heart3d-section">
      <div className="heart3d-header">
        <div>
          <h2>
            Patient-Specific 3D Heart Model
          </h2>

          <p>
            Interactive visualization of
            reconstructed cardiac structures.
          </p>
        </div>

        <div className="heart3d-count">
          <strong>
            {loadedModels.length}/{generatedCount}
          </strong>

          <span>
            3D models loaded
          </span>
        </div>
      </div>

      {patientId && (
        <div className="heart3d-patient">
          Patient ID: {patientId}
        </div>
      )}

      <div className="heart3d-toolbar">
        <button
          type="button"
          onClick={() => setView("front")}
        >
          Front
        </button>

        <button
          type="button"
          onClick={() => setView("back")}
        >
          Back
        </button>

        <button
          type="button"
          onClick={() => setView("left")}
        >
          Left
        </button>

        <button
          type="button"
          onClick={() => setView("right")}
        >
          Right
        </button>

        <button
          type="button"
          onClick={() => setView("top")}
        >
          Top
        </button>

        <button
          type="button"
          className="heart3d-reset"
          onClick={() => setView("reset")}
        >
          Reset View
        </button>
      </div>

      <div className="heart3d-canvas-wrapper">
        <div
          ref={mountRef}
          className="heart3d-canvas"
        />

        {loading && (
          <div className="heart3d-loading">
            Loading reconstructed heart...
          </div>
        )}

        {!loading &&
          loadedModels.length === 0 && (
            <div className="heart3d-loading">
              No 3D models could be loaded.
            </div>
          )}

        <div className="heart3d-instructions">
          <strong>Rotate:</strong> Drag
          <br />
          <strong>Zoom:</strong> Mouse wheel
          <br />
          <strong>Pan:</strong> Right-click + drag
        </div>

        <div className="heart3d-loaded-badge">
          {loadedModels.length}/{generatedCount}{" "}
          3D models loaded
        </div>
      </div>

      {showLegend && (
        <div className="heart3d-legend-section">
          <div className="heart3d-legend-title">
            Cardiac Structures
          </div>

          <div className="heart3d-legend-grid">
            {STRUCTURE_ORDER.map(
              (structureName) => {
                const color =
                  STRUCTURE_COLORS[
                    structureName
                  ];

                const isLoaded =
                  loadedModels.includes(
                    structureName
                  );

                return (
                  <button
                    key={structureName}
                    type="button"
                    className={`heart3d-legend-item ${
                      isLoaded
                        ? ""
                        : "not-loaded"
                    } ${
                      selectedStructure === structureName
                        ? "selected"
                        : ""
                    }`}
                    onClick={() =>
                      toggleStructure(
                        structureName
                      )
                    }
                  >
                    <span
                      className="heart3d-color-dot"
                      style={{
                        backgroundColor: `#${color
                          .toString(16)
                          .padStart(6, "0")}`,
                      }}
                    />

                    <span>
                      {structureName}
                    </span>
                  </button>
                );
              }
            )}
          </div>

          <button
            type="button"
            className="heart3d-hide-legend"
            onClick={() =>
              setShowLegend(false)
            }
          >
            Hide Legend
          </button>
        </div>
      )}

      {!showLegend && (
        <div className="heart3d-show-legend-row">
          <button
            type="button"
            onClick={() =>
              setShowLegend(true)
            }
          >
            Show Legend
          </button>
        </div>
      )}

      {errors.length > 0 && (
        <div className="heart3d-errors">
          {errors.map(
            (errorMessage, index) => (
              <div key={index}>
                {errorMessage}
              </div>
            )
          )}
        </div>
      )}
    </section>
  );
}