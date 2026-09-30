import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const BACKEND = "http://127.0.0.1:8000";

const STRUCTURE_COLORS = {
  "Left Ventricle": 0xff3030,
  "Right Ventricle": 0x287cff,
  "Left Atrium": 0x25d366,
  "Right Atrium": 0xffd60a,
  "Aorta": 0xff9500,
  "Pulmonary Artery": 0x00c7e8,
  "Superior Vena Cava": 0xaf52de,
  "Inferior Vena Cava": 0xbdbdbd,
};

const FALLBACK_COLORS = [
  0xff3030,
  0x287cff,
  0x25d366,
  0xffd60a,
  0xff9500,
  0x00c7e8,
  0xaf52de,
  0xbdbdbd,
];

const DEFECT_COLORS = {
  ASD: 0xff1493,
  VSD: 0xff9500,
};

const STRUCTURE_ALIASES = {
  "Left Ventricle": [
    "Left Ventricle",
    "LV",
    "left_ventricle",
    "left ventricle",
  ],
  "Right Ventricle": [
    "Right Ventricle",
    "RV",
    "right_ventricle",
    "right ventricle",
  ],
  "Left Atrium": [
    "Left Atrium",
    "LA",
    "left_atrium",
    "left atrium",
  ],
  "Right Atrium": [
    "Right Atrium",
    "RA",
    "right_atrium",
    "right atrium",
  ],
  Aorta: ["Aorta", "aorta"],
  "Pulmonary Artery": [
    "Pulmonary Artery",
    "PA",
    "pulmonary_artery",
    "pulmonary artery",
  ],
  "Superior Vena Cava": [
    "Superior Vena Cava",
    "SVC",
    "superior_vena_cava",
    "superior vena cava",
  ],
  "Inferior Vena Cava": [
    "Inferior Vena Cava",
    "IVC",
    "inferior_vena_cava",
    "inferior vena cava",
  ],
};

function normalizeText(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ");
}

function normalizePath(value) {
  return String(value || "").replace(/\\/g, "/");
}

function getFileName(value) {
  const path = normalizePath(value);
  return path.substring(path.lastIndexOf("/") + 1);
}

function getPatientFolder(value, patientId) {
  const path = normalizePath(value);

  const match = path.match(/outputs\/([^/]+)\/3d_models\//i);
  if (match?.[1]) return match[1];

  return patientId || "";
}

function getModelUrls(path, patientId) {
  if (!path) return [];

  const normalized = normalizePath(path);

  if (
    normalized.startsWith("http://") ||
    normalized.startsWith("https://")
  ) {
    return [normalized];
  }

  const fileName = getFileName(normalized);
  if (!fileName) return [];

  const patientFolder = getPatientFolder(normalized, patientId);
  const urls = [];

  if (patientFolder) {
    urls.push(
      `${BACKEND}/models/${encodeURIComponent(
        patientFolder
      )}/${encodeURIComponent(fileName)}`
    );

    urls.push(
      `${BACKEND}/outputs/${encodeURIComponent(
        patientFolder
      )}/3d_models/${encodeURIComponent(fileName)}`
    );
  }

  urls.push(
    `${BACKEND}/models/${encodeURIComponent(fileName)}`
  );

  return [...new Set(urls)];
}

function structureColor(name, index) {
  if (STRUCTURE_COLORS[name] !== undefined) {
    return STRUCTURE_COLORS[name];
  }

  const normalized = normalizeText(name);

  for (const [canonical, aliases] of Object.entries(STRUCTURE_ALIASES)) {
    if (
      aliases.some(
        (alias) => normalizeText(alias) === normalized
      )
    ) {
      return STRUCTURE_COLORS[canonical];
    }
  }

  return FALLBACK_COLORS[index % FALLBACK_COLORS.length];
}

function isDefectDetected(defect, type) {
  if (!defect) return false;

  const item =
    type === "ASD" ? defect.asd : defect.vsd;

  if (item == null) return false;

  if (typeof item === "boolean") {
    return item;
  }

  if (item.detected === true) return true;
  if (item.detected === false) return false;

  if (typeof item.detected === "string") {
    const detectedText = normalizeText(item.detected);

    if (
      detectedText === "false" ||
      detectedText === "0" ||
      detectedText.includes("not detected") ||
      detectedText.includes("not suspected") ||
      detectedText.includes("negative") ||
      detectedText.includes("absent")
    ) {
      return false;
    }

    if (
      detectedText === "true" ||
      detectedText === "1" ||
      detectedText.includes("suspected") ||
      detectedText === "detected" ||
      detectedText.includes("positive")
    ) {
      return true;
    }
  }

  const status = normalizeText(
    item.status ||
      item.result ||
      item.classification ||
      item.prediction ||
      item.label ||
      item.assessment ||
      ""
  );

  if (
    status.includes("not detected") ||
    status.includes("not suspected") ||
    status.includes("negative") ||
    status.includes("absent") ||
    status.includes("no septal defect") ||
    status.includes("no defect")
  ) {
    return false;
  }

  return (
    status.includes("suspected") ||
    status === "detected" ||
    status.includes("positive")
  );
}

function matchesStructure(meshName, canonicalName) {
  const name = normalizeText(meshName);
  const aliases =
    STRUCTURE_ALIASES[canonicalName] || [canonicalName];

  return aliases.some(
    (alias) => normalizeText(alias) === name
  );
}

export default function Heart3DViewer({
  models = [],
  reconstruction = null,
  patientId = "",
  septalDefect = null,
}) {
  const mountRef = useRef(null);

  // FINAL ANATOMICAL 3D FIX: no screen-space defect overlays are used.
  // Defect objects are children of heartGroup and inherit its transform.


  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const heartGroupRef = useRef(null);
  const animationRef = useRef(null);

  /*
   * These are NOT screen positions.
   *
   * Each defect is stored permanently in heartGroup-local
   * coordinates. The actual 3D marker and 3D arrow are
   * children of the same heartGroup.
   */
  const defectDataRef = useRef({
    ASD: null,
    VSD: null,
  });

  const defectObjectsRef = useRef({
    ASD: null,
    VSD: null,
  });

  const [loadedCount, setLoadedCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [autoRotate, setAutoRotate] = useState(false);
  const [webglError, setWebglError] = useState("");

  const autoRotateRef = useRef(false);

  const individualModels = useMemo(() => {
    if (Array.isArray(models) && models.length > 0) {
      return models;
    }

    return (
      reconstruction?.individual_models ||
      reconstruction?.individualModels ||
      reconstruction?.models ||
      []
    );
  }, [models, reconstruction]);

  const asdDetected = isDefectDetected(
    septalDefect,
    "ASD"
  );

  const vsdDetected = isDefectDetected(
    septalDefect,
    "VSD"
  );

  useEffect(() => {
    autoRotateRef.current = autoRotate;
  }, [autoRotate]);

  /*
   * ============================================================
   * CORE HELPERS
   * ============================================================
   */

  const getHeartMeshes = () => {
    const group = heartGroupRef.current;
    if (!group) return [];

    return group.children.filter(
      (child) =>
        child.isMesh &&
        child.userData?.isHeartStructure === true
    );
  };

  const getStructureMesh = (canonicalName) => {
    const meshes = getHeartMeshes();

    return (
      meshes.find((mesh) =>
        matchesStructure(
          mesh.userData?.structure || mesh.name,
          canonicalName
        )
      ) || null
    );
  };

  const disposeObject = (object) => {
    if (!object) return;

    object.traverse((child) => {
      if (child.geometry) {
        child.geometry.dispose();
      }

      if (child.material) {
        if (Array.isArray(child.material)) {
          child.material.forEach((material) =>
            material.dispose()
          );
        } else {
          child.material.dispose();
        }
      }
    });
  };

  /*
   * ============================================================
   * FIND TRUE SEPTAL SURFACE
   * ============================================================
   *
   * The old implementation used the globally closest vertices
   * between two chambers. That is not reliable: the closest
   * LA/RA or LV/RV surfaces can be near the base, outer wall,
   * or another anatomical contact.
   *
   * Instead:
   *
   * 1. Find the center of chamber A.
   * 2. Find the center of chamber B.
   * 3. Cast a ray from A's center toward B's center.
   * 4. The first surface hit on A is the wall facing B.
   * 5. Do the reverse for B.
   * 6. Use the midpoint of those two facing surfaces.
   *
   * This makes the marker represent the interface between
   * the two chambers rather than an arbitrary closest point.
   */

  const getWorldBoxCenter = (mesh) => {
    const box = new THREE.Box3().setFromObject(mesh);
    if (box.isEmpty()) return null;

    const center = new THREE.Vector3();
    box.getCenter(center);
    return center;
  };

  const getSurfaceHit = (
    sourceMesh,
    targetCenter
  ) => {
    if (!sourceMesh || !targetCenter) {
      return null;
    }

    sourceMesh.updateMatrixWorld(true);

    const sourceCenter =
      getWorldBoxCenter(sourceMesh);

    if (!sourceCenter) return null;

    const direction = new THREE.Vector3()
      .subVectors(targetCenter, sourceCenter);

    const distance = direction.length();

    if (distance < 1e-8) return null;

    direction.normalize();

    const raycaster = new THREE.Raycaster(
      sourceCenter,
      direction,
      0,
      distance * 2.5
    );

    /*
     * STL meshes are closed surfaces in the normal case.
     * DoubleSide makes the ray test reliable even if normals
     * in an imported STL are inconsistent.
     */
    const intersections = raycaster.intersectObject(
      sourceMesh,
      false
    );

    if (!intersections.length) {
      return null;
    }

    const hit = intersections[0];

    const point = hit.point.clone();

    let normal = direction.clone();

    if (hit.face?.normal) {
      normal
        .copy(hit.face.normal)
        .transformDirection(
          sourceMesh.matrixWorld
        )
        .normalize();

      /*
       * We want the normal pointing toward the opposite
       * chamber, because this is the septal-facing wall.
       */
      if (normal.dot(direction) < 0) {
        normal.negate();
      }
    }

    return {
      point,
      normal,
    };
  };

  const findSeptalInterface = (
    chamberA,
    chamberB
  ) => {
    const meshA =
      getStructureMesh(chamberA);

    const meshB =
      getStructureMesh(chamberB);

    if (!meshA || !meshB) {
      console.warn(
        `[3D] Cannot find ${chamberA} / ${chamberB} meshes`
      );
      return null;
    }

    const centerA =
      getWorldBoxCenter(meshA);

    const centerB =
      getWorldBoxCenter(meshB);

    if (!centerA || !centerB) {
      return null;
    }

    /*
     * Surface of A facing B.
     */
    const hitA = getSurfaceHit(
      meshA,
      centerB
    );

    /*
     * Surface of B facing A.
     */
    const hitB = getSurfaceHit(
      meshB,
      centerA
    );

    /*
     * Preferred result: both chamber surfaces were found.
     */
    if (hitA && hitB) {
      const point = new THREE.Vector3()
        .addVectors(
          hitA.point,
          hitB.point
        )
        .multiplyScalar(0.5);

      const normal = new THREE.Vector3()
        .addVectors(
          hitA.normal,
          hitB.normal
        );

      if (normal.lengthSq() < 1e-10) {
        normal.subVectors(
          centerB,
          centerA
        );
      }

      normal.normalize();

      return {
        point,
        normal,
        meshA,
        meshB,
        centerA,
        centerB,
      };
    }

    /*
     * Fallback: use whichever chamber surface was found.
     */
    if (hitA) {
      return {
        point: hitA.point,
        normal: hitA.normal,
        meshA,
        meshB,
        centerA,
        centerB,
      };
    }

    if (hitB) {
      return {
        point: hitB.point,
        normal: hitB.normal.clone().negate(),
        meshA,
        meshB,
        centerA,
        centerB,
      };
    }

    /*
     * Last-resort fallback only if raycasting failed.
     * This is still an anatomical point between the two
     * chamber centers, not a screen-space position.
     */
    const fallbackPoint = new THREE.Vector3()
      .addVectors(centerA, centerB)
      .multiplyScalar(0.5);

    const fallbackNormal = new THREE.Vector3()
      .subVectors(centerB, centerA)
      .normalize();

    return {
      point: fallbackPoint,
      normal: fallbackNormal,
      meshA,
      meshB,
      centerA,
      centerB,
    };
  };

  /*
   * ============================================================
   * STORE DEFECT IN HEART-LOCAL COORDINATES
   * ============================================================
   */

  const calculateDefectData = () => {
    const group = heartGroupRef.current;

    if (!group) return;

    group.updateMatrixWorld(true);

    const next = {
      ASD: null,
      VSD: null,
    };

    const saveDefect = (
      type,
      chamberA,
      chamberB
    ) => {
      const result =
        findSeptalInterface(
          chamberA,
          chamberB
        );

      if (!result) return;

      /*
       * Convert the anatomical world point into the
       * COMPLETE heart group's local coordinate system.
       *
       * This coordinate NEVER changes during camera rotation.
       */
      const localPoint =
        group.worldToLocal(
          result.point.clone()
        );

      /*
       * Convert the septal direction into heart-local space.
       */
      const localEnd =
        group.worldToLocal(
          result.point
            .clone()
            .add(result.normal)
        );

      const localNormal =
        localEnd
          .sub(localPoint)
          .normalize();

      next[type] = {
        point: localPoint,
        normal: localNormal,
        chamberA,
        chamberB,
      };

      console.log(
        `[3D] ${type} anatomical target`,
        {
          localPoint: localPoint.toArray(),
          localNormal: localNormal.toArray(),
          chamberA,
          chamberB,
        }
      );
    };

    if (asdDetected) {
      saveDefect(
        "ASD",
        "Left Atrium",
        "Right Atrium"
      );
    }

    if (vsdDetected) {
      saveDefect(
        "VSD",
        "Left Ventricle",
        "Right Ventricle"
      );
    }

    defectDataRef.current = next;
  };

  /*
   * ============================================================
   * 3D DEFECT OBJECT
   * ============================================================
   *
   * There is NO HTML marker, SVG line, CSS arrow, or camera
   * projection used for the actual defect.
   *
   * Everything below is a real THREE.Object3D under
   * heartGroup.
   */

  const createDefectObject = (
    type,
    data
  ) => {
    const group =
      heartGroupRef.current;

    if (!group || !data?.point) {
      return null;
    }

    /*
     * Calculate an appropriate scale from the complete heart.
     */
    const heartMeshes = getHeartMeshes();

    const heartBox = new THREE.Box3();

    heartMeshes.forEach((mesh) => {
      heartBox.union(
        new THREE.Box3().setFromObject(mesh)
      );
    });

    if (heartBox.isEmpty()) {
      return null;
    }

    const size = new THREE.Vector3();
    heartBox.getSize(size);

    const maxDimension = Math.max(
      size.x,
      size.y,
      size.z
    );

    /*
     * Small, anatomical-looking hole.
     */
    const holeRadius =
      maxDimension * 0.018;

    const defectGroup =
      new THREE.Group();

    defectGroup.name =
      `${type}_Anatomical_Defect`;

    defectGroup.position.copy(
      data.point
    );

    defectGroup.userData.isDefectMarker =
      true;

    defectGroup.userData.defectType =
      type;

    /*
     * The local +Z axis points through the defect wall
     * toward the opposite chamber.
     */
    const normal =
      data.normal
        .clone()
        .normalize();

    defectGroup.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 0, 1),
      normal
    );

    const color =
      DEFECT_COLORS[type];

    /*
     * ----------------------------------------------------------
     * BLACK RECESSED OPENING
     * ----------------------------------------------------------
     */

    const opening = new THREE.Mesh(
      new THREE.CircleGeometry(
        holeRadius,
        64
      ),
      new THREE.MeshBasicMaterial({
        color: 0x010105,
        side: THREE.DoubleSide,
        transparent: false,
        depthTest: true,
        depthWrite: false,
      })
    );

    opening.position.z =
      maxDimension * 0.0008;

    opening.renderOrder = 100;

    opening.userData.isDefectMarker =
      true;

    defectGroup.add(opening);

    /*
     * ----------------------------------------------------------
     * THIN COLORED RIM
     * ----------------------------------------------------------
     */

    const rim = new THREE.Mesh(
      new THREE.TorusGeometry(
        holeRadius * 1.04,
        holeRadius * 0.12,
        16,
        64
      ),
      new THREE.MeshBasicMaterial({
        color,
        side: THREE.DoubleSide,
        transparent: false,
        depthTest: true,
        depthWrite: false,
      })
    );

    rim.position.z =
      maxDimension * 0.0012;

    rim.renderOrder = 101;

    rim.userData.isDefectMarker =
      true;

    defectGroup.add(rim);

    /*
     * ----------------------------------------------------------
     * SUBTLE INNER DARK RIM
     * ----------------------------------------------------------
     */

    const innerRim = new THREE.Mesh(
      new THREE.TorusGeometry(
        holeRadius * 0.72,
        holeRadius * 0.045,
        12,
        48
      ),
      new THREE.MeshBasicMaterial({
        color: 0x000000,
        side: THREE.DoubleSide,
        depthTest: true,
        depthWrite: false,
      })
    );

    innerRim.position.z =
      maxDimension * 0.0015;

    innerRim.renderOrder = 102;

    innerRim.userData.isDefectMarker =
      true;

    defectGroup.add(innerRim);

    /*
     * ----------------------------------------------------------
     * SHORT 3D ARROW
     * ----------------------------------------------------------
     *
     * The arrow points TOWARD the hole.
     *
     * It is a child of defectGroup, which is a child of
     * heartGroup. Therefore it follows the exact anatomical
     * transform of the heart.
     */

    const arrowLength =
      maxDimension * 0.095;

    const arrowStart =
      normal
        .clone()
        .multiplyScalar(arrowLength);

    const arrowDirection =
      normal.clone().negate();

    const arrow =
      new THREE.ArrowHelper(
        arrowDirection,
        arrowStart,
        arrowLength,
        color,
        arrowLength * 0.22,
        arrowLength * 0.10
      );

    arrow.userData.isDefectMarker =
      true;

    arrow.line.material.depthTest =
      true;

    arrow.line.material.depthWrite =
      false;

    arrow.cone.material.depthTest =
      true;

    arrow.cone.material.depthWrite =
      false;

    arrow.line.renderOrder = 103;
    arrow.cone.renderOrder = 104;

    defectGroup.add(arrow);

    /*
     * ----------------------------------------------------------
     * SMALL LABEL PLATE
     * ----------------------------------------------------------
     *
     * No text is added here.
     *
     * This is intentionally only a small colored ring + hole.
     * The diagnostic text belongs to the fixed UI below the
     * viewer, never inside the rotating 3D world.
     */

    group.add(defectGroup);

    return defectGroup;
  };

  const removeDefectObjects = () => {
    const group =
      heartGroupRef.current;

    if (!group) return;

    const oldObjects =
      group.children.filter(
        (child) =>
          child.userData?.isDefectMarker ===
          true
      );

    oldObjects.forEach((object) => {
      group.remove(object);
      disposeObject(object);
    });

    defectObjectsRef.current = {
      ASD: null,
      VSD: null,
    };
  };

  const updateDefectObjects = () => {
    const group =
      heartGroupRef.current;

    if (!group) return;

    removeDefectObjects();

    calculateDefectData();

    if (
      asdDetected &&
      defectDataRef.current.ASD
    ) {
      defectObjectsRef.current.ASD =
        createDefectObject(
          "ASD",
          defectDataRef.current.ASD
        );
    }

    if (
      vsdDetected &&
      defectDataRef.current.VSD
    ) {
      defectObjectsRef.current.VSD =
        createDefectObject(
          "VSD",
          defectDataRef.current.VSD
        );
    }
  };

  /*
   * ============================================================
   * FIT HEART
   * ============================================================
   */

  const fitHeart = () => {
    const group =
      heartGroupRef.current;

    const camera =
      cameraRef.current;

    const controls =
      controlsRef.current;

    if (
      !group ||
      !camera ||
      !controls
    ) {
      return;
    }

    const meshes =
      getHeartMeshes();

    if (!meshes.length) return;

    group.updateMatrixWorld(true);

    const box =
      new THREE.Box3();

    meshes.forEach((mesh) => {
      box.union(
        new THREE.Box3().setFromObject(mesh)
      );
    });

    if (box.isEmpty()) return;

    const center =
      new THREE.Vector3();

    const size =
      new THREE.Vector3();

    box.getCenter(center);
    box.getSize(size);

    const maxDimension =
      Math.max(
        size.x,
        size.y,
        size.z
      );

    if (
      !Number.isFinite(maxDimension) ||
      maxDimension <= 0
    ) {
      return;
    }

    /*
     * This is intentionally close to the original working
     * viewer. We do not move or scale the heart group.
     */
    const distance =
      maxDimension * 1.8;

    camera.position.set(
      center.x,
      center.y,
      center.z + distance
    );

    camera.near =
      Math.max(
        maxDimension / 10000,
        0.001
      );

    camera.far =
      Math.max(
        maxDimension * 100,
        1000
      );

    camera.updateProjectionMatrix();

    controls.target.copy(center);

    controls.minDistance =
      maxDimension * 0.15;

    controls.maxDistance =
      maxDimension * 8;

    controls.update();
  };

  /*
   * ============================================================
   * CAMERA PRESETS
   * ============================================================
   */

  const setView = (view) => {
    const group =
      heartGroupRef.current;

    const camera =
      cameraRef.current;

    const controls =
      controlsRef.current;

    if (
      !group ||
      !camera ||
      !controls
    ) {
      return;
    }

    const meshes =
      getHeartMeshes();

    if (!meshes.length) return;

    group.updateMatrixWorld(true);

    const box =
      new THREE.Box3();

    meshes.forEach((mesh) => {
      box.union(
        new THREE.Box3().setFromObject(mesh)
      );
    });

    if (box.isEmpty()) return;

    const center =
      new THREE.Vector3();

    const size =
      new THREE.Vector3();

    box.getCenter(center);
    box.getSize(size);

    const distance =
      Math.max(
        size.x,
        size.y,
        size.z
      ) * 1.8;

    if (view === "front") {
      camera.position.set(
        center.x,
        center.y,
        center.z + distance
      );
      camera.up.set(0, 1, 0);
    } else if (view === "back") {
      camera.position.set(
        center.x,
        center.y,
        center.z - distance
      );
      camera.up.set(0, 1, 0);
    } else if (view === "left") {
      camera.position.set(
        center.x - distance,
        center.y,
        center.z
      );
      camera.up.set(0, 1, 0);
    } else if (view === "right") {
      camera.position.set(
        center.x + distance,
        center.y,
        center.z
      );
      camera.up.set(0, 1, 0);
    } else if (view === "top") {
      camera.position.set(
        center.x,
        center.y + distance,
        center.z
      );
      camera.up.set(0, 0, -1);
    }

    camera.lookAt(center);

    controls.target.copy(center);
    controls.update();
  };

  /*
   * ============================================================
   * INITIALIZE THREE.JS
   * ============================================================
   */

  useEffect(() => {
    const container =
      mountRef.current;

    if (!container) return;

    const scene =
      new THREE.Scene();

    scene.background =
      new THREE.Color(0x02050d);

    sceneRef.current = scene;

    const width =
      container.clientWidth || 1000;

    const height =
      container.clientHeight || 700;

    const camera =
      new THREE.PerspectiveCamera(
        45,
        width / height,
        0.01,
        1000000
      );

    camera.position.set(
      0,
      0,
      300
    );

    cameraRef.current =
      camera;

    let renderer;

    try {
      renderer =
        new THREE.WebGLRenderer({
          antialias: true,
          alpha: false,
          powerPreference:
            "high-performance",
        });
    } catch (error) {
      console.error(
        "[3D] WebGL initialization failed",
        error
      );

      setWebglError(
        "WebGL could not be initialized in this browser."
      );

      setLoading(false);
      return undefined;
    }

    renderer.setPixelRatio(
      Math.min(
        window.devicePixelRatio || 1,
        2
      )
    );

    renderer.setSize(
      width,
      height
    );

    renderer.setClearColor(
      0x02050d,
      1
    );

    renderer.outputColorSpace =
      THREE.SRGBColorSpace;

    renderer.domElement.style.display =
      "block";

    renderer.domElement.style.width =
      "100%";

    renderer.domElement.style.height =
      "100%";

    renderer.domElement.style.position =
      "absolute";

    renderer.domElement.style.inset =
      "0";

    renderer.domElement.style.zIndex =
      "0";

    container.appendChild(
      renderer.domElement
    );

    rendererRef.current =
      renderer;

    renderer.domElement.addEventListener(
      "webglcontextlost",
      (event) => {
        event.preventDefault();
        console.error(
          "[3D] WebGL context lost"
        );
      }
    );

    const controls =
      new OrbitControls(
        camera,
        renderer.domElement
      );

    controls.enableDamping =
      true;

    controls.dampingFactor =
      0.08;

    controls.enableRotate =
      true;

    controls.enableZoom =
      true;

    controls.enablePan =
      true;

    controls.autoRotate =
      false;

    controls.autoRotateSpeed =
      1.5;

    controlsRef.current =
      controls;

    /*
     * Lighting.
     */
    scene.add(
      new THREE.AmbientLight(
        0xffffff,
        2.2
      )
    );

    const light1 =
      new THREE.DirectionalLight(
        0xffffff,
        3
      );

    light1.position.set(
      300,
      300,
      400
    );

    scene.add(light1);

    const light2 =
      new THREE.DirectionalLight(
        0x9dbdff,
        1.8
      );

    light2.position.set(
      -300,
      100,
      200
    );

    scene.add(light2);

    const light3 =
      new THREE.DirectionalLight(
        0xffffff,
        1.5
      );

    light3.position.set(
      0,
      -300,
      -300
    );

    scene.add(light3);

    /*
     * ONE parent for all anatomical structures.
     *
     * The initial MRI/STL orientation is preserved.
     */
    const heartGroup =
      new THREE.Group();

    heartGroup.name =
      "PatientSpecificHeart";

    heartGroup.rotation.x =
      Math.PI;

    scene.add(
      heartGroup
    );

    heartGroupRef.current =
      heartGroup;

    const animate = () => {
      animationRef.current =
        requestAnimationFrame(
          animate
        );

      controls.autoRotate =
        autoRotateRef.current;

      controls.update();

      /*
       * No defect screen projection happens here.
       *
       * The defect is a real 3D child of heartGroup.
       * OrbitControls rotates the camera around the same
       * fixed anatomical model, so the marker remains attached
       * to the exact anatomical location.
       */
      renderer.render(
        scene,
        camera
      );
    };

    animate();

    const resize = () => {
      if (!mountRef.current) return;

      const w =
        mountRef.current.clientWidth;

      const h =
        mountRef.current.clientHeight;

      if (!w || !h) return;

      camera.aspect =
        w / h;

      camera.updateProjectionMatrix();

      renderer.setSize(
        w,
        h
      );
    };

    window.addEventListener(
      "resize",
      resize
    );

    return () => {
      window.removeEventListener(
        "resize",
        resize
      );

      if (animationRef.current) {
        cancelAnimationFrame(
          animationRef.current
        );
      }

      controls.dispose();

      scene.traverse(
        (object) => {
          if (object.geometry) {
            object.geometry.dispose();
          }

          if (object.material) {
            if (Array.isArray(object.material)) {
              object.material.forEach(
                (material) =>
                  material.dispose()
              );
            } else {
              object.material.dispose();
            }
          }
        }
      );

      renderer.dispose();

      if (
        renderer.domElement &&
        container.contains(
          renderer.domElement
        )
      ) {
        container.removeChild(
          renderer.domElement
        );
      }

      sceneRef.current = null;
      cameraRef.current = null;
      rendererRef.current = null;
      controlsRef.current = null;
      heartGroupRef.current = null;
    };
  }, []);

  /*
   * ============================================================
   * LOAD ALL STL MODELS
   * ============================================================
   */

  useEffect(() => {
    const group =
      heartGroupRef.current;

    if (!group) return;

    /*
     * Remove ONLY anatomical mesh children.
     * Defect markers are also removed because a new set will
     * be generated after loading.
     */
    while (group.children.length) {
      const child =
        group.children[0];

      group.remove(child);
      disposeObject(child);
    }

    defectDataRef.current = {
      ASD: null,
      VSD: null,
    };

    defectObjectsRef.current = {
      ASD: null,
      VSD: null,
    };

    setLoadedCount(0);
    setLoading(true);

    if (
      !individualModels ||
      individualModels.length === 0
    ) {
      setLoading(false);
      return;
    }

    const loader =
      new STLLoader();

    let completed = 0;

    const finishOne = () => {
      completed += 1;
      setLoadedCount(completed);

      if (
        completed >=
        individualModels.length
      ) {
        setLoading(false);

        /*
         * Wait one frame so every mesh has its world matrix.
         */
        requestAnimationFrame(() => {
          fitHeart();

          /*
           * Calculate the anatomical defect AFTER every
           * chamber has been loaded.
           */
          requestAnimationFrame(() => {
            updateDefectObjects();
          });
        });
      }
    };

    const loadModel = (
      model,
      index
    ) => {
      const structure =
        model?.structure ||
        model?.name ||
        model?.label ||
        `Structure ${index + 1}`;

      const path =
        model?.stl ||
        model?.url ||
        model?.file_url ||
        model?.path ||
        model?.filename ||
        "";

      const urls = getModelUrls(
        path,
        patientId
      );

      if (
        model?.url &&
        /^https?:\/\//i.test(
          String(model.url)
        )
      ) {
        urls.unshift(
          model.url
        );
      }

      const uniqueUrls =
        [...new Set(urls)];

      let attempt = 0;

      const tryNext = () => {
        if (
          attempt >=
          uniqueUrls.length
        ) {
          console.error(
            "[3D] Failed to load model:",
            structure,
            uniqueUrls
          );

          finishOne();
          return;
        }

        const url =
          uniqueUrls[attempt];

        attempt += 1;

        console.log(
          `[3D] Loading ${structure}:`,
          url
        );

        loader.load(
          url,

          (geometry) => {
            if (
              !geometry?.attributes?.position ||
              geometry.attributes.position.count === 0
            ) {
              console.error(
                `[3D] Empty STL geometry: ${structure}`
              );

              tryNext();
              return;
            }

            geometry.computeVertexNormals();
            geometry.computeBoundingBox();
            geometry.computeBoundingSphere();

            const material =
              new THREE.MeshPhysicalMaterial({
                color:
                  structureColor(
                    structure,
                    index
                  ),
                roughness: 0.4,
                metalness: 0.05,
                clearcoat: 0.2,
                side:
                  THREE.DoubleSide,
              });

            const mesh =
              new THREE.Mesh(
                geometry,
                material
              );

            mesh.name =
              structure;

            mesh.userData.structure =
              structure;

            mesh.userData.isHeartStructure =
              true;

            mesh.castShadow =
              true;

            mesh.receiveShadow =
              true;

            /*
             * CRITICAL:
             *
             * Do not recenter, translate, scale, or rotate
             * individual STL files. Their common patient
             * coordinate system is required for anatomical
             * interface detection.
             */
            group.add(mesh);

            console.log(
              `[3D] Loaded ${structure}`,
              {
                vertices:
                  geometry.attributes.position.count,
                bounds:
                  geometry.boundingBox,
              }
            );

            finishOne();
          },

          undefined,

          (error) => {
            console.warn(
              `[3D] Failed URL for ${structure}:`,
              url,
              error
            );

            tryNext();
          }
        );
      };

      tryNext();
    };

    individualModels.forEach(
      loadModel
    );
  }, [
    individualModels,
    patientId,
  ]);

  /*
   * ============================================================
   * UPDATE DEFECTS WHEN SCREENING RESULT CHANGES
   * ============================================================
   */

  useEffect(() => {
    if (
      loading ||
      loadedCount !==
        individualModels.length ||
      loadedCount === 0
    ) {
      return;
    }

    const timer =
      setTimeout(() => {
        updateDefectObjects();
      }, 100);

    return () =>
      clearTimeout(timer);
  }, [
    asdDetected,
    vsdDetected,
    loading,
    loadedCount,
    individualModels.length,
  ]);

  /*
   * ============================================================
   * RENDER
   * ============================================================
   */

  return (
    <div
      className="heart-viewer-wrapper"
      style={{
        width: "100%",
      }}
    >
      <div
        className="heart-viewer-toolbar"
        style={{
          display: "flex",
          gap: "8px",
          flexWrap: "wrap",
          justifyContent: "center",
          marginBottom: "12px",
        }}
      >
        <button
          type="button"
          onClick={() =>
            setView("front")
          }
        >
          Front
        </button>

        <button
          type="button"
          onClick={() =>
            setView("back")
          }
        >
          Back
        </button>

        <button
          type="button"
          onClick={() =>
            setView("left")
          }
        >
          Left
        </button>

        <button
          type="button"
          onClick={() =>
            setView("right")
          }
        >
          Right
        </button>

        <button
          type="button"
          onClick={() =>
            setView("top")
          }
        >
          Top
        </button>

        <button
          type="button"
          onClick={fitHeart}
        >
          Fit Heart
        </button>

        <button
          type="button"
          onClick={() => {
            const group =
              heartGroupRef.current;

            if (!group) return;

            getHeartMeshes().forEach(
              (mesh) => {
                mesh.visible = true;
              }
            );

            Object.values(
              defectObjectsRef.current
            ).forEach(
              (object) => {
                if (object) {
                  object.visible =
                    true;
                }
              }
            );
          }}
        >
          Show All
        </button>

        <button
          type="button"
          className={
            autoRotate
              ? "active"
              : ""
          }
          onClick={() =>
            setAutoRotate(
              (value) => !value
            )
          }
        >
          {autoRotate
            ? "Stop 360°"
            : "360° Rotate"}
        </button>
      </div>

      <div
        ref={mountRef}
        className="heart-viewer-canvas"
        style={{
          position: "relative",
          width: "100%",
          height: "700px",
          minHeight: "500px",
          overflow: "hidden",
          background: "#02050d",
          borderRadius: "12px",
        }}
      >
        {webglError && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 20,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#ff6b6b",
              background: "#02050d",
              fontSize: "18px",
              textAlign: "center",
              padding: "30px",
            }}
          >
            {webglError}
          </div>
        )}

        {loading && !webglError && (
          <div
            className="heart-viewer-overlay"
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 10,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#ffffff",
              background:
                "rgba(2,5,13,0.35)",
              pointerEvents: "none",
            }}
          >
            Loading 3D heart...
          </div>
        )}

        <div
          className="heart-model-status"
          style={{
            position: "absolute",
            right: "28px",
            top: "28px",
            zIndex: 5,
            padding:
              "14px 22px",
            border:
              "1px solid #5540a8",
            borderRadius: "18px",
            background:
              "rgba(30,26,70,0.88)",
            color: "#ddd9ff",
            fontSize: "18px",
            fontWeight: 700,
            pointerEvents:
              "none",
          }}
        >
          {loadedCount}/
          {individualModels.length}{" "}
          3D models loaded
        </div>

        {/* FIXED VSD / ASD LEGEND
            This is HTML UI, not part of heartGroup.
            It stays fixed while the 3D heart rotates.
            It is placed below the model-status box. */}
        {(vsdDetected || asdDetected) && (
          <div
            className="heart-defect-legend"
            style={{
              position: "absolute",
              top: "104px",
              right: "28px",
              zIndex: 30,
              minWidth: "165px",
              padding: "12px 14px",
              border: "1px solid rgba(255,255,255,0.20)",
              borderRadius: "12px",
              background: "rgba(5,8,18,0.96)",
              boxShadow: "0 8px 26px rgba(0,0,0,0.45)",
              pointerEvents: "none",
              fontFamily: "Arial, sans-serif",
            }}
          >
            <div
              style={{
                color: "#ffffff",
                fontSize: "13px",
                fontWeight: 700,
                marginBottom: "9px",
              }}
            >
              Defect Legend
            </div>

            {vsdDetected && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "9px",
                  marginBottom: asdDetected ? "8px" : "0",
                  color: "#ff9500",
                  fontSize: "14px",
                  fontWeight: 700,
                  lineHeight: "18px",
                }}
              >
                <span
                  style={{
                    display: "inline-block",
                    width: "11px",
                    height: "11px",
                    minWidth: "11px",
                    borderRadius: "50%",
                    background: "#ff9500",
                    boxShadow:
                      "0 0 7px #ff9500, 0 0 13px rgba(255,149,0,0.65)",
                  }}
                />
                <span>VSD hole</span>
              </div>
            )}

            {asdDetected && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "9px",
                  color: "#ff1493",
                  fontSize: "14px",
                  fontWeight: 700,
                  lineHeight: "18px",
                }}
              >
                <span
                  style={{
                    display: "inline-block",
                    width: "11px",
                    height: "11px",
                    minWidth: "11px",
                    borderRadius: "50%",
                    background: "#ff1493",
                    boxShadow:
                      "0 0 7px #ff1493, 0 0 13px rgba(255,20,147,0.65)",
                  }}
                />
                <span>ASD hole</span>
              </div>
            )}
          </div>
        )}

        {/*
         * FIXED UI ONLY.
         *
         * These elements are siblings of the canvas content
         * and are NOT part of heartGroup. They never rotate.
         */}
        <div
          className="heart-viewer-controls"
          style={{
            position: "absolute",
            left: "28px",
            bottom: "28px",
            zIndex: 5,
            padding:
              "16px 20px",
            borderRadius: "16px",
            background:
              "rgba(7,10,22,0.82)",
            color: "#ffffff",
            lineHeight: 1.8,
            pointerEvents:
              "none",
          }}
        >
          <strong>
            Rotate:
          </strong>{" "}
          Drag
          <br />
          <strong>
            Zoom:
          </strong>{" "}
          Mouse wheel
          <br />
          <strong>
            Pan:
          </strong>{" "}
          Right-click + drag
        </div>
      </div>

    </div>
  );
}
