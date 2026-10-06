import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const BACKEND = "https://digital-3d-heart-modelling.onrender.com";

const STRUCTURE_COLORS = {
  "Left Ventricle": 0xff3030,
  "Right Ventricle": 0x287cff,
  "Left Atrium": 0x25d366,
  "Right Atrium": 0xffd60a,
  Aorta: 0xff9500,
  "Pulmonary Artery": 0x00c7e8,
  "Superior Vena Cava": 0xaf52de,
  "Inferior Vena Cava": 0xbdbdbd,
};

const FALLBACK_COLORS = Object.values(STRUCTURE_COLORS);

const STRUCTURE_ALIASES = {
  "Left Ventricle": ["left ventricle", "lv", "left_ventricle"],
  "Right Ventricle": ["right ventricle", "rv", "right_ventricle"],
  "Left Atrium": ["left atrium", "la", "left_atrium"],
  "Right Atrium": ["right atrium", "ra", "right_atrium"],
  Aorta: ["aorta"],
  "Pulmonary Artery": ["pulmonary artery", "pa", "pulmonary_artery"],
  "Superior Vena Cava": ["superior vena cava", "svc", "superior_vena_cava"],
  "Inferior Vena Cava": ["inferior vena cava", "ivc", "inferior_vena_cava"],
};

const DEFECT_COLORS = {
  ASD: 0xFF1493, // ASD = bright pink/magenta
  VSD: 0x8B5A2B, // VSD = brown
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
  return match?.[1] || patientId || "";
}

function getModelUrls(path, patientId) {
  if (!path) return [];
  const normalized = normalizePath(path);

  if (/^https?:\/\//i.test(normalized)) return [normalized];

  const fileName = getFileName(normalized);
  if (!fileName) return [];

  const patientFolder = getPatientFolder(normalized, patientId);
  const urls = [];

  if (patientFolder) {
    urls.push(`${BACKEND}/models/${encodeURIComponent(patientFolder)}/${encodeURIComponent(fileName)}`);
    urls.push(`${BACKEND}/outputs/${encodeURIComponent(patientFolder)}/3d_models/${encodeURIComponent(fileName)}`);
  }

  urls.push(`${BACKEND}/models/${encodeURIComponent(fileName)}`);
  return [...new Set(urls)];
}

function matchesStructure(meshName, canonicalName) {
  const name = normalizeText(meshName);
  return (STRUCTURE_ALIASES[canonicalName] || [canonicalName]).some(
    (alias) => normalizeText(alias) === name || name.includes(normalizeText(alias))
  );
}

function structureColor(name, index) {
  if (STRUCTURE_COLORS[name] !== undefined) return STRUCTURE_COLORS[name];

  const normalized = normalizeText(name);
  for (const [canonical, aliases] of Object.entries(STRUCTURE_ALIASES)) {
    if (aliases.some((alias) => normalizeText(alias) === normalized || normalized.includes(normalizeText(alias)))) {
      return STRUCTURE_COLORS[canonical];
    }
  }

  return FALLBACK_COLORS[index % FALLBACK_COLORS.length];
}

function isTruthyDefectValue(value) {
  if (value === true) return true;
  if (value === false || value == null) return false;

  const text = normalizeText(value);

  // Check negative wording FIRST.
  // This prevents "not detected" from being interpreted as positive
  // merely because it contains the word "detected".
  if (
    text === "false" ||
    text === "0" ||
    text.includes("not detected") ||
    text.includes("not suspected") ||
    text.includes("negative") ||
    text.includes("absent") ||
    text.includes("no septal defect") ||
    text.includes("no defect")
  ) {
    return false;
  }

  return (
    text === "true" ||
    text === "1" ||
    text.includes("suspected") ||
    text.includes("positive") ||
    text === "detected"
  );
}

function isDefectDetected(defect, type) {
  if (!defect) return false;

  const key = type.toLowerCase();
  const item = defect[key] || defect[type] || defect[type.toUpperCase()] || defect[`${key}_screening`] || defect[`${key}_defect`];
  const flat = defect[`${key}_detected`] ?? defect[`${key}Detected`] ?? defect[`${key}_suspected`] ?? defect[`${key}Suspected`];

  if (item == null) return isTruthyDefectValue(flat);
  if (typeof item === "boolean") return item;

  // Explicit detected value takes priority.
  if (item.detected !== undefined) {
    if (item.detected === true) return true;
    if (item.detected === false) return false;
    return isTruthyDefectValue(item.detected);
  }

  return isTruthyDefectValue(
    item.status ||
      item.result ||
      item.classification ||
      item.prediction ||
      item.label ||
      item.assessment
  );
}

/*
 * Optional location support.
 * If the backend later supplies an actual defect coordinate, this function
 * accepts common names without changing the screening result itself.
 */
function getDefectCoordinate(defect, type) {
  if (!defect) return null;

  const key = type.toLowerCase();
  const item = defect[key] || defect[type] || defect[type.toUpperCase()] || {};
  const candidates = [
    item.location,
    item.coordinates,
    item.coordinate,
    item.centroid,
    item.center,
    item.point,
    item.position,
    defect[`${key}_location`],
    defect[`${key}_coordinates`],
    defect[`${key}_coordinate`],
    defect[`${key}_centroid`],
    defect[`${key}_point`],
    defect[`${key}_position`],
  ];

  for (const value of candidates) {
    if (Array.isArray(value) && value.length >= 3) {
      const p = value.slice(0, 3).map(Number);
      if (p.every(Number.isFinite)) return new THREE.Vector3(...p);
    }

    if (value && typeof value === "object") {
      const p = [value.x, value.y, value.z].map(Number);
      if (p.every(Number.isFinite)) return new THREE.Vector3(...p);
    }
  }

  return null;
}

export default function Heart3DViewer({
  models = [],
  reconstruction = null,
  patientId = "",
  septalDefect = null,
}) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const heartGroupRef = useRef(null);
  const animationRef = useRef(null);
  const defectObjectsRef = useRef({ ASD: null, VSD: null });
  const autoRotateRef = useRef(false);
  const defectDataRef = useRef({ ASD: null, VSD: null });

  const [loadedCount, setLoadedCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [autoRotate, setAutoRotate] = useState(false);
  const [webglError, setWebglError] = useState("");

  const individualModels = useMemo(() => {
    if (Array.isArray(models) && models.length) return models;
    return reconstruction?.individual_models || reconstruction?.individualModels || reconstruction?.models || [];
  }, [models, reconstruction]);

  const asdDetected = isDefectDetected(septalDefect, "ASD");
  const vsdDetected = isDefectDetected(septalDefect, "VSD");

  const getHeartMeshes = useCallback(() => {
    const group = heartGroupRef.current;
    if (!group) return [];
    return group.children.filter((child) => child.isMesh && child.userData?.isHeartStructure === true);
  }, []);

  const getStructureMesh = useCallback(
    (canonicalName) => {
      return getHeartMeshes().find((mesh) => matchesStructure(mesh.userData?.structure || mesh.name, canonicalName)) || null;
    },
    [getHeartMeshes]
  );

  const disposeObject = useCallback((object) => {
    if (!object) return;

    object.traverse((child) => {
      if (child.geometry) {
        child.geometry.dispose();
      }

      if (child.material) {
        const materials = Array.isArray(child.material)
          ? child.material
          : [child.material];

        materials.forEach((material) => {
          if (!material) return;

          // Dispose any textures if a material ever uses them.
          Object.keys(material).forEach((key) => {
            const value = material[key];
            if (value && value.isTexture) {
              value.dispose();
            }
          });

          material.dispose();
        });
      }
    });
  }, []);

  const getWorldBounds = useCallback(() => {
    const meshes = getHeartMeshes();
    const box = new THREE.Box3();
    meshes.forEach((mesh) => box.union(new THREE.Box3().setFromObject(mesh)));
    return box.isEmpty() ? null : box;
  }, [getHeartMeshes]);

  const raycastSurface = useCallback((mesh, origin, direction, maxDistance) => {
    const ray = new THREE.Raycaster(origin, direction.clone().normalize(), 0, maxDistance || Infinity);
    const hits = ray.intersectObject(mesh, false);
    if (!hits.length) return null;

    const hit = hits[0];
    const normal = hit.face?.normal
      ? hit.face.normal.clone().transformDirection(mesh.matrixWorld).normalize()
      : direction.clone().normalize();

    const towardOther = direction.clone().normalize();
    if (normal.dot(towardOther) < 0) normal.negate();

    return { point: hit.point.clone(), normal, distance: hit.distance };
  }, []);

  /*
   * Geometry fallback: find a close, mutually-facing surface between the
   * two chambers. This is deliberately used only when the backend has not
   * supplied an explicit defect coordinate.
   */
  const findSeptalInterface = useCallback(
    (chamberA, chamberB) => {
      const meshA = getStructureMesh(chamberA);
      const meshB = getStructureMesh(chamberB);
      if (!meshA || !meshB) return null;

      meshA.updateMatrixWorld(true);
      meshB.updateMatrixWorld(true);

      const boxA = new THREE.Box3().setFromObject(meshA);
      const boxB = new THREE.Box3().setFromObject(meshB);
      const centerA = boxA.getCenter(new THREE.Vector3());
      const centerB = boxB.getCenter(new THREE.Vector3());
      const axis = centerB.clone().sub(centerA);
      const centerDistance = axis.length();
      if (!centerDistance) return null;
      axis.normalize();

      const helper = Math.abs(axis.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
      const tangent1 = new THREE.Vector3().crossVectors(axis, helper).normalize();
      const tangent2 = new THREE.Vector3().crossVectors(axis, tangent1).normalize();

      const directions = [];
      const addDirection = (theta, phi) => {
        directions.push(
          axis.clone().multiplyScalar(Math.cos(theta))
            .add(tangent1.clone().multiplyScalar(Math.sin(theta) * Math.cos(phi)))
            .add(tangent2.clone().multiplyScalar(Math.sin(theta) * Math.sin(phi)))
            .normalize()
        );
      };

      addDirection(0, 0);
      [4, 8, 12, 16, 20, 25, 30, 35].forEach((degrees) => {
        const theta = THREE.MathUtils.degToRad(degrees);
        const count = degrees <= 16 ? 16 : 24;
        for (let i = 0; i < count; i += 1) addDirection(theta, (i / count) * Math.PI * 2);
      });

      let best = null;

      for (const direction of directions) {
        const hitA = raycastSurface(meshA, centerA, direction, centerDistance * 3);
        const hitB = raycastSurface(meshB, centerB, direction.clone().negate(), centerDistance * 3);
        if (!hitA || !hitB) continue;

        const gap = hitA.point.distanceTo(hitB.point);
        if (gap > centerDistance * 0.55) continue;

        const midpoint = hitA.point.clone().add(hitB.point).multiplyScalar(0.5);
        const along = midpoint.clone().sub(centerA).dot(axis);
        const axisPoint = centerA.clone().add(axis.clone().multiplyScalar(along));
        const offAxis = midpoint.distanceTo(axisPoint);

        const facingA = Math.max(0, hitA.normal.dot(direction));
        const facingB = Math.max(0, hitB.normal.dot(direction.clone().negate()));

        const score =
          gap * 1.0 +
          offAxis * 0.65 +
          (1 - facingA) * centerDistance * 0.18 +
          (1 - facingB) * centerDistance * 0.18;

        if (!best || score < best.score) {
          best = { score, hitA, hitB, midpoint, centerA, centerB };
        }
      }

      if (!best) return null;

      return {
        point: best.hitA.point,
        normal: best.hitA.normal,
        meshA,
        meshB,
      };
    },
    [getStructureMesh, raycastSurface]
  );

  const worldPointFromBackend = useCallback(
    (coordinate, type) => {
      if (!coordinate) return null;

      /*
       * Coordinates supplied by the backend are assumed to be in the same
       * patient/STL coordinate system. No recentering of individual STLs is
       * performed anywhere in this viewer.
       */
      const p = coordinate.clone();

      const item = septalDefect?.[type.toLowerCase()] || septalDefect?.[type] || {};
      const units = normalizeText(item?.units || septalDefect?.units || "");

      if (units === "voxel" || units === "voxels") {
        const spacing = reconstruction?.voxel_spacing_mm || reconstruction?.voxelSpacing || null;
        if (Array.isArray(spacing) && spacing.length >= 3) {
          p.set(p.x * Number(spacing[0]), p.y * Number(spacing[1]), p.z * Number(spacing[2]));
        }
      }

      return p;
    },
    [septalDefect, reconstruction]
  );

  const buildDefectData = useCallback(
    (type, chamberA, chamberB) => {
      const group = heartGroupRef.current;
      if (!group) return null;

      const explicit = getDefectCoordinate(septalDefect, type);
      if (explicit) {
        const worldPoint = worldPointFromBackend(explicit, type);
        const meshA = getStructureMesh(chamberA);
        if (worldPoint && meshA) {
          meshA.updateMatrixWorld(true);
          const localPoint = group.worldToLocal(worldPoint.clone());
          const center = new THREE.Box3().setFromObject(meshA).getCenter(new THREE.Vector3());
          const normalWorld = worldPoint.clone().sub(center).normalize();
          const normalEnd = group.worldToLocal(worldPoint.clone().add(normalWorld));
          const localNormal = normalEnd.sub(localPoint).normalize();
          return { point: localPoint, normal: localNormal, source: "backend-coordinate" };
        }
      }

      const result = findSeptalInterface(chamberA, chamberB);
      if (!result) return null;

      const localPoint = group.worldToLocal(result.point.clone());
      const localEnd = group.worldToLocal(result.point.clone().add(result.normal.clone().normalize()));
      const localNormal = localEnd.sub(localPoint).normalize();

      return { point: localPoint, normal: localNormal, source: "anatomical-interface" };
    },
    [findSeptalInterface, getStructureMesh, septalDefect, worldPointFromBackend]
  );

  const removeDefectObjects = useCallback(() => {
    const group = heartGroupRef.current;
    if (!group) return;

    Object.values(defectObjectsRef.current).forEach((object) => {
      if (object) {
        group.remove(object);
        disposeObject(object);
      }
    });

    defectObjectsRef.current = { ASD: null, VSD: null };
    defectDataRef.current = { ASD: null, VSD: null };
  }, [disposeObject]);

  /*
   * Create the defect as a real 3D object in the heart's LOCAL coordinate
   * system.  It is deliberately a recessed cavity shape:
   *
   * - no TorusGeometry
   * - no RingGeometry
   * - no screen-space overlay
   * - no HTML/SVG marker
   *
   * The marker is a child of heartGroup, so its anatomical position is
   * preserved when the complete heart is rotated.
   */
  const createDefectObject = useCallback(
    (type, data) => {
      const group = heartGroupRef.current;
      const box = getWorldBounds();

      if (!group || !data?.point || !data?.normal || !box) {
        return null;
      }

      const size = box.getSize(new THREE.Vector3());
      const maxDimension = Math.max(size.x, size.y, size.z);

      /*
       * Size the opening from the reconstructed heart itself so it remains
       * visible on different STL scales without becoming a giant marker.
       */
      const radius = THREE.MathUtils.clamp(
        maxDimension * 0.028,
        maxDimension * 0.016,
        maxDimension * 0.045
      );

      const normal = data.normal.clone().normalize();

      const marker = new THREE.Group();
      marker.name = `${type}_ANATOMICAL_DEFECT`;
      marker.userData.isDefectMarker = true;
      marker.userData.defectType = type;
      marker.userData.source = data.source;

      /*
       * data.point is already in heartGroup LOCAL coordinates.
       * Keep the defect in that coordinate system forever.
       */
      marker.position.copy(data.point);

      /*
       * The marker's local +Z axis is the anatomical surface normal.
       */
      marker.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 0, 1),
        normal
      );

      /*
       * Recessed dark cavity.
       *
       * The sphere intersects the STL surface instead of sitting in
       * front of it.  Its centre is pushed slightly into the heart and
       * the Z scale makes it shallow along the wall normal.
       *
       * This produces a hole-like cavity from every viewing angle while
       * remaining a normal Three.js 3D object attached to the heart.
       */
      const cavity = new THREE.Mesh(
        new THREE.SphereGeometry(
          radius,
          24,
          16
        ),
        new THREE.MeshStandardMaterial({
          // Match the fixed legend so viewers can immediately
          // understand which defect the 3D opening represents.
          color: type === "ASD" ? DEFECT_COLORS.ASD : DEFECT_COLORS.VSD,
          emissive: type === "ASD" ? DEFECT_COLORS.ASD : DEFECT_COLORS.VSD,
          emissiveIntensity: 2.8,
          roughness: 0.28,
          metalness: 0.0,
          side: THREE.DoubleSide,
          depthTest: true,
          depthWrite: false,
          transparent: false,
        })
      );

      cavity.scale.set(1.18, 1.18, 0.34);
      cavity.position.z = -radius * 0.12;
      cavity.renderOrder = 60;

      cavity.userData.isDefectHole = true;
      cavity.userData.defectType = type;

      marker.add(cavity);

      /*
       * A very small dark inner core gives the recessed centre depth.
       * It is NOT a ring and does not create a circular outline.
       */
      const core = new THREE.Mesh(
        new THREE.SphereGeometry(
          radius * 0.42,
          16,
          12
        ),
        new THREE.MeshStandardMaterial({
          color: 0x050505,
          roughness: 1,
          metalness: 0,
          side: THREE.DoubleSide,
          depthTest: true,
          depthWrite: true,
        })
      );

      core.scale.set(1.0, 1.0, 0.18);
      core.position.z = -radius * 0.28;
      core.renderOrder = 61;

      core.userData.isDefectHoleCore = true;
      core.userData.defectType = type;

      marker.add(core);

      // BLACK 3D BORDER AROUND THE DEFECT
      // The border is attached to the same marker group as the hole,
      // so it stays anatomically aligned while the heart rotates.
      const borderGeometry = new THREE.TorusGeometry(
        radius * 0.86,
        Math.max(radius * 0.075, 0.001),
        8,
        24
      );

      const borderMaterial = new THREE.MeshBasicMaterial({
        color: 0x000000,
        side: THREE.DoubleSide,
        depthTest: true,
        depthWrite: false,
      });

      const border = new THREE.Mesh(
        borderGeometry,
        borderMaterial
      );

      border.position.z = -radius * 0.06;
      border.renderOrder = 63;
      border.userData.isDefectBorder = true;
      border.userData.defectType = type;

      marker.add(border);

      group.add(marker);

      return marker;
    },
    [getWorldBounds]
  );

  const fitHeart = useCallback(() => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const heartGroup = heartGroupRef.current;

    if (!camera || !controls || !heartGroup) return;

    heartGroup.updateMatrixWorld(true);

    const box = getWorldBounds();
    if (!box || box.isEmpty()) return;

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());

    const maxDimension = Math.max(size.x, size.y, size.z);

    if (!Number.isFinite(maxDimension) || maxDimension <= 0) return;

    const fovRadians = THREE.MathUtils.degToRad(camera.fov || 45);

    // Camera distance that keeps the complete heart inside the viewport.
    const distance =
      (maxDimension * 0.5) /
      Math.tan(Math.max(fovRadians * 0.5, 0.01));

    const finalDistance = distance * 1.35;

    camera.position.set(
      center.x,
      center.y,
      center.z + finalDistance
    );

    camera.near = Math.max(maxDimension / 10000, 0.001);
    camera.far = Math.max(maxDimension * 20, 1000);
    camera.updateProjectionMatrix();

    controls.target.copy(center);

    controls.minDistance = Math.max(maxDimension * 0.15, 0.001);
    controls.maxDistance = Math.max(
      maxDimension * 8,
      controls.minDistance * 10
    );

    controls.update();
  }, [getWorldBounds]);

  const updateDefectObjects = useCallback(() => {
    const group = heartGroupRef.current;
    if (!group) return;

    removeDefectObjects();

    group.updateMatrixWorld(true);

    const createFor = (type, chamberA, chamberB) => {
      const detected = type === "ASD" ? asdDetected : vsdDetected;
      if (!detected) return;

      const data = buildDefectData(type, chamberA, chamberB);

      if (!data?.point || !data?.normal) {
        console.warn(
          `[3D] ${type} detected, but no anatomical defect position could be calculated.`
        );
        return;
      }

      const object = createDefectObject(type, data);

      if (object) {
        defectObjectsRef.current[type] = object;
        defectDataRef.current[type] = data;

        // Keep diagnostic metadata on the actual 3D marker.
        object.userData.isAnatomicalDefect = true;
        object.userData.defectType = type;
        object.userData.source = data.source;
      }
    };

    // ASD is represented at the LA/RA septal interface.
    createFor("ASD", "Left Atrium", "Right Atrium");

    // VSD is represented at the LV/RV septal interface.
    createFor("VSD", "Left Ventricle", "Right Ventricle");

    group.updateMatrixWorld(true);
  }, [
    asdDetected,
    vsdDetected,
    buildDefectData,
    createDefectObject,
    removeDefectObjects,
  ]);

  const setView = useCallback(
    (view) => {
      const camera = cameraRef.current;
      const controls = controlsRef.current;
      const box = getWorldBounds();
      if (!camera || !controls || !box) return;

      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const distance = Math.max(size.x, size.y, size.z) * 1.8;

      if (view === "front") camera.position.set(center.x, center.y, center.z + distance);
      if (view === "back") camera.position.set(center.x, center.y, center.z - distance);
      if (view === "left") camera.position.set(center.x - distance, center.y, center.z);
      if (view === "right") camera.position.set(center.x + distance, center.y, center.z);
      if (view === "top") camera.position.set(center.x, center.y + distance, center.z);
      if (view === "front" || view === "back" || view === "left" || view === "right") camera.up.set(0, 1, 0);
      if (view === "top") camera.up.set(0, 0, -1);

      camera.lookAt(center);
      controls.target.copy(center);
      controls.update();
    },
    [getWorldBounds]
  );

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x02050d);
    sceneRef.current = scene;

    const width = container.clientWidth || 1000;
    const height = container.clientHeight || 700;
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1000000);
    camera.position.set(0, 0, 300);
    cameraRef.current = camera;

    let renderer;
    try {
      // Keep the renderer deliberately conservative because patient-specific
      // STL meshes can contain a very large number of triangles.
      renderer = new THREE.WebGLRenderer({
        antialias: false,
        alpha: false,
        powerPreference: "default",
        preserveDrawingBuffer: false,
      });
    } catch (error) {
      console.error("[3D] WebGL initialization failed:", error);
      setWebglError("WebGL could not be initialized in this browser.");
      setLoading(false);
      return undefined;
    }

    // A high-DPI display can otherwise create a very large GPU framebuffer.
    renderer.setPixelRatio(1);
    renderer.setSize(width, height, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x02050d, 1);
    renderer.domElement.style.display = "block";
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.position = "absolute";
    renderer.domElement.style.inset = "0";
    renderer.domElement.style.zIndex = "0";
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const canvas = renderer.domElement;

    const handleContextLost = (event) => {
      event.preventDefault();
      console.error("[3D] WebGL context lost.");
      setWebglError(
        "The 3D graphics context was lost. Please wait a moment or reload the viewer."
      );
      setLoading(false);
    };

    const handleContextRestored = () => {
      console.info("[3D] WebGL context restored.");
      setWebglError("");
      renderer.setPixelRatio(1);
      renderer.setSize(
        container.clientWidth || width,
        container.clientHeight || height,
        false
      );
    };

    canvas.addEventListener("webglcontextlost", handleContextLost, false);
    canvas.addEventListener("webglcontextrestored", handleContextRestored, false);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enableRotate = true;
    controls.enableZoom = true;
    controls.enablePan = true;
    controls.autoRotate = false;
    controls.autoRotateSpeed = 1.5;
    controlsRef.current = controls;

    scene.add(new THREE.AmbientLight(0xffffff, 2.2));

    const light1 = new THREE.DirectionalLight(0xffffff, 3);
    light1.position.set(300, 300, 400);
    scene.add(light1);

    const light2 = new THREE.DirectionalLight(0x9dbdff, 1.8);
    light2.position.set(-300, 100, 200);
    scene.add(light2);

    const light3 = new THREE.DirectionalLight(0xffffff, 1.5);
    light3.position.set(0, -300, -300);
    scene.add(light3);

    const heartGroup = new THREE.Group();
    heartGroup.name = "PatientSpecificHeart";
    heartGroup.rotation.x = Math.PI;
    scene.add(heartGroup);
    heartGroupRef.current = heartGroup;

    let lastFrameTime = 0;
    const targetFrameMs = 1000 / 30;
    let pageVisible = !document.hidden;

    const handleVisibility = () => {
      pageVisible = !document.hidden;
    };
    document.addEventListener("visibilitychange", handleVisibility);

    const animate = (time = 0) => {
      animationRef.current = requestAnimationFrame(animate);
      if (!pageVisible) return;
      if (time - lastFrameTime < targetFrameMs) return;
      lastFrameTime = time;
      controls.autoRotate = autoRotateRef.current;
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const resize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (!w || !h) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    window.addEventListener("resize", resize);

    return () => {
      window.removeEventListener("resize", resize);
      document.removeEventListener("visibilitychange", handleVisibility);
      canvas.removeEventListener("webglcontextlost", handleContextLost);
      canvas.removeEventListener("webglcontextrestored", handleContextRestored);

      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      controls.dispose();
      scene.traverse((object) => {
        if (object.geometry) object.geometry.dispose();
        if (object.material) {
          if (Array.isArray(object.material)) object.material.forEach((m) => m.dispose());
          else object.material.dispose();
        }
      });
      renderer.dispose();
      if (renderer.domElement && container.contains(renderer.domElement)) container.removeChild(renderer.domElement);
      sceneRef.current = null;
      cameraRef.current = null;
      rendererRef.current = null;
      controlsRef.current = null;
      heartGroupRef.current = null;
    };
  }, []);

  useEffect(() => {
    autoRotateRef.current = autoRotate;
  }, [autoRotate]);

  useEffect(() => {
    const group = heartGroupRef.current;
    if (!group) return undefined;

    // Completely release the previous patient's meshes before starting
    // another patient. This is especially important for large STL files.
    while (group.children.length) {
      const child = group.children[0];
      group.remove(child);
      disposeObject(child);
    }

    setLoadedCount(0);
    setLoading(true);
    setWebglError("");
    defectObjectsRef.current = { ASD: null, VSD: null };
    defectDataRef.current = { ASD: null, VSD: null };

    if (!individualModels.length) {
      setLoading(false);
      return undefined;
    }

    const loader = new STLLoader();
    let cancelled = false;
    let currentIndex = 0;

    const loadNext = () => {
      if (cancelled) return;

      if (currentIndex >= individualModels.length) {
        setLoading(false);

        // Fit only after every structure is loaded. The separate defect
        // effect will then place VSD/ASD markers if they are detected.
        requestAnimationFrame(() => {
          if (!cancelled) fitHeart();
        });

        return;
      }

      const index = currentIndex;
      const model = individualModels[index];
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

      const urls = getModelUrls(path, patientId);

      if (model?.url && /^https?:\/\//i.test(String(model.url))) {
        urls.unshift(model.url);
      }

      const uniqueUrls = [...new Set(urls)];
      let attempt = 0;

      const tryNextUrl = () => {
        if (cancelled) return;

        if (attempt >= uniqueUrls.length) {
          console.error(
            "[3D] Failed to load model:",
            structure,
            uniqueUrls
          );

          currentIndex += 1;
          setLoadedCount(currentIndex);

          // Continue even if one structure is unavailable.
          loadNext();
          return;
        }

        const url = uniqueUrls[attempt++];

        loader.load(
          url,

          (geometry) => {
            if (cancelled) {
              geometry.dispose();
              return;
            }

            if (!geometry?.attributes?.position?.count) {
              geometry.dispose();
              tryNextUrl();
              return;
            }

            // Release any normals supplied by the STL and rebuild them once.
            // This avoids keeping unnecessary duplicate geometry attributes.
            if (geometry.hasAttribute("normal")) {
              geometry.deleteAttribute("normal");
            }

            geometry.computeVertexNormals();
            geometry.computeBoundingBox();
            geometry.computeBoundingSphere();

            if (cancelled) {
              geometry.dispose();
              return;
            }

            const mesh = new THREE.Mesh(
              geometry,
              new THREE.MeshLambertMaterial({
                color: structureColor(structure, index),
                side: THREE.FrontSide,
              })
            );

            mesh.name = structure;
            mesh.userData.structure = structure;
            mesh.userData.isHeartStructure = true;

            /* Never recenter, rotate, scale, or translate an individual STL. */
            group.add(mesh);

            currentIndex += 1;
            setLoadedCount(currentIndex);

            // Load one STL at a time. This is the key protection against
            // a large patient causing a GPU/CPU memory spike.
            loadNext();
          },

          undefined,

          (error) => {
            if (cancelled) return;

            console.warn(
              `[3D] Could not load ${structure} from ${url}`,
              error
            );

            tryNextUrl();
          }
        );
      };

      tryNextUrl();
    };

    loadNext();

    return () => {
      cancelled = true;
    };
  }, [individualModels, patientId, disposeObject, fitHeart]);

  useEffect(() => {
    if (
      webglError ||
      loading ||
      loadedCount !== individualModels.length ||
      !loadedCount
    ) {
      return undefined;
    }
    const timer = window.setTimeout(updateDefectObjects, 80);
    return () => window.clearTimeout(timer);
  }, [
    asdDetected,
    vsdDetected,
    webglError,
    loading,
    loadedCount,
    individualModels.length,
    updateDefectObjects,
  ]);

  return (
    <div className="heart-viewer-wrapper">
      <div className="heart-viewer-toolbar">
        {[["Front", "front"], ["Back", "back"], ["Left", "left"], ["Right", "right"], ["Top", "top"]].map(([label, view]) => (
          <button key={view} type="button" onClick={() => setView(view)}>{label}</button>
        ))}
        <button type="button" onClick={fitHeart}>Fit Heart</button>
        <button type="button" className={autoRotate ? "active" : ""} onClick={() => setAutoRotate((v) => !v)}>
          {autoRotate ? "Stop 360°" : "360° Rotate"}
        </button>
      </div>

      <div
        ref={mountRef}
        className="heart-viewer-canvas"
        style={{
          position: "relative",
          width: "100%",
          minHeight: "500px",
          overflow: "hidden",
        }}
      >
        {/* FIXED DEFECT LEGEND — HTML ONLY, NEVER PART OF heartGroup */}
        {(vsdDetected || asdDetected) && (
    <div
      style={{
        position: "absolute",
        top: "24px",
        right: "24px",
        zIndex: 9999,
        padding: "12px 16px",
        minWidth: "220px",
        borderRadius: "10px",
        background: "rgba(5, 8, 18, 0.88)",
        border: "1px solid rgba(255,255,255,0.16)",
        boxShadow: "0 6px 20px rgba(0,0,0,0.30)",
        pointerEvents: "none",
      }}
    >

      <div
        style={{
          color: "#ffffff",
          fontSize: "13px",
          fontWeight: 700,
          marginBottom: "8px",
        }}
      >
        Defect Legend
      </div>

                  {vsdDetected && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  color: "#8B5A2B",
                  fontWeight: 700,
                  marginBottom: asdDetected ? "8px" : "0",
                }}
              >
                <span
                  style={{
                    width: "12px",
                    height: "12px",
                    borderRadius: "50%",
                    background: "#8B5A2B",
                    border: "2px solid #000000",
                    boxSizing: "border-box",
                    flexShrink: 0,
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
                  gap: "8px",
                  color: "#FF1493",
                  fontWeight: 700,
                }}
              >
                <span
                  style={{
                    width: "12px",
                    height: "12px",
                    borderRadius: "50%",
                    background: "#FF1493",
                    border: "2px solid #000000",
                    boxSizing: "border-box",
                    flexShrink: 0,
                  }}
                />
                <span>ASD hole</span>
              </div>
            )}

    </div>
  )}


        {/* FIXED HEART STRUCTURE LEGEND — HTML ONLY, NEVER PART OF heartGroup */}
        <div
          style={{
            position: "absolute",
            top: (vsdDetected || asdDetected) ? "166px" : "24px",
            right: "24px",
            zIndex: 9998,
            width: "220px",
            padding: "14px 16px",
            borderRadius: "10px",
            background: "rgba(5, 8, 18, 0.88)",
            border: "1px solid rgba(255,255,255,0.16)",
            boxShadow: "0 6px 20px rgba(0,0,0,0.30)",
            pointerEvents: "none",
          }}
        >
          <div
            style={{
              color: "#ffffff",
              fontSize: "13px",
              fontWeight: 700,
              marginBottom: "10px",
            }}
          >
            Heart Structures
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              columnGap: "12px",
              rowGap: "8px",
            }}
          >
            {Object.entries(STRUCTURE_COLORS).map(([name, color]) => (
              <div
                key={name}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "7px",
                  minWidth: 0,
                }}
              >
                <span
                  style={{
                    width: "9px",
                    height: "9px",
                    borderRadius: "50%",
                    background: `#${color.toString(16).padStart(6, "0")}`,
                    flexShrink: 0,
                  }}
                />
                <span
                  style={{
                    color: "#e8eaf0",
                    fontSize: "11px",
                    lineHeight: 1.2,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                  title={name}
                >
                  {name}
                </span>
              </div>
            ))}
          </div>
        </div>

        
        {webglError && (
          <div
            className="heart-viewer-error"
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 10000,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: "24px",
              textAlign: "center",
              background: "rgba(2, 5, 13, 0.92)",
              color: "#ffffff",
              fontWeight: 700,
            }}
          >
            <div>
              <div style={{ fontSize: "18px", marginBottom: "8px" }}>
                3D viewer unavailable
              </div>
              <div style={{ fontSize: "13px", opacity: 0.78 }}>
                {webglError}
              </div>
              <button
                type="button"
                onClick={() => window.location.reload()}
                style={{
                  marginTop: "16px",
                  border: "0",
                  borderRadius: "10px",
                  padding: "10px 16px",
                  fontWeight: 700,
                  cursor: "pointer",
                }}
              >
                Reload 3D Viewer
              </button>
            </div>
          </div>
        )}
        {loading && !webglError && <div className="heart-viewer-overlay">Loading 3D heart...</div>}
        <div
          className="heart-model-status"
          style={{
            position: "absolute",
            top: "24px",
            left: "24px",
            right: "auto",
            zIndex: 9997,
          }}
        >
          {loadedCount}/{individualModels.length} 3D models loaded
        </div>
        <div className="heart-viewer-controls">
          <strong>Rotate:</strong> Drag<br />
          <strong>Zoom:</strong> Mouse wheel<br />
          <strong>Pan:</strong> Right-click + drag
        </div>
      </div>
    </div>
  );
}
