// Per-vertex skinning diagnostics: an execution evidence chain for localizing
// deformation errors (e.g. abnormal stretching of garment meshes).
//
// The tracer re-executes the exact CPU skinning math per vertex and records
// every intermediate value as evidence: stored weights, the skinning branch
// actually taken, per-joint bone matrices and rigid results. Diagnostics
// NEVER modify weights or positions — load-time repair happens only in
// validate.js and is itself exposed as evidence (asset.mesh.weightRepairs).
//
// Cause taxonomy (see classifyVertexTrace), separated by twist and
// non-uniform-scale probes:
//   numerical   - float32 / fixed-point divergence between CPU and GPU paths
//   coordinate  - space/matrix handling (e.g. normal matrix under non-uniform scale)
//   algorithmic - inherent LBS limitation: linear interpolation between
//                 strongly rotated joints (candy-wrapper under large twist)
//   weights     - stored weight sum != 1 or negative weights (load-stage problem)
//
// Log cost is bounded: mesh scans keep O(1) aggregates plus top-K records,
// and TraceLog is a fixed-capacity ring.

import { mat4, quat, vec3 } from './math.js';
import { MAX_INFLUENCES, f32ToBase64, base64ToF32, u16ToBase64 } from './format.js';
import {
  loadAsset, buildSkeleton, computeWorldMatrices, computeSkinMatrices,
} from './loader.js';
import { bindPose } from './animation.js';

export const WEIGHT_SUM_TOL = 1e-3;   // matches SUM_WARN_EPS in validate.js
export const RESIDUAL_TOL = 1e-3;     // matches the GPU<->CPU check tolerance
export const ANGLE_TOL = Math.PI / 4; // 45 deg: "large" relative rotation
export const SPREAD_TOL = 0.5;        // model units
export const ANISOTROPY_TOL = 1.01;

export const REPLAY_TYPE = 'rskin-skin-trace';
export const REPLAY_VERSION = 1;

// ---------- per-vertex trace ----------

// Column lengths of the upper 3x3 of a column-major mat4 -> max/min ratio.
function matrixAnisotropy(m, o) {
  const c0 = Math.hypot(m[o], m[o + 1], m[o + 2]);
  const c1 = Math.hypot(m[o + 4], m[o + 5], m[o + 6]);
  const c2 = Math.hypot(m[o + 8], m[o + 9], m[o + 10]);
  const mn = Math.min(c0, c1, c2);
  return mn > 0 ? Math.max(c0, c1, c2) / mn : Infinity;
}

// Gram-Schmidt orthonormalized columns of the upper 3x3 (rotation estimate).
function orthoColumns(m) {
  const c0 = vec3.normalize([m[0], m[1], m[2]]);
  const d = vec3.dot(c0, [m[4], m[5], m[6]]);
  const c1 = vec3.normalize([m[4] - c0[0] * d, m[5] - c0[1] * d, m[6] - c0[2] * d]);
  return [c0, c1, vec3.cross(c0, c1)];
}

// Rotation angle between two affine matrices' rotation parts (radians).
export function rotationAngleBetween(mA, mB) {
  const A = orthoColumns(mA);
  const B = orthoColumns(mB);
  // trace(A * B^T) = sum of per-column dot products
  const tr = vec3.dot(A[0], B[0]) + vec3.dot(A[1], B[1]) + vec3.dot(A[2], B[2]);
  return Math.acos(Math.min(Math.max((tr - 1) / 2, -1), 1));
}

// Load-stage repairs for one vertex: string array, or null when the asset
// never went through the loader (repair history unknown — reported as such).
const repairCache = new WeakMap();
function loadRepairsOf(asset, v) {
  const r = asset.mesh.weightRepairs;
  if (!r) return null;
  let sets = repairCache.get(r);
  if (!sets) {
    sets = {
      clamped: new Set(r.clamped ?? []),
      renormalized: new Set(r.rescaled ?? []),
      merged: new Set(r.merged ?? []),
    };
    repairCache.set(r, sets);
  }
  const out = [];
  if (sets.clamped.has(v)) out.push('clamped');
  if (sets.renormalized.has(v)) out.push('renormalized');
  if (sets.merged.has(v)) out.push('merged');
  return out;
}

function maxComponentDiff(a, b) {
  return Math.max(
    Math.abs(a[0] - b[0]),
    Math.abs(a[1] - b[1]),
    Math.abs(a[2] - b[2]),
  );
}

// Traces one vertex through the CPU skinning path. Uses the weights exactly
// as stored — no renormalization, no clamping, no "fixups".
// observed (optional): { position, normal?, source? } — the pipeline's actual
// output for this vertex (e.g. GPU readback); the residual against it proves
// the traced evidence explains what was really rendered.
export function traceSkinVertex(asset, skinMats, normalMats, v, observed = null) {
  const { joints, weights, positions: pos, normals: nrm } = asset.mesh;
  const o = v * MAX_INFLUENCES;
  const v3 = v * 3;
  const p = [pos[v3], pos[v3 + 1], pos[v3 + 2]];
  const n = [nrm[v3], nrm[v3 + 1], nrm[v3 + 2]];

  const influences = [];
  const outP = [0, 0, 0];
  const outN = [0, 0, 0];
  let totalW = 0;
  let negativeWeights = false;
  for (let k = 0; k < MAX_INFLUENCES; k++) {
    const w = weights[o + k];
    if (w < 0) negativeWeights = true;
    if (w === 0) continue;
    totalW += w;
    const j = joints[o + k];
    const m = j * 16;
    const nm = j * 9;
    const point = [
      skinMats[m] * p[0] + skinMats[m + 4] * p[1] + skinMats[m + 8] * p[2] + skinMats[m + 12],
      skinMats[m + 1] * p[0] + skinMats[m + 5] * p[1] + skinMats[m + 9] * p[2] + skinMats[m + 13],
      skinMats[m + 2] * p[0] + skinMats[m + 6] * p[1] + skinMats[m + 10] * p[2] + skinMats[m + 14],
    ];
    const nPoint = [
      normalMats[nm] * n[0] + normalMats[nm + 3] * n[1] + normalMats[nm + 6] * n[2],
      normalMats[nm + 1] * n[0] + normalMats[nm + 4] * n[1] + normalMats[nm + 7] * n[2],
      normalMats[nm + 2] * n[0] + normalMats[nm + 5] * n[1] + normalMats[nm + 8] * n[2],
    ];
    outP[0] += w * point[0];
    outP[1] += w * point[1];
    outP[2] += w * point[2];
    outN[0] += w * nPoint[0];
    outN[1] += w * nPoint[1];
    outN[2] += w * nPoint[2];
    influences.push({
      slot: k,
      joint: j,
      name: asset.jointNames?.[j] ?? `joint_${j}`,
      weight: w,
      point,
      contribution: [w * point[0], w * point[1], w * point[2]],
      normalContribution: [w * nPoint[0], w * nPoint[1], w * nPoint[2]],
      matrix: Array.from(skinMats.slice(m, m + 16)),
      matrixAnisotropy: matrixAnisotropy(skinMats, m),
    });
  }

  let branch;
  let normalBranch;
  let position;
  let normal;
  if (totalW === 0) {
    // Zero-influence vertices stay rigid to the mesh node (identity skin).
    branch = 'zero-influence';
    normalBranch = 'rigid';
    position = p;
    normal = n;
  } else {
    branch = 'lbs';
    position = outP;
    const len = vec3.length(outN);
    if (len > 0) {
      normalBranch = 'blend-normalized';
      normal = vec3.scale(outN, 1 / len);
    } else {
      // Degenerate blend (opposing normals cancel): keep the authored normal.
      normalBranch = 'degenerate-authored';
      normal = n;
    }
  }

  let spread = 0;
  let maxJointAngle = 0;
  for (let a = 0; a < influences.length; a++) {
    for (let b = a + 1; b < influences.length; b++) {
      spread = Math.max(spread, vec3.length(vec3.sub(influences[a].point, influences[b].point)));
      maxJointAngle = Math.max(maxJointAngle,
        rotationAngleBetween(influences[a].matrix, influences[b].matrix));
    }
  }

  let residual = null;
  let normalResidual = null;
  let observedSource = null;
  if (observed && observed.position) {
    residual = maxComponentDiff(position, observed.position);
    normalResidual = observed.normal ? maxComponentDiff(normal, observed.normal) : null;
    observedSource = observed.source ?? 'unknown';
  }

  return {
    vertex: v,
    branch,
    normalBranch,
    weightSum: totalW,
    negativeWeights,
    loadRepairs: loadRepairsOf(asset, v),
    influences,
    position,
    normal,
    residual,
    normalResidual,
    observedSource,
    spread,
    maxJointAngle,
    nonUniformScale: influences.some((i) => i.matrixAnisotropy > ANISOTROPY_TOL),
  };
}

// ---------- cause classification ----------

// Classifies a traced vertex. ctx:
//   cpuGpuResidual: per-vertex |CPU-GPU| from the numeric check (null = not run)
//   scaleProbeOk:   runScaleProbe().coordinateOk (null = probe not run)
//   residualTol / angleTol / spreadTol: thresholds
// The evidence object always carries the measured numbers, including when the
// verdict is "none" — the panel shows measurements, never assumed defaults.
export function classifyVertexTrace(trace, ctx = {}) {
  const {
    cpuGpuResidual = null,
    residualTol = RESIDUAL_TOL,
    angleTol = ANGLE_TOL,
    spreadTol = SPREAD_TOL,
    scaleProbeOk = null,
  } = ctx;
  const evidence = {
    weightSum: trace.weightSum,
    maxJointAngleDeg: (trace.maxJointAngle * 180) / Math.PI,
    spread: trace.spread,
    nonUniformScale: trace.nonUniformScale,
    cpuGpuResidual,
    scaleProbeOk,
  };

  if (trace.branch !== 'zero-influence'
      && (trace.negativeWeights || Math.abs(trace.weightSum - 1) > WEIGHT_SUM_TOL)) {
    return {
      kind: 'weights', stage: 'load', evidence,
      summary: '权重和不为 1 或存在负权重（加载阶段未按政策修复）；诊断按存储值原样计算，未做归一化',
    };
  }
  if (cpuGpuResidual !== null && cpuGpuResidual > residualTol) {
    return {
      kind: 'numerical', stage: 'skin-gpu', evidence,
      summary: 'CPU 与 GPU 蒙皮结果超出容差：float32 / 定点编码数值分歧',
    };
  }
  if (trace.branch === 'zero-influence') {
    return {
      kind: 'none', stage: null, evidence,
      summary: '零影响顶点：刚性跟随网格节点（政策行为）',
    };
  }
  if (trace.maxJointAngle > angleTol && trace.spread > spreadTol) {
    return {
      kind: 'algorithmic', stage: 'skin-cpu', evidence,
      summary: '影响关节间相对旋转大且刚性结果散布大：LBS 在发散刚性结果间线性插值，产生固有塌陷 / 拉扯',
    };
  }
  if (trace.nonUniformScale && scaleProbeOk === false) {
    return {
      kind: 'coordinate', stage: 'skin-matrix', evidence,
      summary: '蒙皮矩阵含非均匀缩放且缩放探针未通过：法线 / 坐标处理异常',
    };
  }
  return { kind: 'none', stage: null, evidence, summary: '未见异常' };
}

// ---------- bounded mesh scan ----------

// Maps anomaly kinds to the pipeline stage they originate from. When several
// kinds fire on one vertex the earliest stage wins (root-cause direction).
function stageOfKinds(kinds, refStage, trace) {
  if (kinds.includes('non-finite')) {
    return Number.isFinite(trace.weightSum) ? 'skin-matrix' : 'load';
  }
  if (kinds.includes('weight-sum') || kinds.includes('negative-weight')) return 'load';
  if (kinds.includes('residual')) return refStage;
  return null;
}

// Scans the whole mesh and localizes anomalies to pipeline stages.
// Cost is bounded by contract: O(N) compute, but only O(1) aggregates plus
// at most `maxRecords` full per-vertex traces are kept as evidence.
// opts.reference: { positions, stage } — actual pipeline output to compare
// against (typically the GPU readback, stage 'skin-gpu'); omit to scan only
// stored-weight evidence (residuals are then reported as unavailable, not 0).
export function scanMesh(asset, skinMats, normalMats, opts = {}) {
  const {
    reference = null,
    residualTol = RESIDUAL_TOL,
    weightSumTol = WEIGHT_SUM_TOL,
    maxRecords = 16,
    maxCandidates = 1024,
  } = opts;
  const n = asset.mesh.vertexCount;
  const refStage = reference?.stage ?? 'skin-gpu';

  const branchCounts = { lbs: 0, 'zero-influence': 0 };
  let degenerateNormals = 0;
  const maxWeightSumDeviation = { value: 0, vertex: -1 };
  const maxResidual = reference ? { value: 0, vertex: -1 } : null;
  const anomalyCounts = { 'weight-sum': 0, 'negative-weight': 0, 'non-finite': 0, residual: 0 };
  const candidates = [];
  let candidatesDropped = 0;

  for (let v = 0; v < n; v++) {
    const observed = reference
      ? { position: reference.positions.subarray(v * 3, v * 3 + 3), source: refStage }
      : null;
    const tr = traceSkinVertex(asset, skinMats, normalMats, v, observed);

    branchCounts[tr.branch]++;
    if (tr.normalBranch === 'degenerate-authored') degenerateNormals++;

    const kinds = [];
    let severity = 0;
    if (!Number.isFinite(tr.weightSum) || !tr.position.every(Number.isFinite)) {
      kinds.push('non-finite');
      severity = Infinity;
    }
    if (tr.negativeWeights) {
      kinds.push('negative-weight');
      severity = Math.max(severity, 1);
    }
    if (tr.branch !== 'zero-influence' && Number.isFinite(tr.weightSum)) {
      const dev = Math.abs(tr.weightSum - 1);
      if (dev > maxWeightSumDeviation.value) {
        maxWeightSumDeviation.value = dev;
        maxWeightSumDeviation.vertex = v;
      }
      if (dev > weightSumTol) {
        kinds.push('weight-sum');
        severity = Math.max(severity, dev / weightSumTol);
      }
    }
    if (tr.residual !== null) {
      if (tr.residual > maxResidual.value) {
        maxResidual.value = tr.residual;
        maxResidual.vertex = v;
      }
      if (tr.residual > residualTol) {
        kinds.push('residual');
        severity = Math.max(severity, tr.residual / residualTol);
      }
    }
    if (kinds.length === 0) continue;
    for (const k of kinds) anomalyCounts[k]++;
    if (candidates.length < maxCandidates) {
      candidates.push({ vertex: v, kinds, severity, trace: tr });
    } else {
      candidatesDropped++;
    }
  }

  candidates.sort((a, b) => b.severity - a.severity || a.vertex - b.vertex);
  const records = candidates.slice(0, maxRecords).map((c) => ({
    vertex: c.vertex,
    kinds: c.kinds,
    severity: c.severity,
    stage: stageOfKinds(c.kinds, refStage, c.trace),
    trace: c.trace,
  }));

  return {
    vertexCount: n,
    branchCounts,
    degenerateNormals,
    maxWeightSumDeviation,
    maxResidual,
    referenceStage: reference ? refStage : null,
    anomalyCounts,
    records,
    candidatesConsidered: candidates.length,
    candidatesDropped,
    maxRecords,
    maxCandidates,
  };
}

// ---------- probes: twist vs non-uniform scale ----------

// Builds a two-joint asset: J0 identity, J1 at (10,0,0) twisted `angleDeg`
// about the longitudinal X axis. Vertex 0 is rigid to J1 (LBS == exact rigid
// transform, so any error is numerical); vertex 1 is blended 50/50 (LBS
// collapses toward the axis by the analytic factor |w0 + w1*e^{iθ}| — the
// algorithmic limitation). Returns measurements plus the vertex traces.
export function runTwistProbe(angleDeg = 180) {
  const angle = (angleDeg * Math.PI) / 180;
  const nodes = [
    { name: 'Root', parent: -1, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    { name: 'J0', parent: 0, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    { name: 'J1', parent: 0, translation: [10, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    { name: 'Mesh', parent: 0, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
  ];
  const worlds = [];
  for (let i = 0; i < nodes.length; i++) {
    const l = mat4.compose(nodes[i].translation, nodes[i].rotation, nodes[i].scale);
    worlds[i] = nodes[i].parent === -1 ? l : mat4.multiply(worlds[nodes[i].parent], l);
  }
  const ibms = new Float32Array(32);
  ibms.set(mat4.multiply(mat4.invert(worlds[1]), worlds[3]), 0);
  ibms.set(mat4.multiply(mat4.invert(worlds[2]), worlds[3]), 16);
  const asset = loadAsset({
    format: 'rskin',
    version: 1,
    nodes,
    skeleton: { joints: [1, 2], inverseBindMatrices: f32ToBase64(ibms) },
    mesh: {
      node: 3,
      // both vertices at radius 5 from the twist axis through J1
      positions: f32ToBase64(new Float32Array([10, 0, 5, 10, 0, 5])),
      normals: f32ToBase64(new Float32Array([0, 0, 1, 0, 0, 1])),
      joints: u16ToBase64(new Uint16Array([1, 0, 0, 0, 0, 1, 0, 0])),
      weights: f32ToBase64(new Float32Array([1, 0, 0, 0, 0.5, 0.5, 0, 0])),
      indicesU16: u16ToBase64(new Uint16Array([0, 1, 0])),
    },
    animations: [],
  });
  const skeleton = buildSkeleton(asset);
  const pose = bindPose(asset);
  pose[2].q = quat.fromAxisAngle([1, 0, 0], angle);
  const world = computeWorldMatrices(asset, skeleton, pose);
  const { skin, normals } = computeSkinMatrices(asset, world);

  const trSingle = traceSkinVertex(asset, skin, normals, 0);
  const trBlend = traceSkinVertex(asset, skin, normals, 1);

  const q = quat.fromAxisAngle([1, 0, 0], angle);
  const expectedSingle = vec3.add([10, 0, 0], quat.rotateVec3(q, [0, 0, 5]));
  const singleJointError = vec3.length(vec3.sub(trSingle.position, expectedSingle));

  const w0 = 0.5;
  const w1 = 0.5;
  const lbsPredicted = [
    w0 * 10 + w1 * expectedSingle[0],
    w0 * 0 + w1 * expectedSingle[1],
    w0 * 5 + w1 * expectedSingle[2],
  ];
  const implError = vec3.length(vec3.sub(trBlend.position, lbsPredicted));
  const collapseAnalytic = Math.hypot(w0 + w1 * Math.cos(angle), w1 * Math.sin(angle));
  const collapseMeasured = vec3.length(vec3.sub(trBlend.position, [10, 0, 0])) / 5;

  return {
    angleDeg,
    singleJointError,
    blended: {
      measured: trBlend.position,
      lbsPredicted,
      implError,
      collapseMeasured,
      collapseAnalytic,
    },
    traces: { single: trSingle, blended: trBlend },
    verdict: {
      numericalBound: singleJointError,
      implementationMatchesLBS: implError < 1e-6,
      algorithmicCollapse: collapseAnalytic,
    },
  };
}

// Builds a joint chain whose posed skin matrix contains non-uniform scale /
// shear (parent scale [2, 0.5, 1], joint rotated 90 deg about Z). A rigidly
// bound vertex checks the two things that must hold for any invertible
// affine skin matrix: positions transform exactly, and the inverse-transpose
// normal matrix keeps the normal perpendicular to transformed tangents.
export function runScaleProbe() {
  const nodes = [
    { name: 'Root', parent: -1, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    { name: 'Squeezer', parent: 0, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [2, 0.5, 1] },
    { name: 'J0', parent: 1, translation: [1, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    { name: 'Mesh', parent: 0, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
  ];
  const worlds = [];
  for (let i = 0; i < nodes.length; i++) {
    const l = mat4.compose(nodes[i].translation, nodes[i].rotation, nodes[i].scale);
    worlds[i] = nodes[i].parent === -1 ? l : mat4.multiply(worlds[nodes[i].parent], l);
  }
  const ibm = mat4.multiply(mat4.invert(worlds[2]), worlds[3]);
  const asset = loadAsset({
    format: 'rskin',
    version: 1,
    nodes,
    skeleton: { joints: [2], inverseBindMatrices: f32ToBase64(new Float32Array(ibm)) },
    mesh: {
      node: 3,
      positions: f32ToBase64(new Float32Array([3, 1, 0])),
      normals: f32ToBase64(new Float32Array([1, 0, 0])),
      joints: u16ToBase64(new Uint16Array([0, 0, 0, 0])),
      weights: f32ToBase64(new Float32Array([1, 0, 0, 0])),
      indicesU16: u16ToBase64(new Uint16Array([0, 0, 0])),
    },
    animations: [],
  });
  const skeleton = buildSkeleton(asset);
  const pose = bindPose(asset);
  pose[2].q = quat.fromAxisAngle([0, 0, 1], Math.PI / 2);
  const world = computeWorldMatrices(asset, skeleton, pose);
  const { skin, normals } = computeSkinMatrices(asset, world);

  const trace = traceSkinVertex(asset, skin, normals, 0);
  const sm = trace.influences[0].matrix;
  const expected = mat4.transformPoint(sm, [3, 1, 0]);
  const positionError = vec3.length(vec3.sub(trace.position, expected));
  const tangent = mat4.transformDir(sm, [0, 1, 0]);
  const normalTangentDot = vec3.dot(vec3.normalize(tangent), trace.normal);

  return {
    positionError,
    normalTangentDot,
    anisotropy: trace.influences[0].matrixAnisotropy,
    coordinateOk: positionError < 1e-9 && Math.abs(normalTangentDot) < 1e-9,
    trace,
  };
}

// ---------- replay capture ----------

// Captures a small, self-contained replay package: the posed node graph
// (TRS per node), joint list, inverse bind matrices, and only the selected
// vertices' stored skinning inputs. Re-running it re-executes the whole
// chain (pose -> world -> skin matrices -> per-vertex trace) deterministically.
export function captureReplay(asset, pose, vertexIds, meta = {}) {
  const nodes = asset.nodes.map((nd, i) => ({
    name: nd.name,
    parent: nd.parent,
    t: pose[i].t.slice(),
    q: pose[i].q.slice(),
    s: pose[i].s.slice(),
  }));
  const vertices = vertexIds.map((v) => {
    const o = v * MAX_INFLUENCES;
    const v3 = v * 3;
    return {
      index: v,
      position: Array.from(asset.mesh.positions.slice(v3, v3 + 3)),
      normal: Array.from(asset.mesh.normals.slice(v3, v3 + 3)),
      joints: Array.from(asset.mesh.joints.slice(o, o + MAX_INFLUENCES)),
      weights: Array.from(asset.mesh.weights.slice(o, o + MAX_INFLUENCES)),
    };
  });
  const replay = {
    type: REPLAY_TYPE,
    version: REPLAY_VERSION,
    meta: {
      asset: meta.asset ?? asset.name ?? null,
      clip: meta.clip ?? null,
      time: meta.time ?? null,
    },
    jointNames: asset.jointNames.slice(),
    nodes,
    joints: Array.from(asset.jointNodes),
    meshNode: asset.mesh.node,
    ibms: f32ToBase64(asset.inverseBindMatrices),
    vertices,
  };
  replay.digest = replayTrace(replay).digest;
  return replay;
}

// Re-executes a replay package. Deterministic: same package -> same traces
// -> same digest. Skin matrices are stored as float32 exactly like the live
// pipeline (computeSkinMatrices), so replayed traces match live traces.
export function replayTrace(replay) {
  if (!replay || replay.type !== REPLAY_TYPE) {
    throw new Error('不是有效的蒙皮重放包（type 不匹配）');
  }
  if (replay.version !== REPLAY_VERSION) {
    throw new Error(`重放包版本 ${replay.version} 不受支持（期望 ${REPLAY_VERSION}）`);
  }
  replay.nodes.forEach((nd, i) => {
    if (!Number.isInteger(nd.parent) || nd.parent < -1 || nd.parent >= i) {
      throw new Error(`重放包 nodes[${i}] 父索引 ${nd.parent} 破坏拓扑序`);
    }
  });
  const world = new Array(replay.nodes.length);
  replay.nodes.forEach((nd, i) => {
    const local = mat4.compose(nd.t, nd.q, nd.s);
    world[i] = nd.parent === -1 ? local : mat4.multiply(world[nd.parent], local);
  });
  const ibms = base64ToF32(replay.ibms);
  const invMeshWorld = mat4.invert(world[replay.meshNode]);
  if (!invMeshWorld) throw new Error('重放包网格节点世界矩阵奇异');
  const jointCount = replay.joints.length;
  const skin = new Float32Array(jointCount * 16);
  const normals = new Float32Array(jointCount * 9);
  for (let j = 0; j < jointCount; j++) {
    const jw = world[replay.joints[j]];
    const ibm = Array.from(ibms.subarray(j * 16, j * 16 + 16));
    const sm = mat4.multiply(mat4.multiply(invMeshWorld, jw), ibm);
    skin.set(sm, j * 16);
    const nm = mat4.normalMatrix(sm);
    if (!nm) throw new Error(`重放包关节 ${j} 的蒙皮矩阵奇异`);
    normals.set(nm, j * 9);
  }
  // Shim asset over just the captured vertices. No weightRepairs: the replay
  // stores post-load weights as they were used; load-stage repair evidence
  // stays with the original asset, so loadRepairs is null (shown as such).
  const shim = {
    jointNames: replay.jointNames,
    mesh: {
      vertexCount: replay.vertices.length,
      positions: replay.vertices.flatMap((rv) => rv.position),
      normals: replay.vertices.flatMap((rv) => rv.normal),
      joints: replay.vertices.flatMap((rv) => rv.joints),
      weights: replay.vertices.flatMap((rv) => rv.weights),
    },
  };
  const traces = replay.vertices.map((rv, i) => traceSkinVertex(shim, skin, normals, i));
  traces.forEach((tr, i) => { tr.vertex = replay.vertices[i].index; });
  return { traces, digest: digestOfTraces(traces, replay.meta), meta: replay.meta };
}

// ---------- deterministic digest ----------

function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

function canon(x) {
  if (x === null || x === undefined) return '∅';
  if (typeof x === 'number') return Number.isFinite(x) ? x.toPrecision(12) : String(x);
  return String(x);
}

// Stable digest over the evidence that determines the traces: meta plus
// per-vertex branch, stored weight sum, and skinned position/normal.
export function digestOfTraces(traces, meta = null) {
  const parts = [];
  if (meta) parts.push(canon(meta.asset), canon(meta.clip), canon(meta.time));
  for (const tr of traces) {
    parts.push(
      tr.vertex,
      tr.branch,
      tr.normalBranch,
      canon(tr.weightSum),
      ...tr.position.map(canon),
      ...tr.normal.map(canon),
    );
  }
  return `${traces.length}:${fnv1a(parts.join('|'))}`;
}

// ---------- bounded trace log ----------

// Fixed-capacity ring for stage-tagged diagnostic events. Pushes are O(1)
// amortized, memory is O(capacity), and dropped entries are counted so the
// cost limit itself is observable.
export class TraceLog {
  constructor(capacity = 64) {
    this.capacity = capacity;
    this.records = [];
    this.dropped = 0;
    this.total = 0;
  }

  push(stage, kind, detail) {
    this.total++;
    const rec = { seq: this.total, stage, kind, detail };
    if (this.records.length === this.capacity) {
      this.records.shift();
      this.dropped++;
    }
    this.records.push(rec);
    return rec;
  }

  latest(n = this.records.length) {
    return this.records.slice(-n);
  }
}
