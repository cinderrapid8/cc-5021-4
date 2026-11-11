// Parses a distributable .rskin.json document into a validated runtime asset.
//
// Space conventions (see README.md):
//   - "model space" is the space the mesh positions and inverse bind matrices
//     are authored in. Skinning runs entirely in model space.
//   - "world space" is reached through the mesh node's world matrix:
//         worldPos = meshWorld * (sum_j w_j * skinMatrix_j * modelPos)
//         skinMatrix_j = inverse(meshWorld) * jointWorld_j * inverseBind_j
//     This is exact for any node transforms above or beside the skeleton,
//     including non-uniform scale, as long as every matrix is invertible.

import {
  FORMAT_ID, FORMAT_VERSION, MAX_INFLUENCES,
  base64ToF32, base64ToU16, base64ToU32,
} from './format.js';
import { mat4, quat, isFiniteArray } from './math.js';
import {
  AssetError, assertFinite, validateInverseBindMatrices, validateWeights,
} from './validate.js';

function requireField(obj, name, where) {
  if (!(name in obj)) throw new AssetError(`${where}: missing field "${name}"`);
  return obj[name];
}

function parseNode(raw, i) {
  const where = `nodes[${i}]`;
  const t = raw.translation ?? [0, 0, 0];
  const r = raw.rotation ?? [0, 0, 0, 1];
  const s = raw.scale ?? [1, 1, 1];
  if (t.length !== 3 || r.length !== 4 || s.length !== 3) {
    throw new AssetError(`${where}: translation/rotation/scale must have lengths 3/4/3`);
  }
  if (!isFiniteArray(t) || !isFiniteArray(r) || !isFiniteArray(s)) {
    throw new AssetError(`${where}: TRS contains NaN/Infinity`);
  }
  const qn = quat.normalize(r);
  return {
    name: raw.name ?? `node_${i}`,
    parent: raw.parent ?? -1,
    translation: t.slice(),
    rotation: qn,
    scale: s.slice(),
  };
}

export function loadAsset(json) {
  const warnings = [];
  if (json.format !== FORMAT_ID) {
    throw new AssetError(`unsupported format "${json.format}" (expected "${FORMAT_ID}")`);
  }
  if (json.version !== FORMAT_VERSION) {
    throw new AssetError(`unsupported version ${json.version} (expected ${FORMAT_VERSION})`);
  }

  // ---- nodes ----
  const rawNodes = requireField(json, 'nodes', 'asset');
  const nodes = rawNodes.map(parseNode);
  nodes.forEach((n, i) => {
    if (n.parent >= i) {
      throw new AssetError(`nodes[${i}]: parent index ${n.parent} must be < ${i} (topological order, no cycles)`);
    }
    if (n.parent < -1 || n.parent >= nodes.length) {
      throw new AssetError(`nodes[${i}]: parent index ${n.parent} out of range`);
    }
  });

  // ---- skeleton ----
  const rawSkel = requireField(json, 'skeleton', 'asset');
  const jointNodes = Array.from(requireField(rawSkel, 'joints', 'skeleton'));
  const jointCount = jointNodes.length;
  if (jointCount === 0) throw new AssetError('skeleton: joints must not be empty');
  jointNodes.forEach((ni, j) => {
    if (!Number.isInteger(ni) || ni < 0 || ni >= nodes.length) {
      throw new AssetError(`skeleton.joints[${j}]: node index ${ni} out of range`);
    }
  });
  const ibms = base64ToF32(requireField(rawSkel, 'inverseBindMatrices', 'skeleton'));
  if (ibms.length !== jointCount * 16) {
    throw new AssetError(`skeleton.inverseBindMatrices: expected ${jointCount * 16} floats, got ${ibms.length}`);
  }
  validateInverseBindMatrices(ibms, jointCount);

  // ---- mesh ----
  const rawMesh = requireField(json, 'mesh', 'asset');
  const meshNode = requireField(rawMesh, 'node', 'mesh');
  if (!Number.isInteger(meshNode) || meshNode < 0 || meshNode >= nodes.length) {
    throw new AssetError(`mesh.node: node index ${meshNode} out of range`);
  }
  const positions = base64ToF32(requireField(rawMesh, 'positions', 'mesh'));
  if (positions.length % 3 !== 0) throw new AssetError('mesh.positions: not a multiple of 3');
  const vertexCount = positions.length / 3;
  const normals = base64ToF32(requireField(rawMesh, 'normals', 'mesh'));
  if (normals.length !== positions.length) throw new AssetError('mesh.normals: length mismatch with positions');
  assertFinite(positions, 'mesh.positions');
  assertFinite(normals, 'mesh.normals');

  const jointsArr = base64ToU16(requireField(rawMesh, 'joints', 'mesh'));
  const weightsArr = base64ToF32(requireField(rawMesh, 'weights', 'mesh'));
  if (jointsArr.length !== vertexCount * MAX_INFLUENCES || weightsArr.length !== vertexCount * MAX_INFLUENCES) {
    throw new AssetError(`mesh.joints/weights: expected ${vertexCount * MAX_INFLUENCES} entries each`);
  }
  const weightResult = validateWeights(jointsArr, weightsArr, vertexCount, jointCount);
  warnings.push(...weightResult.warnings);

  const indices = rawMesh.indicesU16
    ? base64ToU16(rawMesh.indicesU16)
    : base64ToU32(requireField(rawMesh, 'indicesU32', 'mesh'));
  for (let i = 0; i < indices.length; i++) {
    if (indices[i] >= vertexCount) {
      throw new AssetError(`mesh.indices[${i}] = ${indices[i]} out of range (vertexCount=${vertexCount})`);
    }
  }

  // ---- animations ----
  const animations = (json.animations ?? []).map((raw, ai) => {
    const where = `animations[${ai}]`;
    const channels = requireField(raw, 'channels', where).map((rc, ci) => {
      const cwhere = `${where}.channels[${ci}]`;
      const node = requireField(rc, 'node', cwhere);
      if (!Number.isInteger(node) || node < 0 || node >= nodes.length) {
        throw new AssetError(`${cwhere}: node index ${node} out of range`);
      }
      const path = requireField(rc, 'path', cwhere);
      const comp = { translation: 3, rotation: 4, scale: 3 }[path];
      if (!comp) throw new AssetError(`${cwhere}: unknown path "${path}"`);
      const times = base64ToF32(requireField(rc, 'times', cwhere));
      const values = base64ToF32(requireField(rc, 'values', cwhere));
      if (times.length === 0) throw new AssetError(`${cwhere}: empty times`);
      if (values.length !== times.length * comp) {
        throw new AssetError(`${cwhere}: values length ${values.length}, expected ${times.length} * ${comp}`);
      }
      assertFinite(times, `${cwhere}.times`);
      assertFinite(values, `${cwhere}.values`);
      for (let k = 1; k < times.length; k++) {
        if (times[k] <= times[k - 1]) throw new AssetError(`${cwhere}: times must be strictly increasing`);
      }
      const interpolation = rc.interpolation ?? 'LINEAR';
      if (interpolation !== 'LINEAR' && interpolation !== 'STEP') {
        throw new AssetError(`${cwhere}: unknown interpolation "${interpolation}"`);
      }
      return { node, path, times, values, interpolation, components: comp };
    });
    return { name: raw.name ?? `clip_${ai}`, channels };
  });

  const asset = {
    name: json.name ?? 'unnamed',
    nodes,
    jointNodes,
    jointNames: jointNodes.map((ni) => nodes[ni].name),
    inverseBindMatrices: ibms,
    mesh: {
      name: rawMesh.name ?? 'mesh',
      node: meshNode,
      vertexCount,
      positions,
      normals,
      joints: weightResult.joints,
      weights: weightResult.weights,
      indices,
      zeroInfluence: weightResult.zeroInfluence,
      weightRepairs: weightResult.repairs,
    },
    animations,
    warnings,
  };
  return asset;
}

// ---- pose evaluation ----

// Local bind matrices and parent list, precomputed once.
export function buildSkeleton(asset) {
  const n = asset.nodes.length;
  const locals = new Array(n);
  const parents = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const nd = asset.nodes[i];
    locals[i] = mat4.compose(nd.translation, nd.rotation, nd.scale);
    parents[i] = nd.parent;
  }
  return { locals, parents, count: n };
}

// World matrix for every node given per-node local TRS overrides (posed).
// pose: array of {t, q, s} per node (already animation-applied).
export function computeWorldMatrices(asset, skeleton, pose) {
  const n = skeleton.count;
  const world = new Array(n);
  for (let i = 0; i < n; i++) {
    const p = pose[i];
    const local = mat4.compose(p.t, p.q, p.s);
    const parent = skeleton.parents[i];
    world[i] = parent === -1 ? local : mat4.multiply(world[parent], local);
  }
  return world;
}

// Per-joint skinning matrices in model space, plus per-joint 3x3 normal
// matrices (inverse-transpose, exact under non-uniform scale and shear).
// Returns { skin: Float32Array(16*J), normals: Float32Array(9*J), meshWorld }
export function computeSkinMatrices(asset, world) {
  const jointCount = asset.jointNodes.length;
  const meshWorld = world[asset.mesh.node];
  const invMeshWorld = mat4.invert(meshWorld);
  if (!invMeshWorld) {
    throw new AssetError('mesh node world matrix is singular at this pose');
  }
  const skin = new Float32Array(jointCount * 16);
  const normals = new Float32Array(jointCount * 9);
  for (let j = 0; j < jointCount; j++) {
    const jw = world[asset.jointNodes[j]];
    const ibm = Array.from(asset.inverseBindMatrices.subarray(j * 16, j * 16 + 16));
    const sm = mat4.multiply(mat4.multiply(invMeshWorld, jw), ibm);
    skin.set(sm, j * 16);
    const nm = mat4.normalMatrix(sm);
    if (!nm) throw new AssetError(`skin matrix of joint ${j} is singular at this pose`);
    normals.set(nm, j * 9);
  }
  return { skin, normals, meshWorld };
}
