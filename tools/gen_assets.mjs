// Generates assets/runway_character.rskin.json — a procedural runway character.
//
// The node graph deliberately contains non-unit parent transforms so loaders
// cannot assume identity hierarchies:
//   Stage   (translation + 35° yaw, world placement)
//   UnitFix (non-uniform cm->m scale [0.011, 0.01, 0.0095])
//   RigRoot -> skeleton joints (bind translations in cm)
//   BodyMesh (mesh node, child of Stage with its OWN non-uniform scale,
//             not shared with the joint chain)
// Inverse bind matrices are computed through this real graph, and mesh
// positions are authored in BodyMesh-local space (centimeters).

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { mat4, quat } from '../src/core/math.js';
import {
  FORMAT_ID, FORMAT_VERSION, MAX_INFLUENCES,
  f32ToBase64, u16ToBase64,
} from '../src/core/format.js';

const DEG = Math.PI / 180;

// ---------- skeleton (model space: cm, Y-up, character faces +Z, T-pose) ----------

const JOINT_DEFS = [
  ['Hips', null, [0, 95, 0]],
  ['Spine', 'Hips', [0, 108, 0]],
  ['Chest', 'Spine', [0, 125, 0]],
  ['Neck', 'Chest', [0, 148, 0]],
  ['Head', 'Neck', [0, 160, 0]],
  ['L_UpperArm', 'Chest', [14, 142, 0]],
  ['L_Forearm', 'L_UpperArm', [40, 142, 0]],
  ['L_Hand', 'L_Forearm', [66, 142, 0]],
  ['R_UpperArm', 'Chest', [-14, 142, 0]],
  ['R_Forearm', 'R_UpperArm', [-40, 142, 0]],
  ['R_Hand', 'R_Forearm', [-66, 142, 0]],
  ['L_UpLeg', 'Hips', [9, 92, 0]],
  ['L_LoLeg', 'L_UpLeg', [9, 48, 0]],
  ['L_Foot', 'L_LoLeg', [9, 8, 2]],
  ['R_UpLeg', 'Hips', [-9, 92, 0]],
  ['R_LoLeg', 'R_UpLeg', [-9, 48, 0]],
  ['R_Foot', 'R_LoLeg', [-9, 8, 2]],
];
const JOINT_INDEX = Object.fromEntries(JOINT_DEFS.map(([name], i) => [name, i]));

const MESH_SCALE = [0.011, 0.01, 0.0095]; // non-uniform cm->m, slightly anisotropic

// ---------- node graph ----------

function buildNodes() {
  const nodes = [];
  nodes.push({
    name: 'Stage', parent: -1,
    translation: [1.5, 0, -2.0],
    rotation: quat.fromAxisAngle([0, 1, 0], 35 * DEG),
    scale: [1, 1, 1],
  });
  nodes.push({
    name: 'UnitFix', parent: 0,
    translation: [0, 0, 0], rotation: [0, 0, 0, 1],
    scale: MESH_SCALE.slice(),
  });
  nodes.push({
    name: 'RigRoot', parent: 1,
    translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1],
  });
  const jointNodeBase = nodes.length;
  const nodeOfJoint = new Map();
  JOINT_DEFS.forEach(([name, parentName, pos], j) => {
    const nodeIdx = nodes.length;
    nodeOfJoint.set(name, nodeIdx);
    const parentNode = parentName === null ? 2 : nodeOfJoint.get(parentName);
    // local translation = bind pos - parent bind pos (bind rotations are identity)
    const parentPos = parentName === null ? [0, 0, 0] : JOINT_DEFS[JOINT_INDEX[parentName]][2];
    nodes.push({
      name, parent: parentNode,
      translation: [pos[0] - parentPos[0], pos[1] - parentPos[1], pos[2] - parentPos[2]],
      rotation: [0, 0, 0, 1],
      scale: [1, 1, 1],
    });
  });
  const meshNode = nodes.length;
  nodes.push({
    name: 'BodyMesh', parent: 0, // child of Stage, own non-uniform scale
    translation: [0, 0, 0], rotation: [0, 0, 0, 1],
    scale: MESH_SCALE.slice(),
  });
  return { nodes, jointNodeBase, nodeOfJoint, meshNode };
}

// nodes are stored topologically ordered (parent index < child index)
function computeWorlds(nodes) {
  const world = new Array(nodes.length);
  for (let i = 0; i < nodes.length; i++) {
    const nd = nodes[i];
    const local = mat4.compose(nd.translation, nd.rotation, nd.scale);
    world[i] = nd.parent === -1 ? local : mat4.multiply(world[nd.parent], local);
  }
  return world;
}

// ---------- mesh building ----------

const positions = [];
const normals = [];
const weights = []; // per vertex: [[jointIdx, w], ...]
const indices = [];

function smoothstep(a, b, x) {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
}

function addVertex(p, n, w) {
  positions.push(p[0], p[1], p[2]);
  const nl = Math.hypot(n[0], n[1], n[2]) || 1;
  normals.push(n[0] / nl, n[1] / nl, n[2] / nl);
  weights.push(w);
  return positions.length / 3 - 1;
}

function stitchRows(rows, segments) {
  for (let r = 0; r < rows.length - 1; r++) {
    for (let s = 0; s < segments; s++) {
      const i0 = rows[r][s];
      const i1 = rows[r][s + 1];
      const i2 = rows[r + 1][s];
      const i3 = rows[r + 1][s + 1];
      indices.push(i0, i2, i1, i1, i2, i3);
    }
  }
}

const J = (name) => JOINT_INDEX[name];

function buildTorso() {
  const segments = 14;
  const rings = [];
  for (let y = 92; y <= 152; y += 4) rings.push(y);
  const rows = [];
  for (const y of rings) {
    const t = (y - 92) / 60;
    const rx = 16 - 3.5 * t;
    const rz = 10 - 2.5 * t;
    const row = [];
    for (let s = 0; s <= segments; s++) {
      const a = (s / segments) * Math.PI * 2;
      const p = [Math.cos(a) * rx, y, Math.sin(a) * rz];
      const n = [Math.cos(a) / rx, 0, Math.sin(a) / rz];
      row.push(addVertex(p, n, torsoWeights(y)));
    }
    rows.push(row);
  }
  stitchRows(rows, segments);
}

function torsoWeights(y) {
  // piecewise-smooth blends along the spine
  const bands = [
    [92, 100, ['Hips'], ['Hips']],
    [100, 112, ['Hips', 'Spine'], null],
    [112, 120, ['Spine'], ['Spine']],
    [120, 132, ['Spine', 'Chest'], null],
    [132, 142, ['Chest'], ['Chest']],
    [142, 150, ['Chest', 'Neck'], null],
    [150, 152, ['Neck'], ['Neck']],
  ];
  for (const [y0, y1, js] of bands) {
    if (y >= y0 && y <= y1) {
      if (js.length === 1) return [[J(js[0]), 1]];
      const t = smoothstep(y0, y1, y);
      return [[J(js[0]), 1 - t], [J(js[1]), t]];
    }
  }
  return [[J('Neck'), 1]];
}

function buildHead() {
  const center = [0, 166, 0];
  const r = 11;
  const segU = 12;
  const segV = 8;
  const rows = [];
  for (let v = 0; v <= segV; v++) {
    const phi = (v / segV) * Math.PI;
    const row = [];
    for (let u = 0; u <= segU; u++) {
      const theta = (u / segU) * Math.PI * 2;
      const n = [
        Math.sin(phi) * Math.cos(theta),
        Math.cos(phi),
        Math.sin(phi) * Math.sin(theta),
      ];
      row.push(addVertex(
        [center[0] + n[0] * r, center[1] + n[1] * r, center[2] + n[2] * r],
        n,
        [[J('Head'), 1]],
      ));
    }
    rows.push(row);
  }
  stitchRows(rows, segU);
}

function buildArm(side) {
  const sgn = side === 'L' ? 1 : -1;
  const up = `${side}_UpperArm`;
  const fore = `${side}_Forearm`;
  const hand = `${side}_Hand`;
  const segments = 10;
  const rings = [];
  for (let x = 10; x <= 72; x += 2) rings.push(x);
  const rows = [];
  for (const x of rings) {
    const rad = x <= 40 ? 5 : x <= 66 ? 4.5 : 4;
    const row = [];
    for (let s = 0; s <= segments; s++) {
      const a = (s / segments) * Math.PI * 2;
      const n = [0, Math.cos(a), Math.sin(a)];
      const p = [sgn * x, 142 + n[1] * rad, n[2] * rad];
      row.push(addVertex(p, n, armWeights(x, up, fore, hand)));
    }
    rows.push(row);
  }
  stitchRows(rows, segments);
}

function armWeights(x, up, fore, hand) {
  if (x <= 18) {
    const t = smoothstep(10, 18, x);
    return [[J('Chest'), 1 - t], [J(up), t]];
  }
  if (x <= 34) return [[J(up), 1]];
  if (x <= 48) {
    const t = smoothstep(34, 48, x);
    return [[J(up), 1 - t], [J(fore), t]];
  }
  if (x <= 64) return [[J(fore), 1]];
  const t = smoothstep(64, 72, x);
  return [[J(fore), 1 - t], [J(hand), t]];
}

function buildLeg(side) {
  const sgn = side === 'L' ? 1 : -1;
  const up = `${side}_UpLeg`;
  const lo = `${side}_LoLeg`;
  const foot = `${side}_Foot`;
  const segments = 10;
  const rings = [];
  for (let y = 92; y >= 4; y -= 4) rings.push(y);
  const rows = [];
  for (const y of rings) {
    const rad = y > 48 ? 7 : 5.5;
    const row = [];
    for (let s = 0; s <= segments; s++) {
      const a = (s / segments) * Math.PI * 2;
      const n = [Math.cos(a), 0, Math.sin(a)];
      const p = [sgn * 9 + n[0] * rad, y, n[2] * rad + (y < 12 ? 2 : 0)];
      row.push(addVertex(p, n, legWeights(y, up, lo, foot)));
    }
    rows.push(row);
  }
  stitchRows(rows, segments);
}

function legWeights(y, up, lo, foot) {
  if (y >= 56) return [[J(up), 1]];
  if (y >= 40) {
    const t = smoothstep(56, 40, y);
    return [[J(up), 1 - t], [J(lo), t]];
  }
  if (y >= 12) return [[J(lo), 1]];
  const t = smoothstep(12, 4, y);
  return [[J(lo), 1 - t], [J(foot), t]];
}

// A skirt: the "garment" of the runway character. Weights blend between Hips
// and the two thigh joints by angular proximity, ramped by height, so the hem
// follows the legs — classic clothing skinning.
function buildSkirt() {
  const segments = 20;
  const rows = [];
  const rings = [];
  for (let y = 102; y >= 58; y -= 4.4) rings.push(y);
  for (const y of rings) {
    const t = (102 - y) / 44;
    const r = 19 + 15 * t;
    const row = [];
    for (let s = 0; s <= segments; s++) {
      const a = (s / segments) * Math.PI * 2;
      const p = [Math.cos(a) * r, y, Math.sin(a) * r];
      const n = [Math.cos(a), 0.35, Math.sin(a)]; // slanted cone normal
      row.push(addVertex(p, n, skirtWeights(a, t)));
    }
    rows.push(row);
  }
  stitchRows(rows, segments);
}

function skirtWeights(angle, heightT) {
  // legs sit at angle 0 (+X, left) and PI (-X, right)
  const dL = Math.abs(((angle + Math.PI) % (2 * Math.PI)) - Math.PI); // dist to 0
  const dL0 = Math.min(dL, Math.abs(dL - 2 * Math.PI));
  const dR = Math.abs(angle - Math.PI);
  const sig = 0.9;
  const wL = Math.exp(-(dL0 * dL0) / (2 * sig * sig));
  const wR = Math.exp(-(dR * dR) / (2 * sig * sig));
  const legInfluence = heightT * 0.85; // hem follows legs, waist follows hips
  return [
    [J('Hips'), 1 - legInfluence],
    [J('L_UpLeg'), legInfluence * wL / (wL + wR)],
    [J('R_UpLeg'), legInfluence * wR / (wL + wR)],
  ];
}

// ---------- weight packing ----------

function packWeights() {
  const n = weights.length;
  const joints = new Uint16Array(n * MAX_INFLUENCES);
  const wts = new Float32Array(n * MAX_INFLUENCES);
  for (let v = 0; v < n; v++) {
    const list = weights[v].filter(([, w]) => w > 1e-6);
    if (list.length > MAX_INFLUENCES) {
      list.sort((a, b) => b[1] - a[1]);
      list.length = MAX_INFLUENCES;
    }
    let sum = 0;
    for (const [, w] of list) sum += w;
    if (sum === 0) continue; // zero-influence: stays all zero (policy)
    for (let k = 0; k < list.length; k++) {
      joints[v * MAX_INFLUENCES + k] = list[k][0];
      wts[v * MAX_INFLUENCES + k] = list[k][1] / sum;
    }
  }
  return { joints, weights: wts };
}

// ---------- animations ----------

function keyQuat(axis, deg) {
  return quat.fromAxisAngle(axis, deg * DEG);
}

function buildAnimations(nodeOfJoint) {
  const hips = nodeOfJoint.get('Hips');
  const lFore = nodeOfJoint.get('L_Forearm');
  const rUpArm = nodeOfJoint.get('R_UpperArm');

  function rotChannel(node, times, quats) {
    return {
      node, path: 'rotation', interpolation: 'LINEAR',
      times: f32ToBase64(new Float32Array(times)),
      values: f32ToBase64(new Float32Array(quats.flat())),
    };
  }
  function transChannel(node, times, vecs) {
    return {
      node, path: 'translation', interpolation: 'LINEAR',
      times: f32ToBase64(new Float32Array(times)),
      values: f32ToBase64(new Float32Array(vecs.flat())),
    };
  }

  const I = [0, 0, 0, 1];
  // Elbow flex: forearm rotates about local Y (vertical in T-pose) so the
  // hand sweeps forward. 0 -> -115 deg -> 0.
  const elbow_flex = {
    name: 'elbow_flex',
    channels: [
      rotChannel(lFore, [0, 0.5, 1.0], [I, keyQuat([0, 1, 0], -115), I]),
    ],
  };
  // Forearm twist: 180 deg about the forearm's longitudinal axis (local X).
  const forearm_twist = {
    name: 'forearm_twist',
    channels: [
      rotChannel(lFore, [0, 0.6, 1.2], [I, keyQuat([1, 0, 0], 180), I]),
    ],
  };
  // Big turn: hips yaw 0 -> 200 deg while translating (root motion), plus a
  // small arm swing. Exercises model/world separation.
  const big_turn = {
    name: 'big_turn',
    channels: [
      rotChannel(hips, [0, 1.0, 2.0], [I, keyQuat([0, 1, 0], 100), keyQuat([0, 1, 0], 200)]),
      transChannel(hips, [0, 1.0, 2.0], [
        [0, 95, 0], [25, 95, 18], [50, 95, 0],
      ]),
      rotChannel(rUpArm, [0, 1.0, 2.0], [I, keyQuat([1, 0, 0], 25), I]),
    ],
  };
  return [elbow_flex, forearm_twist, big_turn];
}

// ---------- main ----------

function main() {
  const { nodes, nodeOfJoint, meshNode } = buildNodes();
  const worlds = computeWorlds(nodes);
  const meshWorld = worlds[meshNode];

  const jointNodes = JOINT_DEFS.map(([name]) => nodeOfJoint.get(name));
  const ibms = new Float32Array(jointNodes.length * 16);
  jointNodes.forEach((ni, j) => {
    const inv = mat4.invert(worlds[ni]);
    if (!inv) throw new Error(`singular bind world for joint ${j}`);
    ibms.set(mat4.multiply(inv, meshWorld), j * 16);
  });

  buildTorso();
  buildHead();
  buildArm('L');
  buildArm('R');
  buildLeg('L');
  buildLeg('R');
  buildSkirt();
  const { joints, weights: wts } = packWeights();

  const asset = {
    format: FORMAT_ID,
    version: FORMAT_VERSION,
    name: 'runway_character',
    coordinateSystem: { handedness: 'right', up: '+Y', units: 'm' },
    nodes: nodes.map((n) => ({
      name: n.name,
      parent: n.parent,
      translation: n.translation,
      rotation: n.rotation,
      scale: n.scale,
    })),
    skeleton: {
      joints: jointNodes,
      inverseBindMatrices: f32ToBase64(ibms),
    },
    mesh: {
      name: 'body',
      node: meshNode,
      positions: f32ToBase64(new Float32Array(positions)),
      normals: f32ToBase64(new Float32Array(normals)),
      joints: u16ToBase64(joints),
      weights: f32ToBase64(wts),
      indicesU16: u16ToBase64(new Uint16Array(indices)),
    },
    animations: buildAnimations(nodeOfJoint),
  };

  const out = join(dirname(fileURLToPath(import.meta.url)), '..', 'assets', 'runway_character.rskin.json');
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify(asset));
  console.log(`wrote ${out}`);
  console.log(`vertices: ${positions.length / 3}, triangles: ${indices.length / 3}, joints: ${jointNodes.length}`);
}

main();
