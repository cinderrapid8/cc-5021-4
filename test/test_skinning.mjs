import assert from 'node:assert/strict';
import { mat4, quat, vec3 } from '../src/core/math.js';
import { skinMeshCPU, skinVertexCPU } from '../src/core/skinning.js';
import { loadAsset, buildSkeleton, computeWorldMatrices, computeSkinMatrices } from '../src/core/loader.js';
import { bindPose, samplePose } from '../src/core/animation.js';
import { FORMAT_ID, FORMAT_VERSION, f32ToBase64, u16ToBase64 } from '../src/core/format.js';
import { loadRunwayAsset, makeSkeleton, evaluatePose } from './helpers.mjs';

const DEG = Math.PI / 180;
const asset = loadRunwayAsset();
const skeleton = makeSkeleton(asset);

function findVertex(pred) {
  const { joints, weights } = asset.mesh;
  for (let v = 0; v < asset.mesh.vertexCount; v++) {
    const jw = [];
    for (let k = 0; k < 4; k++) jw.push([joints[v * 4 + k], weights[v * 4 + k]]);
    if (pred(jw, v)) return v;
  }
  throw new Error('no vertex matches');
}

const JI = (name) => asset.jointNames.indexOf(name);

function rotAboutPoint(center, quatRot, p) {
  const d = vec3.sub(p, center);
  return vec3.add(center, quat.rotateVec3(quatRot, d));
}

await test('bind pose: CPU skinning reproduces authored positions and normals', () => {
  const { skin, normals } = evaluatePose(asset, skeleton, null, 0);
  const out = skinMeshCPU(asset, skin, normals);
  let maxP = 0;
  let maxN = 0;
  for (let v = 0; v < asset.mesh.vertexCount; v++) {
    for (let i = 0; i < 3; i++) {
      maxP = Math.max(maxP, Math.abs(out.positions[v * 3 + i] - asset.mesh.positions[v * 3 + i]));
      maxN = Math.max(maxN, Math.abs(out.normals[v * 3 + i] - asset.mesh.normals[v * 3 + i]));
    }
  }
  assert.ok(maxP < 1e-4, `max position deviation ${maxP}`);
  assert.ok(maxN < 1e-4, `max normal deviation ${maxN}`);
});

await test('elbow_flex t=0.5: forearm vertex matches analytic rotation about the elbow', () => {
  const { skin, normals } = evaluatePose(asset, skeleton, 'elbow_flex', 0.5);
  const out = skinMeshCPU(asset, skin, normals);
  const v = findVertex((jw) => jw.some(([j, w]) => j === JI('L_Forearm') && w === 1));
  const p = asset.mesh.positions.slice(v * 3, v * 3 + 3);
  const n = asset.mesh.normals.slice(v * 3, v * 3 + 3);
  const elbow = [40, 142, 0];
  const q = quat.fromAxisAngle([0, 1, 0], -115 * DEG);
  const expected = rotAboutPoint(elbow, q, Array.from(p));
  const expectedN = quat.rotateVec3(q, Array.from(n));
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(out.positions[v * 3 + i] - expected[i]) < 1e-4,
      `position[${i}] ${out.positions[v * 3 + i]} != ${expected[i]}`);
    assert.ok(Math.abs(out.normals[v * 3 + i] - expectedN[i]) < 1e-4,
      `normal[${i}] ${out.normals[v * 3 + i]} != ${expectedN[i]}`);
  }
});

await test('elbow_flex: blended elbow vertex is the weight-average of both joints', () => {
  const { skin, normals } = evaluatePose(asset, skeleton, 'elbow_flex', 0.5);
  const out = skinMeshCPU(asset, skin, normals);
  const v = findVertex((jw) => {
    const a = jw.find(([j]) => j === JI('L_UpperArm'));
    const b = jw.find(([j]) => j === JI('L_Forearm'));
    return a && b && Math.abs(a[1] - 0.5) < 0.15 && Math.abs(b[1] - 0.5) < 0.15;
  });
  const p = asset.mesh.positions.slice(v * 3, v * 3 + 3);
  const q = quat.fromAxisAngle([0, 1, 0], -115 * DEG);
  const rotated = rotAboutPoint([40, 142, 0], q, Array.from(p));
  const wUp = asset.mesh.weights[v * 4 + (asset.mesh.joints[v * 4] === JI('L_UpperArm') ? 0 : 1)];
  const wFo = 1 - wUp;
  const expected = [
    wUp * p[0] + wFo * rotated[0],
    wUp * p[1] + wFo * rotated[1],
    wUp * p[2] + wFo * rotated[2],
  ];
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(out.positions[v * 3 + i] - expected[i]) < 1e-3,
      `blended position[${i}] ${out.positions[v * 3 + i]} != ${expected[i]}`);
  }
});

await test('forearm_twist t=0.6: forearm vertex rotates 180° about the arm axis', () => {
  const { skin, normals } = evaluatePose(asset, skeleton, 'forearm_twist', 0.6);
  const out = skinMeshCPU(asset, skin, normals);
  const v = findVertex((jw) => jw.some(([j, w]) => j === JI('L_Forearm') && w === 1));
  const p = asset.mesh.positions.slice(v * 3, v * 3 + 3);
  const elbow = [40, 142, 0];
  const q = quat.fromAxisAngle([1, 0, 0], 180 * DEG);
  const expected = rotAboutPoint(elbow, q, Array.from(p));
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(out.positions[v * 3 + i] - expected[i]) < 1e-4,
      `position[${i}] ${out.positions[v * 3 + i]} != ${expected[i]}`);
  }
});

await test('forearm_twist t=0.3 (90°): twist graduates along the forearm', () => {
  // Blending a 90° twist with identity gives partial rotation proportional to
  // the forearm weight, so vertices nearer the elbow twist less.
  const { skin, normals } = evaluatePose(asset, skeleton, 'forearm_twist', 0.3);
  const out = skinMeshCPU(asset, skin, normals);
  const elbow = [40, 142, 0];
  const q = quat.fromAxisAngle([1, 0, 0], 90 * DEG);
  const twistAngleOf = (v) => {
    const y = out.positions[v * 3 + 1] - elbow[1];
    const z = out.positions[v * 3 + 2] - elbow[2];
    const y0 = asset.mesh.positions[v * 3 + 1] - elbow[1];
    const z0 = asset.mesh.positions[v * 3 + 2] - elbow[2];
    return Math.atan2(y0 * z - z0 * y, y0 * y + z0 * z); // signed delta about X
  };
  const weightOf = (v, joint) => {
    for (let k = 0; k < 4; k++) {
      if (asset.mesh.joints[v * 4 + k] === joint) return asset.mesh.weights[v * 4 + k];
    }
    return 0;
  };
  const vNear = findVertex((jw, v) => {
    const x = asset.mesh.positions[v * 3];
    return x > 41 && x < 43;
  });
  const vFar = findVertex((jw, v) => {
    const x = asset.mesh.positions[v * 3];
    return x > 55 && x < 57;
  });
  // analytic check from the actual weights of each vertex
  for (const v of [vNear, vFar]) {
    const wF = weightOf(v, JI('L_Forearm'));
    const wU = weightOf(v, JI('L_UpperArm'));
    const p = Array.from(asset.mesh.positions.slice(v * 3, v * 3 + 3));
    const rot = rotAboutPoint(elbow, q, p);
    const expected = [
      wU * p[0] + wF * rot[0],
      wU * p[1] + wF * rot[1],
      wU * p[2] + wF * rot[2],
    ];
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(out.positions[v * 3 + i] - expected[i]) < 1e-4,
        `vertex ${v} position[${i}] ${out.positions[v * 3 + i]} != ${expected[i]}`);
    }
  }
  const aNear = Math.abs(twistAngleOf(vNear));
  const aFar = Math.abs(twistAngleOf(vFar));
  assert.ok(aFar > aNear + 0.2,
    `far vertex twists (${aFar} rad) more than near-elbow vertex (${aNear} rad)`);
  assert.ok(Math.abs(aFar - Math.PI / 2) < 1e-3, `fully-weighted vertex twists 90 deg, got ${aFar}`);
});

await test('big_turn t=1.0: body follows hips rigidly in model space', () => {
  const { skin, normals } = evaluatePose(asset, skeleton, 'big_turn', 1.0);
  const out = skinMeshCPU(asset, skin, normals);
  const q = quat.fromAxisAngle([0, 1, 0], 100 * DEG);
  const hipsBind = [0, 95, 0];
  const hipsNew = [25, 95, 18];
  const check = (v) => {
    const p = asset.mesh.positions.slice(v * 3, v * 3 + 3);
    const expected = vec3.add(hipsNew, quat.rotateVec3(q, vec3.sub(Array.from(p), hipsBind)));
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(out.positions[v * 3 + i] - expected[i]) < 1e-3,
        `vertex ${v} position[${i}] ${out.positions[v * 3 + i]} != ${expected[i]}`);
    }
  };
  check(findVertex((jw) => jw.some(([j, w]) => j === JI('Hips') && w === 1)));
  check(findVertex((jw) => jw.some(([j, w]) => j === JI('Head') && w === 1)));
  check(findVertex((jw) => jw.some(([j, w]) => j === JI('L_Hand') && w > 0.9)));
  check(findVertex((jw) => jw.some(([j, w]) => j === JI('R_Foot') && w > 0.9)));
});

await test('big_turn: skinned result is invariant to the shared Stage transform', () => {
  // Skinning happens in model space: moving the shared ancestor (Stage) must
  // not change model-space skinned positions at all.
  const { skin, normals } = evaluatePose(asset, skeleton, 'big_turn', 1.0);
  const base = skinMeshCPU(asset, skin, normals);
  // recompute every world matrix with a moved Stage (node 0)
  const clip = asset.animations.find((a) => a.name === 'big_turn');
  const posed = samplePose(asset, clip, 1.0);
  const parents = asset.nodes.map((n) => n.parent);
  const world2 = new Array(asset.nodes.length);
  for (let i = 0; i < world2.length; i++) {
    const local = i === 0
      ? mat4.compose([10, 5, -7], quat.fromAxisAngle([0, 1, 0], 0.5), [1, 1, 1])
      : mat4.compose(posed[i].t, posed[i].q, posed[i].s);
    world2[i] = parents[i] === -1 ? local : mat4.multiply(world2[parents[i]], local);
  }
  const sm2 = computeSkinMatrices(asset, world2);
  const out2 = skinMeshCPU(asset, sm2.skin, sm2.normals);
  let maxD = 0;
  for (let i = 0; i < base.positions.length; i++) {
    maxD = Math.max(maxD, Math.abs(base.positions[i] - out2.positions[i]));
  }
  assert.ok(maxD < 1e-4, `model-space skinning changed under Stage motion by ${maxD}`);
});

// ---- synthetic assets for policy / scale coverage ----

function syntheticAsset({ jointLocalScale, meshNodeScale, zeroInfluence }) {
  const nodes = [
    { name: 'Root', parent: -1, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    { name: 'J0', parent: 0, translation: [1, 0, 0], rotation: [0, 0, 0, 1], scale: jointLocalScale },
    { name: 'Mesh', parent: 0, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: meshNodeScale },
  ];
  const worlds = [null, null, null];
  for (let i = 0; i < 3; i++) {
    const l = mat4.compose(nodes[i].translation, nodes[i].rotation, nodes[i].scale);
    worlds[i] = nodes[i].parent === -1 ? l : mat4.multiply(worlds[nodes[i].parent], l);
  }
  const ibm = mat4.multiply(mat4.invert(worlds[1]), worlds[2]);
  const positions = zeroInfluence ? [2, 1, 0, 5, 5, 5] : [2, 1, 0];
  const normals = zeroInfluence ? [1, 0, 0, 0, 1, 0] : [1, 0, 0];
  const joints = zeroInfluence ? [0, 0, 0, 0, 0, 0, 0, 0] : [0, 0, 0, 0];
  const weights = zeroInfluence ? [1, 0, 0, 0, 0, 0, 0, 0] : [1, 0, 0, 0];
  return loadAsset({
    format: FORMAT_ID,
    version: FORMAT_VERSION,
    nodes,
    skeleton: { joints: [1], inverseBindMatrices: f32ToBase64(new Float32Array(ibm)) },
    mesh: {
      node: 2,
      positions: f32ToBase64(new Float32Array(positions)),
      normals: f32ToBase64(new Float32Array(normals)),
      joints: u16ToBase64(new Uint16Array(joints)),
      weights: f32ToBase64(new Float32Array(weights)),
      indicesU16: u16ToBase64(new Uint16Array(zeroInfluence ? [0, 1, 0] : [0, 0, 0])),
    },
    animations: [],
  });
}

await test('zero-influence vertex stays rigid to the mesh node when joints move', () => {
  const a = syntheticAsset({ jointLocalScale: [1, 1, 1], meshNodeScale: [1, 1, 1], zeroInfluence: true });
  const skel = buildSkeleton(a);
  const pose = bindPose(a);
  pose[1].q = quat.fromAxisAngle([0, 0, 1], Math.PI / 2); // rotate joint 90 deg
  const world = computeWorldMatrices(a, skel, pose);
  const { skin, normals } = computeSkinMatrices(a, world);
  const out = skinMeshCPU(a, skin, normals);
  // vertex 0 follows the joint; vertex 1 (zero-influence) stays at [5,5,5]
  assert.ok(Math.abs(out.positions[3] - 5) < 1e-6);
  assert.ok(Math.abs(out.positions[4] - 5) < 1e-6);
  assert.ok(Math.abs(out.positions[5] - 5) < 1e-6);
  // and vertex 0 did move
  const moved = Math.hypot(out.positions[0] - 2, out.positions[1] - 1, out.positions[2] - 0);
  assert.ok(moved > 0.5);
});

await test('degenerate normal blend (opposing normals cancel) keeps authored normal', () => {
  // two joints: identity and 180° about Z; a 0.5/0.5 vertex whose normal
  // cancels out must fall back to the authored normal instead of NaN
  const nodes = [
    { name: 'Root', parent: -1, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    { name: 'J0', parent: 0, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    { name: 'J1', parent: 0, translation: [0, 0, 0], rotation: quat.fromAxisAngle([0, 0, 1], Math.PI), scale: [1, 1, 1] },
    { name: 'Mesh', parent: 0, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
  ];
  const worlds = [];
  for (let i = 0; i < 4; i++) {
    const l = mat4.compose(nodes[i].translation, nodes[i].rotation, nodes[i].scale);
    worlds[i] = nodes[i].parent === -1 ? l : mat4.multiply(worlds[nodes[i].parent], l);
  }
  const ibms = new Float32Array(32);
  ibms.set(mat4.multiply(mat4.invert(worlds[1]), worlds[3]), 0);
  ibms.set(mat4.multiply(mat4.invert(worlds[2]), worlds[3]), 16);
  const a = loadAsset({
    format: FORMAT_ID,
    version: FORMAT_VERSION,
    nodes,
    skeleton: { joints: [1, 2], inverseBindMatrices: f32ToBase64(ibms) },
    mesh: {
      node: 3,
      positions: f32ToBase64(new Float32Array([1, 0, 0])),
      normals: f32ToBase64(new Float32Array([1, 0, 0])),
      joints: u16ToBase64(new Uint16Array([0, 1, 0, 0])),
      weights: f32ToBase64(new Float32Array([0.5, 0.5, 0, 0])),
      indicesU16: u16ToBase64(new Uint16Array([0, 0, 0])),
    },
    animations: [],
  });
  const skel = buildSkeleton(a);
  const world = computeWorldMatrices(a, skel, bindPose(a));
  const { skin, normals } = computeSkinMatrices(a, world);
  const out = skinVertexCPU(skin, normals, a.mesh.joints, a.mesh.weights, 0,
    a.mesh.positions, a.mesh.normals);
  assert.ok(Number.isFinite(out.normal[0]) && Number.isFinite(out.normal[1]) && Number.isFinite(out.normal[2]),
    'normal must not be NaN');
  assert.ok(Math.abs(out.normal[0] - 1) < 1e-6);
  assert.ok(Math.abs(out.normal[1]) < 1e-6);
  assert.ok(Math.abs(out.normal[2]) < 1e-6);
});

await test('non-uniform scale inside skin matrices: normals stay perpendicular to tangents', () => {
  // Joint parent carries non-uniform scale that the mesh node chain does not
  // have, so posed skin matrices contain shear/scale. The per-bone
  // inverse-transpose normal matrix must keep normals perpendicular.
  const nodes = [
    { name: 'Root', parent: -1, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    { name: 'Squeezer', parent: 0, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [2, 1, 1] },
    { name: 'J0', parent: 1, translation: [1, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    { name: 'Mesh', parent: 0, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
  ];
  const worlds = [];
  for (let i = 0; i < 4; i++) {
    const l = mat4.compose(nodes[i].translation, nodes[i].rotation, nodes[i].scale);
    worlds[i] = nodes[i].parent === -1 ? l : mat4.multiply(worlds[nodes[i].parent], l);
  }
  const ibm = mat4.multiply(mat4.invert(worlds[2]), worlds[3]);
  const a = loadAsset({
    format: FORMAT_ID,
    version: FORMAT_VERSION,
    nodes,
    skeleton: { joints: [2], inverseBindMatrices: f32ToBase64(new Float32Array(ibm)) },
    mesh: {
      node: 3,
      // bind vertex at joint origin + offset; normal (1,0,0), tangent (0,1,0)
      positions: f32ToBase64(new Float32Array([3, 1, 0])),
      normals: f32ToBase64(new Float32Array([1, 0, 0])),
      joints: u16ToBase64(new Uint16Array([0, 0, 0, 0])),
      weights: f32ToBase64(new Float32Array([1, 0, 0, 0])),
      indicesU16: u16ToBase64(new Uint16Array([0, 0, 0])),
    },
    animations: [],
  });
  const skel = buildSkeleton(a);
  const pose = bindPose(a);
  pose[2].q = quat.fromAxisAngle([0, 0, 1], Math.PI / 2); // rotate joint 90 deg about Z
  const world = computeWorldMatrices(a, skel, pose);
  const { skin, normals } = computeSkinMatrices(a, world);
  const sm = Array.from(skin.subarray(0, 16));
  // the skin matrix genuinely contains non-uniform scale/shear now
  const col0 = Math.hypot(sm[0], sm[1], sm[2]);
  const col1 = Math.hypot(sm[4], sm[5], sm[6]);
  assert.ok(Math.abs(col0 - col1) > 0.1, 'test setup: skin matrix should be non-uniform');
  // tangent perpendicular to the bind normal, transformed as a direction
  const tangent = mat4.transformDir(sm, [0, 1, 0]);
  const out = skinVertexCPU(skin, normals, a.mesh.joints, a.mesh.weights, 0,
    a.mesh.positions, a.mesh.normals);
  const dot = vec3.dot(vec3.normalize(tangent), out.normal);
  assert.ok(Math.abs(dot) < 1e-6, `normal not perpendicular to tangent: dot=${dot}`);
  // position matches the direct matrix transform
  const expected = mat4.transformPoint(sm, [3, 1, 0]);
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(out.position[i] - expected[i]) < 1e-6);
});
