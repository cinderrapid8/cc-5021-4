import assert from 'node:assert/strict';
import { mat4 } from '../src/core/math.js';
import { skinMeshCPU, skinVertexCPU } from '../src/core/skinning.js';
import {
  loadAsset, buildSkeleton, computeWorldMatrices, computeSkinMatrices,
} from '../src/core/loader.js';
import { bindPose } from '../src/core/animation.js';
import { FORMAT_ID, FORMAT_VERSION, f32ToBase64, u16ToBase64 } from '../src/core/format.js';
import {
  traceSkinVertex, classifyVertexTrace, scanMesh,
  runTwistProbe, runScaleProbe,
  captureReplay, replayTrace, digestOfTraces, TraceLog,
  WEIGHT_SUM_TOL,
} from '../src/core/diagnose.js';
import { loadRunwayAsset, makeSkeleton, evaluatePose } from './helpers.mjs';

const asset = loadRunwayAsset();
const skeleton = makeSkeleton(asset);
const JI = (name) => asset.jointNames.indexOf(name);
const I9 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

function findVertex(pred) {
  const { joints, weights } = asset.mesh;
  for (let v = 0; v < asset.mesh.vertexCount; v++) {
    const jw = [];
    for (let k = 0; k < 4; k++) jw.push([joints[v * 4 + k], weights[v * 4 + k]]);
    if (pred(jw, v)) return v;
  }
  throw new Error('no vertex matches');
}

// Minimal asset-like object that bypasses the loader (no validation, no
// weight repair) — used to prove diagnostics report stored weights as-is.
function rawAsset(weights, joints = [0, 0, 0, 0]) {
  return {
    jointNodes: [0],
    jointNames: ['J0'],
    mesh: {
      vertexCount: 1,
      positions: [1, 2, 3],
      normals: [0, 1, 0],
      joints,
      weights,
      zeroInfluence: [],
    },
  };
}

// ---------- trace fidelity & honesty ----------

await test('trace reproduces the CPU skinner exactly, contributions reconstruct position', () => {
  const { skin, normals } = evaluatePose(asset, skeleton, 'elbow_flex', 0.5);
  for (const v of [0, 500, 1000, asset.mesh.vertexCount - 1]) {
    const tr = traceSkinVertex(asset, skin, normals, v);
    const ref = skinVertexCPU(skin, normals, asset.mesh.joints, asset.mesh.weights, v,
      asset.mesh.positions, asset.mesh.normals);
    for (let c = 0; c < 3; c++) {
      assert.ok(Math.abs(tr.position[c] - ref.position[c]) < 1e-12, `v${v} pos[${c}]`);
      assert.ok(Math.abs(tr.normal[c] - ref.normal[c]) < 1e-12, `v${v} nrm[${c}]`);
    }
    if (tr.branch === 'lbs') {
      const sum = [0, 0, 0];
      for (const inf of tr.influences) {
        for (let c = 0; c < 3; c++) sum[c] += inf.contribution[c];
      }
      for (let c = 0; c < 3; c++) {
        assert.ok(Math.abs(sum[c] - tr.position[c]) < 1e-12, `v${v} reconstruction[${c}]`);
      }
    }
  }
});

await test('trace uses stored weights as-is: no silent renormalization', () => {
  // weight sum 0.5 — a loader would renormalize; the tracer must NOT
  const a = rawAsset([0.5, 0, 0, 0]);
  const tr = traceSkinVertex(a, mat4.identity(), I9, 0);
  assert.equal(tr.weightSum, 0.5);
  // identity skin matrix: position = 0.5 * p, NOT p (which renormalizing to 1 would give)
  assert.deepEqual(tr.position, [0.5, 1, 1.5]);
  assert.equal(tr.loadRepairs, null); // asset never went through the loader
  const cls = classifyVertexTrace(tr);
  assert.equal(cls.kind, 'weights');
  assert.equal(cls.stage, 'load');
});

await test('diagnostics never mutate weights, joints or positions', () => {
  const w0 = asset.mesh.weights.slice();
  const j0 = asset.mesh.joints.slice();
  const p0 = asset.mesh.positions.slice();
  const { pose, skin, normals } = evaluatePose(asset, skeleton, 'forearm_twist', 0.6);
  const v = findVertex((jw) => jw.filter(([, w]) => w > 0).length >= 2);
  traceSkinVertex(asset, skin, normals, v);
  scanMesh(asset, skin, normals, { maxRecords: 4 });
  const replay = captureReplay(asset, pose, [v], { asset: 'runway', clip: 'forearm_twist', time: 0.6 });
  replayTrace(replay);
  runTwistProbe(180);
  runScaleProbe();
  assert.ok(asset.mesh.weights.every((w, i) => w === w0[i]), 'weights mutated');
  assert.ok(asset.mesh.joints.every((j, i) => j === j0[i]), 'joints mutated');
  assert.ok(asset.mesh.positions.every((p, i) => p === p0[i]), 'positions mutated');
});

await test('zero-influence vertex: rigid branch, weight sum 0, authored position', () => {
  const a = rawAsset([0, 0, 0, 0]);
  const tr = traceSkinVertex(a, mat4.identity(), I9, 0);
  assert.equal(tr.branch, 'zero-influence');
  assert.equal(tr.normalBranch, 'rigid');
  assert.equal(tr.weightSum, 0);
  assert.deepEqual(tr.position, [1, 2, 3]);
  const cls = classifyVertexTrace(tr);
  assert.equal(cls.kind, 'none');
});

await test('degenerate normal blend is traced as degenerate-authored', () => {
  // two joints bound at identity; J1 is POSED 180° about Z, so the skin
  // matrices differ by 180° and the 0.5/0.5 blended normal cancels out
  const nodes = [
    { name: 'Root', parent: -1, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    { name: 'J0', parent: 0, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    { name: 'J1', parent: 0, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    { name: 'Mesh', parent: 0, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
  ];
  const ibms = new Float32Array(32);
  ibms.set(mat4.identity(), 0);
  ibms.set(mat4.identity(), 16);
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
  const skel = makeSkeleton(a);
  const pose = bindPose(a);
  // exact 180° quaternion (fromAxisAngle would leave cos(π/2) ≈ 6e-17 residue,
  // and the degenerate branch requires an exactly zero blended normal)
  pose[2].q = [0, 0, 1, 0];
  const world = computeWorldMatrices(a, skel, pose);
  const { skin, normals } = computeSkinMatrices(a, world);
  const tr = traceSkinVertex(a, skin, normals, 0);
  assert.equal(tr.branch, 'lbs');
  assert.equal(tr.normalBranch, 'degenerate-authored');
  assert.deepEqual(tr.normal, [1, 0, 0]);
});

// ---------- load-stage repair linkage ----------

await test('load-stage weight repair is linked into the trace (honest chain)', () => {
  // weight sum 2.0 -> loader renormalizes and records the repair
  const nodes = [
    { name: 'Root', parent: -1, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    { name: 'J0', parent: 0, translation: [1, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    { name: 'Mesh', parent: 0, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
  ];
  const a = loadAsset({
    format: FORMAT_ID,
    version: FORMAT_VERSION,
    nodes,
    skeleton: { joints: [1], inverseBindMatrices: f32ToBase64(new Float32Array(mat4.identity())) },
    mesh: {
      node: 2,
      positions: f32ToBase64(new Float32Array([2, 1, 0])),
      normals: f32ToBase64(new Float32Array([1, 0, 0])),
      joints: u16ToBase64(new Uint16Array([0, 0, 0, 0])),
      weights: f32ToBase64(new Float32Array([2, 0, 0, 0])),
      indicesU16: u16ToBase64(new Uint16Array([0, 0, 0])),
    },
    animations: [],
  });
  assert.deepEqual(a.mesh.weightRepairs.rescaled, [0]);
  const { skin, normals } = evaluatePose(a, makeSkeleton(a), null, 0);
  const tr = traceSkinVertex(a, skin, normals, 0);
  assert.deepEqual(tr.loadRepairs, ['renormalized']);
  assert.ok(Math.abs(tr.weightSum - 1) < 1e-6, 'post-repair sum is 1 — and the repair is visible');
  const cls = classifyVertexTrace(tr);
  assert.equal(cls.kind, 'none', 'repaired weights are not flagged as an anomaly');
});

// ---------- classification ----------

await test('classification: CPU/GPU divergence is numerical @ skin-gpu', () => {
  const { skin, normals } = evaluatePose(asset, skeleton, null, 0);
  const tr = traceSkinVertex(asset, skin, normals, 0);
  const cls = classifyVertexTrace(tr, { cpuGpuResidual: 0.01 });
  assert.equal(cls.kind, 'numerical');
  assert.equal(cls.stage, 'skin-gpu');
  const clsOk = classifyVertexTrace(tr, { cpuGpuResidual: 1e-5 });
  assert.equal(clsOk.kind, 'none');
});

await test('classification: large twist + spread is algorithmic @ skin-cpu', () => {
  const probe = runTwistProbe(180);
  const cls = classifyVertexTrace(probe.traces.blended);
  assert.equal(cls.kind, 'algorithmic');
  assert.equal(cls.stage, 'skin-cpu');
  assert.ok(cls.evidence.maxJointAngleDeg > 170);
  assert.ok(cls.evidence.spread > 9);
  // single-joint vertex of the same probe: nothing to blend -> no anomaly
  const clsSingle = classifyVertexTrace(probe.traces.single);
  assert.equal(clsSingle.kind, 'none');
});

await test('classification: non-uniform scale blames coordinate only when the probe fails', () => {
  const probe = runScaleProbe();
  assert.ok(probe.trace.nonUniformScale);
  const okCls = classifyVertexTrace(probe.trace, { scaleProbeOk: true });
  assert.equal(okCls.kind, 'none');
  const badCls = classifyVertexTrace(probe.trace, { scaleProbeOk: false });
  assert.equal(badCls.kind, 'coordinate');
  assert.equal(badCls.stage, 'skin-matrix');
});

// ---------- probes ----------

await test('twist probe: rigid vertex is exact, blended collapse matches analytic LBS', () => {
  const probe = runTwistProbe(180);
  assert.ok(probe.singleJointError < 1e-9, `numerical bound ${probe.singleJointError}`);
  assert.ok(probe.blended.implError < 1e-9, 'implementation matches analytic LBS');
  assert.ok(Math.abs(probe.blended.collapseAnalytic) < 1e-9, '180 deg 50/50 collapses to the axis');
  assert.ok(Math.abs(probe.blended.collapseMeasured - probe.blended.collapseAnalytic) < 1e-6);
  assert.ok(probe.verdict.implementationMatchesLBS);
  const p90 = runTwistProbe(90);
  assert.ok(Math.abs(p90.blended.collapseAnalytic - Math.SQRT1_2) < 1e-9);
});

await test('scale probe: positions exact, normals perpendicular under non-uniform scale', () => {
  const probe = runScaleProbe();
  assert.ok(probe.anisotropy > 1.5, 'probe really contains non-uniform scale');
  assert.ok(probe.positionError < 1e-9);
  assert.ok(Math.abs(probe.normalTangentDot) < 1e-9);
  assert.ok(probe.coordinateOk);
});

// ---------- mesh scan (bounded cost, stage localization) ----------

await test('scanMesh: clean loaded asset yields no anomalies, bounded records', () => {
  const { skin, normals } = evaluatePose(asset, skeleton, 'forearm_twist', 0.6);
  const scan = scanMesh(asset, skin, normals, { maxRecords: 8 });
  assert.equal(scan.vertexCount, asset.mesh.vertexCount);
  assert.equal(scan.branchCounts.lbs + scan.branchCounts['zero-influence'], scan.vertexCount);
  assert.ok(scan.records.length <= 8);
  assert.ok(scan.maxWeightSumDeviation.value < 1e-6,
    `loader normalized every weight (max dev ${scan.maxWeightSumDeviation.value})`);
  assert.equal(scan.maxResidual, null, 'no reference -> residual unavailable, not zero');
  assert.equal(scan.referenceStage, null);
  assert.equal(scan.anomalyCounts['weight-sum'], 0);
});

await test('scanMesh: residual anomalies localize to the reference stage', () => {
  const { skin, normals } = evaluatePose(asset, skeleton, 'elbow_flex', 0.5);
  const cpu = skinMeshCPU(asset, skin, normals);
  const shifted = Float64Array.from(cpu.positions);
  shifted[0] += 0.01; // corrupt only vertex 0, x component
  const scan = scanMesh(asset, skin, normals, {
    reference: { positions: shifted, stage: 'skin-gpu' },
    maxRecords: 8,
  });
  assert.equal(scan.records.length, 1);
  assert.equal(scan.records[0].vertex, 0);
  assert.deepEqual(scan.records[0].kinds, ['residual']);
  assert.equal(scan.records[0].stage, 'skin-gpu');
  assert.ok(Math.abs(scan.maxResidual.value - 0.01) < 1e-12);
  assert.equal(scan.maxResidual.vertex, 0);
});

await test('scanMesh: weight-sum anomaly localizes to the load stage', () => {
  const a = rawAsset([0.5, 0, 0, 0]);
  const scan = scanMesh(a, mat4.identity(), I9, {});
  assert.equal(scan.records.length, 1);
  assert.ok(scan.records[0].kinds.includes('weight-sum'));
  assert.equal(scan.records[0].stage, 'load');
  assert.ok(Math.abs(scan.maxWeightSumDeviation.value - 0.5) < 1e-9);
  assert.equal(scan.maxWeightSumDeviation.vertex, 0);
});

await test('scanMesh: candidate and record caps bound the evidence size', () => {
  // 5 vertices, all with weight sum 0.5; caps force drops
  const a = {
    jointNodes: [0],
    jointNames: ['J0'],
    mesh: {
      vertexCount: 5,
      positions: [1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0],
      normals: [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
      joints: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      weights: [0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0],
      zeroInfluence: [],
    },
  };
  const scan = scanMesh(a, mat4.identity(), I9, { maxRecords: 2, maxCandidates: 3 });
  assert.equal(scan.anomalyCounts['weight-sum'], 5, 'aggregates count every vertex');
  assert.equal(scan.candidatesConsidered, 3);
  assert.equal(scan.candidatesDropped, 2);
  assert.equal(scan.records.length, 2, 'records stay within maxRecords');
});

// ---------- replay ----------

await test('replay: deterministic digest, matches live pipeline, tamper-evident', () => {
  const { pose, skin, normals } = evaluatePose(asset, skeleton, 'forearm_twist', 0.6);
  const vBlend = findVertex((jw) => {
    const nz = jw.filter(([, w]) => w > 0);
    return nz.length === 2 && Math.abs(nz[0][1] - 0.5) < 0.1;
  });
  const ids = [vBlend, 0, asset.mesh.vertexCount - 1];
  const replay = captureReplay(asset, pose, ids, { asset: 'runway', clip: 'forearm_twist', time: 0.6 });

  const r1 = replayTrace(replay);
  const r2 = replayTrace(JSON.parse(JSON.stringify(replay))); // JSON round-trip
  assert.equal(r1.digest, r2.digest, 're-run evidence must be consistent');
  assert.equal(r1.digest, replay.digest, 'digest stored at capture time matches re-run');

  // replayed traces match the live full-asset pipeline exactly
  const live = skinMeshCPU(asset, skin, normals);
  r1.traces.forEach((tr, i) => {
    const v = ids[i];
    assert.equal(tr.vertex, v, 'original vertex index preserved');
    for (let c = 0; c < 3; c++) {
      assert.ok(Math.abs(tr.position[c] - live.positions[v * 3 + c]) < 1e-9,
        `vertex ${v} pos[${c}]: replay ${tr.position[c]} vs live ${live.positions[v * 3 + c]}`);
    }
  });

  // weight sums in the replay are the stored ones
  r1.traces.forEach((tr, i) => {
    const v = ids[i];
    let s = 0;
    for (let k = 0; k < 4; k++) s += asset.mesh.weights[v * 4 + k];
    assert.ok(Math.abs(tr.weightSum - s) < 1e-12);
  });

  // tampering with a weight changes the digest
  const bad = JSON.parse(JSON.stringify(replay));
  bad.vertices[0].weights[0] += 0.01;
  assert.notEqual(replayTrace(bad).digest, replay.digest);
  // tampering with meta changes the digest
  const badMeta = JSON.parse(JSON.stringify(replay));
  badMeta.meta.clip = 'other';
  assert.notEqual(replayTrace(badMeta).digest, replay.digest);
});

await test('replay: rejects wrong type, bad version and non-topological nodes', () => {
  const { pose } = evaluatePose(asset, skeleton, null, 0);
  const replay = captureReplay(asset, pose, [0], {});
  assert.throws(() => replayTrace({ ...replay, type: 'other' }), /type/);
  assert.throws(() => replayTrace({ ...replay, version: 99 }), /版本/);
  const bad = JSON.parse(JSON.stringify(replay));
  bad.nodes[2].parent = 10;
  assert.throws(() => replayTrace(bad), /拓扑/);
});

await test('digest covers branch, weight sum and skinned result', () => {
  const { skin, normals } = evaluatePose(asset, skeleton, null, 0);
  const t1 = traceSkinVertex(asset, skin, normals, 0);
  const d1 = digestOfTraces([t1], { asset: 'a', clip: null, time: 0 });
  const d2 = digestOfTraces([t1], { asset: 'a', clip: null, time: 0 });
  assert.equal(d1, d2, 'same evidence -> same digest');
  const t2 = traceSkinVertex(asset, skin, normals, 1);
  assert.notEqual(digestOfTraces([t2], { asset: 'a', clip: null, time: 0 }), d1);
});

// ---------- trace log ----------

await test('TraceLog: fixed capacity, dropped entries counted', () => {
  const log = new TraceLog(4);
  for (let i = 0; i < 10; i++) log.push('skin-cpu', 'trace', `event ${i}`);
  assert.equal(log.records.length, 4);
  assert.equal(log.dropped, 6);
  assert.equal(log.total, 10);
  const latest = log.latest(2);
  assert.equal(latest[0].detail, 'event 8');
  assert.equal(latest[1].detail, 'event 9');
  assert.ok(latest[1].seq > latest[0].seq, 'sequence numbers increase');
});

// ---------- tolerance sanity ----------

await test('weight-sum tolerance matches the loader warn policy', () => {
  assert.equal(WEIGHT_SUM_TOL, 1e-3);
});
