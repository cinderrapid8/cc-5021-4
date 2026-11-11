import assert from 'node:assert/strict';
import { loadRunwayAsset, makeSkeleton, evaluatePose } from './helpers.mjs';
import { skinMeshCPU } from '../src/core/skinning.js';

// This test runs the real GPU skinning path (WebGL1 vertex shader + bone
// textures + fixed-point readback) and compares it numerically against the
// CPU reference. It needs the optional `gl` (headless-gl) package:
//   npm install --save-dev gl
// In browsers the same check runs live in the inspector's "GPU vs CPU" panel.

let createGL = null;
try {
  const mod = await import('gl');
  createGL = mod.default;
} catch {
  // headless-gl not installed / not buildable on this machine
}

const POSITION_TOL = 1e-3; // cm (GPU float32 vs CPU float64)
const NORMAL_TOL = 1e-3;

await test('GPU skinning matches CPU reference on all test clips', async () => {
  if (!createGL) return 'SKIP';
  const { GpuSkinner } = await import('../src/gpu/skinner.js');
  const { checkCapabilities } = await import('../src/gpu/webgl.js');

  const gl = createGL(64, 64, { preserveDrawingBuffer: true });
  if (!gl) return 'SKIP';
  checkCapabilities(gl);

  const asset = loadRunwayAsset();
  const skeleton = makeSkeleton(asset);
  const skinner = new GpuSkinner(gl, asset);

  const cases = [
    [null, 0],
    ['elbow_flex', 0.5],
    ['elbow_flex', 0.25],
    ['forearm_twist', 0.6],
    ['big_turn', 1.0],
    ['big_turn', 2.0],
  ];
  for (const [clip, time] of cases) {
    const { skin, normals } = evaluatePose(asset, skeleton, clip, time);
    const cpu = skinMeshCPU(asset, skin, normals);
    skinner.setPose(skin, normals);
    const gpu = skinner.skinMeshGPU();
    let maxP = 0;
    let maxN = 0;
    let worstV = -1;
    for (let v = 0; v < asset.mesh.vertexCount; v++) {
      for (let i = 0; i < 3; i++) {
        const dp = Math.abs(cpu.positions[v * 3 + i] - gpu.positions[v * 3 + i]);
        const dn = Math.abs(cpu.normals[v * 3 + i] - gpu.normals[v * 3 + i]);
        if (dp > maxP) { maxP = dp; worstV = v; }
        if (dn > maxN) maxN = dn;
      }
    }
    const label = clip ? `${clip}@${time}s` : 'bind pose';
    assert.ok(Number.isFinite(maxP) && Number.isFinite(maxN), `${label}: NaN in GPU output`);
    assert.ok(maxP < POSITION_TOL, `${label}: max position error ${maxP} (vertex ${worstV})`);
    assert.ok(maxN < NORMAL_TOL, `${label}: max normal error ${maxN}`);
    console.log(`    ${label}: max |pos| err ${maxP.toExponential(2)}, max |nrm| err ${maxN.toExponential(2)}`);
  }
});
