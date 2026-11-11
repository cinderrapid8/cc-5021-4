import assert from 'node:assert/strict';
import { mat4 } from '../src/core/math.js';
import { loadRunwayAsset, makeSkeleton, evaluatePose } from './helpers.mjs';

const asset = loadRunwayAsset();
const skeleton = makeSkeleton(asset);

await test('asset loads with non-unit parent transforms (no warnings expected)', () => {
  assert.equal(asset.jointNodes.length, 17);
  assert.ok(asset.mesh.vertexCount > 1000);
  assert.deepEqual(asset.warnings, []);
  // sanity: the hierarchy really is non-trivial
  const stage = asset.nodes.find((n) => n.name === 'Stage');
  const unitFix = asset.nodes.find((n) => n.name === 'UnitFix');
  const meshNode = asset.nodes[asset.mesh.node];
  assert.notDeepEqual(stage.rotation, [0, 0, 0, 1]);
  assert.notDeepEqual(stage.translation, [0, 0, 0]);
  assert.notDeepEqual(unitFix.scale, [1, 1, 1]);
  assert.notDeepEqual(meshNode.scale, [1, 1, 1]);
  // non-uniform, not just non-unit
  assert.ok(unitFix.scale[0] !== unitFix.scale[1]);
});

await test('bind pose skin matrices are identity despite non-unit parents', () => {
  const { skin } = evaluatePose(asset, skeleton, null, 0);
  const I = mat4.identity();
  for (let j = 0; j < asset.jointNodes.length; j++) {
    const m = Array.from(skin.subarray(j * 16, j * 16 + 16));
    for (let k = 0; k < 16; k++) {
      assert.ok(Math.abs(m[k] - I[k]) < 1e-4,
        `joint ${j} (${asset.jointNames[j]}) skin matrix not identity at bind, element ${k}: ${m[k]} vs ${I[k]}`);
    }
  }
});

await test('mesh world matrix includes Stage placement and mesh-node scale', () => {
  const { meshWorld } = evaluatePose(asset, skeleton, null, 0);
  // mesh node = child of Stage with scale [0.011, 0.01, 0.0095]
  const expected = mat4.multiply(
    mat4.compose([1.5, 0, -2.0], asset.nodes[0].rotation, [1, 1, 1]),
    mat4.compose([0, 0, 0], [0, 0, 0, 1], [0.011, 0.01, 0.0095]),
  );
  assert.ok(mat4.almostEqual(meshWorld, expected, 1e-9));
});

await test('joint bind world positions match model-space positions through the graph', () => {
  const { world, meshWorld } = evaluatePose(asset, skeleton, null, 0);
  // Hips authored at model-space (0, 95, 0) cm
  const hipsNode = asset.jointNodes[asset.jointNames.indexOf('Hips')];
  const hipsWorld = world[hipsNode].slice(12, 15);
  const expected = mat4.transformPoint(meshWorld, [0, 95, 0]);
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(hipsWorld[i] - expected[i]) < 1e-6);
});

await test('inverse bind matrices are consistent: IBM * meshWorld^-1 * jointWorld = I at bind', () => {
  const { world } = evaluatePose(asset, skeleton, null, 0);
  const invMesh = mat4.invert(world[asset.mesh.node]);
  for (let j = 0; j < asset.jointNodes.length; j++) {
    const ibm = Array.from(asset.inverseBindMatrices.subarray(j * 16, j * 16 + 16));
    const jw = world[asset.jointNodes[j]];
    const m = mat4.multiply(mat4.multiply(invMesh, jw), ibm);
    assert.ok(mat4.almostEqual(m, mat4.identity(), 1e-4), `joint ${j}`);
  }
});
