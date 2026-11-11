import assert from 'node:assert/strict';
import { validateWeights, validateInverseBindMatrices, AssetError } from '../src/core/validate.js';
import { mat4 } from '../src/core/math.js';

const JOINTS = 4;

function makeWeights(rows) {
  return {
    joints: new Uint16Array(rows.flatMap((r) => r.j)),
    weights: new Float32Array(rows.flatMap((r) => r.w)),
  };
}

await test('clean weights pass without warnings', () => {
  const { joints, weights } = makeWeights([
    { j: [0, 1, 0, 0], w: [0.5, 0.5, 0, 0] },
    { j: [2, 0, 0, 0], w: [1, 0, 0, 0] },
  ]);
  const r = validateWeights(joints, weights, 2, JOINTS);
  assert.equal(r.warnings.length, 0);
  assert.equal(r.zeroInfluence.length, 0);
});

await test('NaN weight -> asset rejected', () => {
  const { joints, weights } = makeWeights([
    { j: [0, 0, 0, 0], w: [NaN, 0, 0, 0] },
  ]);
  assert.throws(() => validateWeights(joints, weights, 1, JOINTS), AssetError);
});

await test('Infinity weight -> asset rejected', () => {
  const { joints, weights } = makeWeights([
    { j: [0, 0, 0, 0], w: [Infinity, 0, 0, 0] },
  ]);
  assert.throws(() => validateWeights(joints, weights, 1, JOINTS), AssetError);
});

await test('joint index out of range -> asset rejected', () => {
  const { joints, weights } = makeWeights([
    { j: [0, 7, 0, 0], w: [0.5, 0.5, 0, 0] },
  ]);
  assert.throws(
    () => validateWeights(joints, weights, 1, JOINTS),
    (e) => e instanceof AssetError && /out of range/.test(e.message + e.problems),
  );
});

await test('negative weight beyond epsilon -> asset rejected', () => {
  const { joints, weights } = makeWeights([
    { j: [0, 1, 0, 0], w: [1.2, -0.2, 0, 0] },
  ]);
  assert.throws(() => validateWeights(joints, weights, 1, JOINTS), AssetError);
});

await test('weight sum != 1 -> renormalized with warning', () => {
  const { joints, weights } = makeWeights([
    { j: [0, 1, 0, 0], w: [0.5, 0.3, 0, 0] },
  ]);
  const r = validateWeights(joints, weights, 1, JOINTS);
  assert.equal(r.warnings.length, 1);
  const sum = r.weights[0] + r.weights[1];
  assert.ok(Math.abs(sum - 1) < 1e-7);
  // ratio preserved
  assert.ok(Math.abs(r.weights[0] / r.weights[1] - 0.5 / 0.3) < 1e-6);
});

await test('tiny negative weight -> clamped, then renormalized', () => {
  const { joints, weights } = makeWeights([
    { j: [0, 1, 0, 0], w: [1.00005, -1e-6, 0, 0] },
  ]);
  const r = validateWeights(joints, weights, 1, JOINTS);
  assert.ok(r.weights[1] === 0);
  assert.ok(Math.abs(r.weights[0] - 1) < 1e-6);
});

await test('zero-influence vertex -> kept rigid, listed in warning', () => {
  const { joints, weights } = makeWeights([
    { j: [0, 0, 0, 0], w: [0, 0, 0, 0] },
    { j: [1, 0, 0, 0], w: [1, 0, 0, 0] },
  ]);
  const r = validateWeights(joints, weights, 2, JOINTS);
  assert.deepEqual(r.zeroInfluence, [0]);
  assert.equal(r.warnings.length, 1);
  assert.ok(/zero-influence/.test(r.warnings[0]));
});

await test('duplicate joint on one vertex -> merged', () => {
  const { joints, weights } = makeWeights([
    { j: [2, 2, 0, 0], w: [0.3, 0.7, 0, 0] },
  ]);
  const r = validateWeights(joints, weights, 1, JOINTS);
  assert.ok(Math.abs(r.weights[0] - 1) < 1e-7);
  assert.equal(r.weights[1], 0);
});

await test('singular inverse bind matrix -> asset rejected', () => {
  const ibms = new Float32Array(32);
  ibms.set(mat4.identity(), 0);
  ibms.set(mat4.scaling(1, 0, 1).concat(), 16); // singular
  assert.throws(() => validateInverseBindMatrices(ibms, 2), AssetError);
});

await test('NaN inside inverse bind matrices -> asset rejected', () => {
  const ibms = new Float32Array(16);
  ibms.set(mat4.identity(), 0);
  ibms[5] = NaN;
  assert.throws(() => validateInverseBindMatrices(ibms, 1), AssetError);
});
