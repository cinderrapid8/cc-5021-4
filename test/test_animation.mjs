import assert from 'node:assert/strict';
import { quat } from '../src/core/math.js';
import { samplePose, clipDuration } from '../src/core/animation.js';
import { loadRunwayAsset } from './helpers.mjs';

const asset = loadRunwayAsset();
const clip = (name) => asset.animations.find((a) => a.name === name);
const nodeIdx = (name) => asset.nodes.findIndex((n) => n.name === name);

await test('clip durations match key ranges', () => {
  assert.ok(Math.abs(clipDuration(clip('elbow_flex')) - 1.0) < 1e-6);
  assert.ok(Math.abs(clipDuration(clip('forearm_twist')) - 1.2) < 1e-6);
  assert.ok(Math.abs(clipDuration(clip('big_turn')) - 2.0) < 1e-6);
});

await test('sampling hits exact key values at key times', () => {
  const c = clip('elbow_flex');
  const fore = nodeIdx('L_Forearm');
  const mid = samplePose(asset, c, 0.5);
  const expected = quat.fromAxisAngle([0, 1, 0], -115 * Math.PI / 180);
  for (let i = 0; i < 4; i++) assert.ok(Math.abs(mid[fore].q[i] - expected[i]) < 1e-6);
  const start = samplePose(asset, c, 0.0);
  assert.deepEqual(start[fore].q, [0, 0, 0, 1]);
});

await test('mid-key interpolation is slerp (half angle)', () => {
  const c = clip('elbow_flex');
  const fore = nodeIdx('L_Forearm');
  const p = samplePose(asset, c, 0.25);
  const expected = quat.fromAxisAngle([0, 1, 0], -57.5 * Math.PI / 180);
  for (let i = 0; i < 4; i++) assert.ok(Math.abs(p[fore].q[i] - expected[i]) < 1e-5);
});

await test('translation channel interpolates linearly', () => {
  const c = clip('big_turn');
  const hips = nodeIdx('Hips');
  const p = samplePose(asset, c, 0.5);
  // halfway between [0,95,0] and [25,95,18]
  assert.ok(Math.abs(p[hips].t[0] - 12.5) < 1e-4);
  assert.ok(Math.abs(p[hips].t[1] - 95) < 1e-4);
  assert.ok(Math.abs(p[hips].t[2] - 9) < 1e-4);
});

await test('time is clamped to the clip range', () => {
  const c = clip('elbow_flex');
  const fore = nodeIdx('L_Forearm');
  const p = samplePose(asset, c, 99);
  assert.deepEqual(p[fore].q, [0, 0, 0, 1]);
});

await test('untouched nodes keep bind TRS during animation', () => {
  const c = clip('elbow_flex');
  const head = nodeIdx('Head');
  const p = samplePose(asset, c, 0.5);
  assert.deepEqual(p[head].q, [0, 0, 0, 1]);
  assert.deepEqual(p[head].t, asset.nodes[head].translation);
});
