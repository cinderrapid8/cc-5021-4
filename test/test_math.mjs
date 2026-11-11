import assert from 'node:assert/strict';
import { mat4, quat, vec3 } from '../src/core/math.js';

await test('mat4 multiply matches sequential transform application', () => {
  const t = mat4.translation(1, 2, 3);
  const r = mat4.fromQuat(quat.fromAxisAngle([0, 1, 0], Math.PI / 2));
  const s = mat4.scaling(2, 3, 4);
  const m = mat4.multiply(t, mat4.multiply(r, s));
  const p = [1, 0, 0];
  const step = mat4.transformPoint(t, mat4.transformPoint(r, mat4.transformPoint(s, p)));
  const once = mat4.transformPoint(m, p);
  assert.ok(vec3.length(vec3.sub(step, once)) < 1e-12);
});

await test('mat4 invert round-trips under non-uniform scale + rotation + translation', () => {
  const m = mat4.compose([5, -3, 2], quat.fromAxisAngle([1, 2, 3], 0.7), [0.011, 1, 9.5]);
  const inv = mat4.invert(m);
  assert.ok(inv, 'invertible');
  assert.ok(mat4.almostEqual(mat4.multiply(m, inv), mat4.identity(), 1e-10));
  assert.ok(mat4.almostEqual(mat4.multiply(inv, m), mat4.identity(), 1e-10));
});

await test('mat4 invert returns null for singular matrices', () => {
  const m = mat4.scaling(1, 0, 1);
  assert.equal(mat4.invert(m), null);
});

await test('normal matrix keeps normals perpendicular under non-uniform scale', () => {
  // Plane z = 0 with tangent (1,0,0) and normal (0,0,1), transformed by a
  // non-uniform scale + rotation. The inverse-transpose normal must remain
  // perpendicular to the transformed tangent.
  const m = mat4.compose([1, 2, 3], quat.fromAxisAngle([0, 1, 1], 0.9), [4, 0.5, 2]);
  const nm = mat4.normalMatrix(m);
  const tangent = mat4.transformDir(m, [1, 0, 0]);
  const normal = [
    nm[0] * 0 + nm[3] * 0 + nm[6] * 1,
    nm[1] * 0 + nm[4] * 0 + nm[7] * 1,
    nm[2] * 0 + nm[5] * 0 + nm[8] * 1,
  ];
  const dot = vec3.dot(vec3.normalize(tangent), vec3.normalize(normal));
  assert.ok(Math.abs(dot) < 1e-10, `dot=${dot}`);
});

await test('quat slerp midpoint is the half-angle rotation', () => {
  const a = quat.identity();
  const b = quat.fromAxisAngle([0, 1, 0], Math.PI / 2);
  const mid = quat.slerp(a, b, 0.5);
  const expected = quat.fromAxisAngle([0, 1, 0], Math.PI / 4);
  for (let i = 0; i < 4; i++) assert.ok(Math.abs(mid[i] - expected[i]) < 1e-12);
});

await test('quat slerp takes the short path across hemispheres', () => {
  const a = quat.identity();
  const b = [-0, -0, -0, -1]; // same rotation, opposite hemisphere
  const mid = quat.slerp(a, b, 0.5);
  assert.ok(Math.abs(mid[3]) > 0.9, 'did not flip the long way');
});

await test('quat rotateVec3 matches matrix rotation', () => {
  const q = quat.fromAxisAngle([1, 2, 3], 1.1);
  const m = mat4.fromQuat(q);
  const v = [0.3, -0.7, 1.2];
  const a = quat.rotateVec3(q, v);
  const b = mat4.transformDir(m, v);
  assert.ok(vec3.length(vec3.sub(a, b)) < 1e-12);
});
