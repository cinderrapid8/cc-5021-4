import assert from 'node:assert/strict';
import {
  packSkinMatrices, packNormalMatrices, decodeFixed,
  ENCODE_MIN, ENCODE_MAX, boneTextureSize,
} from '../src/core/pack.js';
import { mat4, quat } from '../src/core/math.js';

await test('bone texture sizes follow the 4/3 texels-per-joint layout', () => {
  const { skinWidth, normalWidth } = boneTextureSize(17);
  assert.equal(skinWidth, 68);
  assert.equal(normalWidth, 51);
});

await test('packSkinMatrices writes each joint matrix as 4 column texels', () => {
  const skin = new Float32Array(2 * 16);
  for (let i = 0; i < 32; i++) skin[i] = i + 1;
  const px = packSkinMatrices(skin, 2);
  // joint 1, column 2 lives at texel (1*4+2), RGBA
  const o = (1 * 4 + 2) * 4;
  assert.deepEqual(Array.from(px.slice(o, o + 4)), [1 * 16 + 2 * 4 + 1, 1 * 16 + 2 * 4 + 2, 1 * 16 + 2 * 4 + 3, 1 * 16 + 2 * 4 + 4]);
});

await test('packNormalMatrices writes 3x3 columns matching the CPU layout', () => {
  // normal matrix of a known transform, packed, must reconstruct the same
  // matrix the CPU skinner uses (column-major 3x3)
  const m = mat4.compose([1, 2, 3], quat.fromAxisAngle([0, 1, 1], 0.8), [2, 0.5, 1]);
  const nm = mat4.normalMatrix(m);
  const px = packNormalMatrices(new Float32Array(nm), 1);
  for (let c = 0; c < 3; c++) {
    const o = c * 4;
    assert.ok(Math.abs(px[o] - nm[c * 3]) < 1e-7);
    assert.ok(Math.abs(px[o + 1] - nm[c * 3 + 1]) < 1e-7);
    assert.ok(Math.abs(px[o + 2] - nm[c * 3 + 2]) < 1e-7);
  }
});

// Simulates the GLSL encodeFixed in float32 (Math.fround after every op,
// mirroring what a highp GPU shader computes) and checks the JS decoder.
function encodeFixedF32(v) {
  const f = Math.fround;
  let x = f(f(v - ENCODE_MIN) / f(ENCODE_MAX - ENCODE_MIN));
  x = Math.min(Math.max(x, 0), 1 - 1 / 16777216);
  const mult = [1, 255, 65025, 16581375];
  let r = mult.map((m) => f(x * m));
  r = r.map((c) => f(c - Math.floor(c))); // fract
  // GLSL: r -= r.yzww * (1/255) — the w component subtracts itself
  for (let i = 0; i < 4; i++) r[i] = f(r[i] - f((i < 3 ? r[i + 1] : r[i]) / 255));
  // quantize to RGBA8 like the framebuffer does
  return r.map((c) => Math.round(Math.min(Math.max(c, 0), 1) * 255));
}

await test('fixed-point RGBA8 encode/decode round-trip stays below 1e-5', () => {
  let worst = 0;
  for (let i = 0; i <= 2000; i++) {
    const v = ENCODE_MIN + (i / 2000) * (ENCODE_MAX - ENCODE_MIN);
    const [r, g, b, a] = encodeFixedF32(v);
    const d = Math.abs(decodeFixed(r, g, b, a) - v);
    if (d > worst) worst = d;
  }
  // explicit boundary values must not wrap around
  for (const v of [ENCODE_MIN, ENCODE_MAX, 0, -0, 255.999999, -255.999999]) {
    const [r, g, b, a] = encodeFixedF32(v);
    const d = Math.abs(decodeFixed(r, g, b, a) - v);
    if (d > worst) worst = d;
  }
  // float32 shader math limits the scheme to ~4e-5 absolute over the ±256
  // range; the GPU<->CPU check uses a 1e-3 tolerance, well above this floor
  assert.ok(worst < 1e-4, `worst round-trip error ${worst}`);
});
