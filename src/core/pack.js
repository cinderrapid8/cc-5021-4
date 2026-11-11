// Shared data layout between the CPU packer and the GLSL skinning shader.
//
// Bone matrices are uploaded in two RGBA float textures:
//   - skin texture: 4 texels per joint = the 4 columns of the mat4
//   - normal texture: 3 texels per joint = the 3 columns of the 3x3 normal matrix
// Textures are 1 pixel tall, NEAREST filtered, sampled in the vertex shader.
//
// The GPU->CPU readback pass cannot rely on float framebuffers on every
// WebGL1 implementation, so skinned values are written as 32-bit fixed point
// packed into RGBA8 (same scheme as classic depth packing). With the range
// below the quantum is ~6e-8, far below float32 vertex precision.

export const ENCODE_MIN = -256.0;
export const ENCODE_MAX = 256.0;

export function boneTextureSize(jointCount) {
  return { skinWidth: jointCount * 4, normalWidth: jointCount * 3 };
}

// Packs skin matrices (Float32Array 16*J, column-major) into RGBA pixels.
export function packSkinMatrices(skinMats, jointCount) {
  const out = new Float32Array(jointCount * 4 * 4);
  for (let j = 0; j < jointCount; j++) {
    for (let c = 0; c < 4; c++) {
      const dst = (j * 4 + c) * 4;
      out[dst] = skinMats[j * 16 + c * 4];
      out[dst + 1] = skinMats[j * 16 + c * 4 + 1];
      out[dst + 2] = skinMats[j * 16 + c * 4 + 2];
      out[dst + 3] = skinMats[j * 16 + c * 4 + 3];
    }
  }
  return out;
}

// Packs normal matrices (Float32Array 9*J, column-major 3x3) as 3 column texels.
export function packNormalMatrices(normalMats, jointCount) {
  const out = new Float32Array(jointCount * 3 * 4);
  for (let j = 0; j < jointCount; j++) {
    for (let c = 0; c < 3; c++) {
      const dst = (j * 3 + c) * 4;
      out[dst] = normalMats[j * 9 + c * 3];
      out[dst + 1] = normalMats[j * 9 + c * 3 + 1];
      out[dst + 2] = normalMats[j * 9 + c * 3 + 2];
      out[dst + 3] = 0;
    }
  }
  return out;
}

// GLSL ES 1.00 snippets shared by shaders.
export const GLSL_ENCODE_CONSTS = `
const float ENC_MIN = ${ENCODE_MIN.toFixed(1)};
const float ENC_MAX = ${ENCODE_MAX.toFixed(1)};
`;

export const GLSL_ENCODE_FN = `
vec4 encodeFixed(float v) {
  float x = clamp((v - ENC_MIN) / (ENC_MAX - ENC_MIN), 0.0, 1.0 - 1.0 / 16777216.0);
  vec4 r = vec4(1.0, 255.0, 65025.0, 16581375.0) * x;
  r = fract(r);
  r -= r.yzww * (1.0 / 255.0);
  return r;
}
`;

// Decodes one RGBA8 pixel back to a float in [ENCODE_MIN, ENCODE_MAX].
export function decodeFixed(r, g, b, a) {
  const x = r / 255.0 + g / (255.0 * 255.0) + b / (255.0 * 65025.0) + a / (255.0 * 16581375.0);
  return x * (ENCODE_MAX - ENCODE_MIN) + ENCODE_MIN;
}
