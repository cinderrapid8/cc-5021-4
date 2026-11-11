// GLSL ES 1.00 shader sources. The app requests a WebGL1 context, which every
// WebGL2-capable browser also provides, so one shader set serves both.

import { GLSL_ENCODE_CONSTS, GLSL_ENCODE_FN } from '../core/pack.js';

// Shared skinning chunk: same math as src/core/skinning.js.
// Bone matrices come from float textures (4 texels per joint = mat4 columns,
// 3 texels per joint = normal-matrix columns).
export const SKINNING_CHUNK = `
attribute vec3 aPosition;
attribute vec3 aNormal;
attribute vec4 aJoints;
attribute vec4 aWeights;

uniform sampler2D uSkinTex;
uniform sampler2D uNormalTex;
uniform float uSkinTexW;
uniform float uNormalTexW;

vec4 boneColumn(int j, int c) {
  float x = (float(j * 4 + c) + 0.5) / uSkinTexW;
  return texture2D(uSkinTex, vec2(x, 0.5));
}

mat4 boneMatrix(int j) {
  return mat4(boneColumn(j, 0), boneColumn(j, 1), boneColumn(j, 2), boneColumn(j, 3));
}

vec3 normalColumn(int j, int c) {
  float x = (float(j * 3 + c) + 0.5) / uNormalTexW;
  return texture2D(uNormalTex, vec2(x, 0.5)).xyz;
}

mat3 normalMatrix(int j) {
  return mat3(normalColumn(j, 0), normalColumn(j, 1), normalColumn(j, 2));
}

// Zero-influence vertices (all weights zero) stay rigid to the mesh node:
// they keep their authored position and normal.
void skinVertex(out vec3 pos, out vec3 nrm) {
  pos = vec3(0.0);
  nrm = vec3(0.0);
  float totalW = 0.0;
  for (int k = 0; k < 4; k++) {
    float w = aWeights[k];
    totalW += w;
    if (w > 0.0) {
      int j = int(aJoints[k] + 0.5);
      pos += w * (boneMatrix(j) * vec4(aPosition, 1.0)).xyz;
      nrm += w * (normalMatrix(j) * aNormal);
    }
  }
  if (totalW == 0.0) {
    pos = aPosition;
    nrm = aNormal;
  } else if (dot(nrm, nrm) > 0.0) {
    nrm = normalize(nrm);
  } else {
    nrm = aNormal; // degenerate blend (opposing normals cancel): keep authored
  }
}
`;

// ---- display pass ----

export const MESH_VS = `
${SKINNING_CHUNK}
uniform mat4 uMeshWorld;
uniform mat4 uNormalWorld; // inverse-transpose of meshWorld (upper 3x3 in mat4)
uniform mat4 uViewProj;
uniform int uColorMode;    // 0 = shaded, 1 = joint-weight heatmap, 2 = error heatmap
uniform int uHeatJoint;
attribute float aError;    // per-vertex |CPU-GPU| error, filled after a check
varying vec3 vNormalW;
varying vec3 vPosW;
varying float vHeat;
varying float vError;

void main() {
  vec3 pos;
  vec3 nrm;
  skinVertex(pos, nrm);
  vec4 world = uMeshWorld * vec4(pos, 1.0);
  vPosW = world.xyz;
  vNormalW = normalize((uNormalWorld * vec4(nrm, 0.0)).xyz);
  vHeat = 0.0;
  if (uColorMode == 1) {
    for (int k = 0; k < 4; k++) {
      if (int(aJoints[k] + 0.5) == uHeatJoint) vHeat += aWeights[k];
    }
  }
  vError = aError;
  gl_Position = uViewProj * world;
}
`;

export const MESH_FS = `
precision highp float;
varying vec3 vNormalW;
varying vec3 vPosW;
varying float vHeat;
varying float vError;
uniform int uColorMode;
uniform vec3 uBaseColor;
uniform vec3 uLightDir;
uniform float uErrorScale; // error value that maps to full red

vec3 heatRamp(float t) {
  t = clamp(t, 0.0, 1.0);
  return mix(vec3(0.05, 0.1, 0.35), vec3(1.0, 0.15, 0.0), t)
       + vec3(0.0, 0.6, 0.2) * (1.0 - abs(t - 0.5) * 2.0) * 0.5;
}

void main() {
  if (uColorMode == 1) {
    gl_FragColor = vec4(heatRamp(vHeat), 1.0);
    return;
  }
  if (uColorMode == 2) {
    gl_FragColor = vec4(heatRamp(vError / uErrorScale), 1.0);
    return;
  }
  vec3 n = normalize(vNormalW);
  float diff = max(dot(n, normalize(uLightDir)), 0.0);
  vec3 col = uBaseColor * (0.25 + 0.75 * diff);
  gl_FragColor = vec4(col, 1.0);
}
`;

// ---- GPU compute/readback pass ----
// Renders one point per (vertex, component-slot). Slot 0..2 = position xyz,
// slot 3..5 = normal xyz. The fragment shader writes the value as 32-bit
// fixed point in RGBA8 so results are exact on any WebGL1 implementation.

export const COMPUTE_VS = `
${SKINNING_CHUNK}
attribute float aVertexId;
attribute float aSlot;
uniform vec2 uComputeSize; // tiled texture size
varying float vValue;

float pickComponent(vec3 a, vec3 b, float slot) {
  if (slot < 0.5) return a.x;
  if (slot < 1.5) return a.y;
  if (slot < 2.5) return a.z;
  if (slot < 3.5) return b.x;
  if (slot < 4.5) return b.y;
  return b.z;
}

void main() {
  vec3 pos;
  vec3 nrm;
  skinVertex(pos, nrm);
  vValue = pickComponent(pos, nrm, aSlot);
  float idx = aVertexId * 6.0 + aSlot;
  float x = mod(idx, uComputeSize.x);
  float y = floor(idx / uComputeSize.x);
  vec2 ndc = (vec2(x + 0.5, y + 0.5) / uComputeSize) * 2.0 - 1.0;
  gl_Position = vec4(ndc, 0.0, 1.0);
  gl_PointSize = 1.0;
}
`;

export const COMPUTE_FS = `
precision highp float;
${GLSL_ENCODE_CONSTS}
${GLSL_ENCODE_FN}
varying float vValue;

void main() {
  gl_FragColor = encodeFixed(vValue);
}
`;

// ---- skeleton overlay ----

export const LINES_VS = `
attribute vec3 aPosition; // world space
uniform mat4 uViewProj;
void main() {
  gl_Position = uViewProj * vec4(aPosition, 1.0);
}
`;

export const LINES_FS = `
precision highp float;
uniform vec3 uColor;
void main() {
  gl_FragColor = vec4(uColor, 1.0);
}
`;

// ---- selected vertex marker ----

export const POINT_VS = `
${SKINNING_CHUNK}
uniform mat4 uMeshWorld;
uniform mat4 uViewProj;
uniform float uPointSize;
void main() {
  vec3 pos;
  vec3 nrm;
  skinVertex(pos, nrm);
  gl_Position = uViewProj * uMeshWorld * vec4(pos, 1.0);
  gl_PointSize = uPointSize;
}
`;

export const POINT_FS = `
precision highp float;
uniform vec3 uColor;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  gl_FragColor = vec4(uColor, 1.0);
}
`;
