// GPU skinning + numeric readback.
//
// The compute pass renders 6 points per vertex (position xyz + normal xyz)
// into an RGBA8 texture, each value encoded as 32-bit fixed point. Reading it
// back gives the exact vertex-shader output on any WebGL1 implementation,
// which the inspector compares against the CPU reference.

import { COMPUTE_VS, COMPUTE_FS } from './shaders.js';
import {
  linkProgram, programInfo, makeFloatTexture, uploadFloatTexture,
} from './webgl.js';
import {
  boneTextureSize, packSkinMatrices, packNormalMatrices, decodeFixed,
} from '../core/pack.js';
import { MAX_INFLUENCES } from '../core/format.js';

const COMPUTE_WIDTH = 1024;

export class GpuSkinner {
  constructor(gl, asset) {
    this.gl = gl;
    this.asset = asset;
    const n = asset.mesh.vertexCount;
    this.vertexCount = n;
    this.jointCount = asset.jointNodes.length;

    const prog = linkProgram(gl, COMPUTE_VS, COMPUTE_FS);
    this.prog = programInfo(gl, prog, [
      'aPosition', 'aNormal', 'aJoints', 'aWeights', 'aVertexId', 'aSlot',
    ], ['uSkinTex', 'uNormalTex', 'uSkinTexW', 'uNormalTexW', 'uComputeSize']);

    // --- compute geometry: 6 points per vertex ---
    const points = n * 6;
    const pos = new Float32Array(points * 3);
    const nrm = new Float32Array(points * 3);
    const jnt = new Float32Array(points * 4);
    const wgt = new Float32Array(points * 4);
    const vid = new Float32Array(points);
    const slot = new Float32Array(points);
    for (let v = 0; v < n; v++) {
      for (let s = 0; s < 6; s++) {
        const p = v * 6 + s;
        pos.set(asset.mesh.positions.subarray(v * 3, v * 3 + 3), p * 3);
        nrm.set(asset.mesh.normals.subarray(v * 3, v * 3 + 3), p * 3);
        for (let k = 0; k < MAX_INFLUENCES; k++) {
          jnt[p * 4 + k] = asset.mesh.joints[v * MAX_INFLUENCES + k];
          wgt[p * 4 + k] = asset.mesh.weights[v * MAX_INFLUENCES + k];
        }
        vid[p] = v;
        slot[p] = s;
      }
    }
    this.buffers = {
      aPosition: makeArrayBuffer(gl, pos, 3),
      aNormal: makeArrayBuffer(gl, nrm, 3),
      aJoints: makeArrayBuffer(gl, jnt, 4),
      aWeights: makeArrayBuffer(gl, wgt, 4),
      aVertexId: makeArrayBuffer(gl, vid, 1),
      aSlot: makeArrayBuffer(gl, slot, 1),
    };
    this.pointCount = points;

    // --- bone textures ---
    const { skinWidth, normalWidth } = boneTextureSize(this.jointCount);
    this.skinTexW = skinWidth;
    this.normalTexW = normalWidth;
    this.skinTex = makeFloatTexture(gl, skinWidth, 1);
    this.normalTex = makeFloatTexture(gl, normalWidth, 1);

    // --- compute target ---
    const total = points;
    this.computeW = Math.min(COMPUTE_WIDTH, total);
    this.computeH = Math.ceil(total / this.computeW);
    this.computeTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.computeTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.computeW, this.computeH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.computeTex, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`compute framebuffer incomplete: ${status}`);
    }
  }

  // Uploads per-frame bone data (Float32Array 16*J and 9*J).
  setPose(skinMats, normalMats) {
    const gl = this.gl;
    uploadFloatTexture(gl, this.skinTex, this.skinTexW, 1,
      packSkinMatrices(skinMats, this.jointCount));
    uploadFloatTexture(gl, this.normalTex, this.normalTexW, 1,
      packNormalMatrices(normalMats, this.jointCount));
  }

  bindBoneTextures(progInfo) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.skinTex);
    gl.uniform1i(progInfo.uniforms.uSkinTex, 0);
    gl.uniform1f(progInfo.uniforms.uSkinTexW, this.skinTexW);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.normalTex);
    gl.uniform1i(progInfo.uniforms.uNormalTex, 1);
    gl.uniform1f(progInfo.uniforms.uNormalTexW, this.normalTexW);
  }

  // Runs the compute pass and decodes every vertex.
  // Returns { positions: Float64Array(3N), normals: Float64Array(3N) } in model space.
  skinMeshGPU() {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, this.computeW, this.computeH);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(this.prog.prog);
    this.bindBoneTextures(this.prog);
    gl.uniform2f(this.prog.uniforms.uComputeSize, this.computeW, this.computeH);
    for (const [name, buf] of Object.entries(this.buffers)) {
      bindAttrib(gl, this.prog.attribs[name], buf);
    }
    gl.drawArrays(gl.POINTS, 0, this.pointCount);

    const px = new Uint8Array(this.computeW * this.computeH * 4);
    gl.readPixels(0, 0, this.computeW, this.computeH, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    const positions = new Float64Array(this.vertexCount * 3);
    const normals = new Float64Array(this.vertexCount * 3);
    for (let v = 0; v < this.vertexCount; v++) {
      for (let s = 0; s < 6; s++) {
        const idx = v * 6 + s;
        const o = idx * 4;
        const val = decodeFixed(px[o], px[o + 1], px[o + 2], px[o + 3]);
        if (s < 3) positions[v * 3 + s] = val;
        else normals[v * 3 + (s - 3)] = val;
      }
    }
    return { positions, normals };
  }
}

function makeArrayBuffer(gl, data, size) {
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  return { buf, size };
}

export function bindAttrib(gl, location, buffer) {
  if (location < 0) return;
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer.buf);
  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, buffer.size, gl.FLOAT, false, 0, 0);
}
