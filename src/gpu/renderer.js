// Display renderer: skinned mesh (shaded / joint-heatmap / error-heatmap),
// skeleton overlay, selected-vertex marker. All skinning runs on the GPU.

import { MESH_VS, MESH_FS, LINES_VS, LINES_FS, POINT_VS, POINT_FS } from './shaders.js';
import { linkProgram, programInfo } from './webgl.js';
import { bindAttrib } from './skinner.js';
import { mat4, vec3 } from '../core/math.js';
import { MAX_INFLUENCES } from '../core/format.js';

export class Renderer {
  constructor(gl, asset, skinner) {
    this.gl = gl;
    this.asset = asset;
    this.skinner = skinner;

    this.meshProg = programInfo(gl, linkProgram(gl, MESH_VS, MESH_FS), [
      'aPosition', 'aNormal', 'aJoints', 'aWeights', 'aError',
    ], [
      'uSkinTex', 'uNormalTex', 'uSkinTexW', 'uNormalTexW',
      'uMeshWorld', 'uNormalWorld', 'uViewProj',
      'uColorMode', 'uHeatJoint', 'uBaseColor', 'uLightDir', 'uErrorScale',
    ]);
    this.lineProg = programInfo(gl, linkProgram(gl, LINES_VS, LINES_FS), [
      'aPosition',
    ], ['uViewProj', 'uColor']);
    this.pointProg = programInfo(gl, linkProgram(gl, POINT_VS, POINT_FS), [
      'aPosition', 'aNormal', 'aJoints', 'aWeights',
    ], [
      'uSkinTex', 'uNormalTex', 'uSkinTexW', 'uNormalTexW',
      'uMeshWorld', 'uViewProj', 'uPointSize', 'uColor',
    ]);

    const n = asset.mesh.vertexCount;
    const jointsF = new Float32Array(n * MAX_INFLUENCES);
    for (let i = 0; i < n * MAX_INFLUENCES; i++) jointsF[i] = asset.mesh.joints[i];

    this.meshBuffers = {
      aPosition: makeBuffer(gl, asset.mesh.positions, 3),
      aNormal: makeBuffer(gl, asset.mesh.normals, 3),
      aJoints: makeBuffer(gl, jointsF, 4),
      aWeights: makeBuffer(gl, asset.mesh.weights, 4),
      aError: makeBuffer(gl, new Float32Array(n), 1),
    };
    this.indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, asset.mesh.indices, gl.STATIC_DRAW);
    this.indexCount = asset.mesh.indices.length;
    this.indexType = asset.mesh.indices instanceof Uint16Array ? gl.UNSIGNED_SHORT : gl.UNSIGNED_INT;

    this.lineBuffer = gl.createBuffer();
    this.lineVertCount = 0;

    // joint -> parent joint segments (in node-index space, resolved once)
    const jointSet = new Set(asset.jointNodes);
    this.boneSegments = [];
    asset.jointNodes.forEach((ni) => {
      let p = asset.nodes[ni].parent;
      while (p !== -1 && !jointSet.has(p)) p = asset.nodes[p].parent;
      if (p !== -1) this.boneSegments.push([p, ni]);
    });

    this.colorMode = 0;
    this.heatJoint = 0;
    this.errorScale = 1e-3;
    this.baseColor = [0.82, 0.72, 0.62];
  }

  setErrors(errors) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.meshBuffers.aError.buf);
    gl.bufferData(gl.ARRAY_BUFFER, errors, gl.DYNAMIC_DRAW);
  }

  updateSkeleton(world) {
    const gl = this.gl;
    const pts = [];
    for (const [a, b] of this.boneSegments) {
      pts.push(...world[a].slice(12, 15), ...world[b].slice(12, 15));
    }
    // joint markers as tiny crosses
    const s = 0.02;
    for (const ni of this.asset.jointNodes) {
      const p = world[ni].slice(12, 15);
      pts.push(p[0] - s, p[1], p[2], p[0] + s, p[1], p[2]);
      pts.push(p[0], p[1] - s, p[2], p[0], p[1] + s, p[2]);
      pts.push(p[0], p[1], p[2] - s, p[0], p[1], p[2] + s);
    }
    this.lineVertCount = pts.length / 3;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pts), gl.DYNAMIC_DRAW);
  }

  draw(camera, meshWorld, selectedVertex) {
    const gl = this.gl;
    const { width, height } = gl.canvas;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0.11, 0.12, 0.14, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);

    const proj = mat4.perspective(Math.PI / 4, width / height, 0.05, 100);
    const view = mat4.lookAt(camera.eye(), camera.target, [0, 1, 0]);
    const viewProj = mat4.multiply(proj, view);
    const normalWorld = mat4.transpose(mat4.invert(meshWorld));

    // mesh
    gl.useProgram(this.meshProg.prog);
    this.skinner.bindBoneTextures(this.meshProg);
    gl.uniformMatrix4fv(this.meshProg.uniforms.uMeshWorld, false, meshWorld);
    gl.uniformMatrix4fv(this.meshProg.uniforms.uNormalWorld, false, normalWorld);
    gl.uniformMatrix4fv(this.meshProg.uniforms.uViewProj, false, viewProj);
    gl.uniform1i(this.meshProg.uniforms.uColorMode, this.colorMode);
    gl.uniform1i(this.meshProg.uniforms.uHeatJoint, this.heatJoint);
    gl.uniform3fv(this.meshProg.uniforms.uBaseColor, this.baseColor);
    gl.uniform3fv(this.meshProg.uniforms.uLightDir, vec3.normalize([0.5, 1, 0.8]));
    gl.uniform1f(this.meshProg.uniforms.uErrorScale, this.errorScale);
    for (const [name, buf] of Object.entries(this.meshBuffers)) {
      bindAttrib(gl, this.meshProg.attribs[name], buf);
    }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.drawElements(gl.TRIANGLES, this.indexCount, this.indexType, 0);

    // skeleton overlay (x-ray: visible through the mesh)
    if (this.lineVertCount > 0) {
      gl.disable(gl.DEPTH_TEST);
      gl.useProgram(this.lineProg.prog);
      gl.uniformMatrix4fv(this.lineProg.uniforms.uViewProj, false, viewProj);
      gl.uniform3f(this.lineProg.uniforms.uColor, 0.2, 0.9, 1.0);
      bindAttrib(gl, this.lineProg.attribs.aPosition, { buf: this.lineBuffer, size: 3 });
      gl.drawArrays(gl.LINES, 0, this.lineVertCount);
      gl.enable(gl.DEPTH_TEST);
    }

    // selected vertex marker (skinned on GPU so it tracks deformation)
    if (selectedVertex !== null && selectedVertex >= 0) {
      gl.disable(gl.DEPTH_TEST);
      gl.useProgram(this.pointProg.prog);
      this.skinner.bindBoneTextures(this.pointProg);
      gl.uniformMatrix4fv(this.pointProg.uniforms.uMeshWorld, false, meshWorld);
      gl.uniformMatrix4fv(this.pointProg.uniforms.uViewProj, false, viewProj);
      gl.uniform1f(this.pointProg.uniforms.uPointSize, 10.0);
      gl.uniform3f(this.pointProg.uniforms.uColor, 1.0, 0.9, 0.1);
      bindAttrib(gl, this.pointProg.attribs.aPosition, this.meshBuffers.aPosition);
      bindAttrib(gl, this.pointProg.attribs.aNormal, this.meshBuffers.aNormal);
      bindAttrib(gl, this.pointProg.attribs.aJoints, this.meshBuffers.aJoints);
      bindAttrib(gl, this.pointProg.attribs.aWeights, this.meshBuffers.aWeights);
      gl.drawArrays(gl.POINTS, selectedVertex, 1);
      gl.enable(gl.DEPTH_TEST);
    }
  }
}

function makeBuffer(gl, data, size) {
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  return { buf, size };
}

export class OrbitCamera {
  constructor(target, dist) {
    this.target = target.slice();
    this.dist = dist;
    this.theta = 0.6;
    this.phi = 1.1;
  }
  eye() {
    const sp = Math.sin(this.phi);
    return [
      this.target[0] + this.dist * sp * Math.sin(this.theta),
      this.target[1] + this.dist * Math.cos(this.phi),
      this.target[2] + this.dist * sp * Math.cos(this.theta),
    ];
  }
  rotate(dx, dy) {
    this.theta += dx * 0.01;
    this.phi = Math.min(Math.max(this.phi + dy * 0.01, 0.05), Math.PI - 0.05);
  }
  zoom(d) {
    this.dist = Math.min(Math.max(this.dist * (d > 0 ? 1.1 : 0.9), 0.2), 50);
  }
  serialize() {
    return { target: this.target, dist: this.dist, theta: this.theta, phi: this.phi };
  }
  static deserialize(o) {
    const c = new OrbitCamera(o.target, o.dist);
    c.theta = o.theta;
    c.phi = o.phi;
    return c;
  }
}
