// CPU reference implementation of linear blend skinning.
// Mirrors the GLSL path exactly (same matrices, same normal handling) so the
// GPU result can be checked numerically against this.
//
// Positions: p' = sum_k w_k * (skin[j_k] * p)          (model space)
// Normals:   n' = normalize( sum_k w_k * (normalMat[j_k] * n) )
// Zero-influence vertices (all weights 0) stay rigid to the mesh node.

import { MAX_INFLUENCES } from './format.js';
import { vec3 } from './math.js';

export function skinVertexCPU(skinMats, normalMats, joints, weights, v, pos, nrm) {
  const o = v * MAX_INFLUENCES;
  const v3 = v * 3;
  const p = [pos[v3], pos[v3 + 1], pos[v3 + 2]];
  const n = [nrm[v3], nrm[v3 + 1], nrm[v3 + 2]];
  const outP = [0, 0, 0];
  const outN = [0, 0, 0];
  let totalW = 0;
  for (let k = 0; k < MAX_INFLUENCES; k++) {
    const w = weights[o + k];
    if (w === 0) continue;
    totalW += w;
    const j = joints[o + k];
    const m = j * 16;
    outP[0] += w * (skinMats[m] * p[0] + skinMats[m + 4] * p[1] + skinMats[m + 8] * p[2] + skinMats[m + 12]);
    outP[1] += w * (skinMats[m + 1] * p[0] + skinMats[m + 5] * p[1] + skinMats[m + 9] * p[2] + skinMats[m + 13]);
    outP[2] += w * (skinMats[m + 2] * p[0] + skinMats[m + 6] * p[1] + skinMats[m + 10] * p[2] + skinMats[m + 14]);
    const nm = j * 9; // column-major 3x3 (extracted from the column-major mat4)
    outN[0] += w * (normalMats[nm] * n[0] + normalMats[nm + 3] * n[1] + normalMats[nm + 6] * n[2]);
    outN[1] += w * (normalMats[nm + 1] * n[0] + normalMats[nm + 4] * n[1] + normalMats[nm + 7] * n[2]);
    outN[2] += w * (normalMats[nm + 2] * n[0] + normalMats[nm + 5] * n[1] + normalMats[nm + 8] * n[2]);
  }
  // Zero-influence vertices stay rigid to the mesh node (identity skin).
  if (totalW === 0) return { position: p, normal: n };
  // Degenerate blend (opposing normals cancel): keep the authored normal.
  const len = vec3.length(outN);
  return { position: outP, normal: len > 0 ? vec3.scale(outN, 1 / len) : n };
}

// Skins the whole mesh. Returns { positions: Float64Array, normals: Float64Array } in model space.
export function skinMeshCPU(asset, skinMats, normalMats) {
  const count = asset.mesh.vertexCount;
  const positions = new Float64Array(count * 3);
  const normals = new Float64Array(count * 3);
  const { joints, weights, positions: pos, normals: nrm } = asset.mesh;
  for (let v = 0; v < count; v++) {
    const r = skinVertexCPU(skinMats, normalMats, joints, weights, v, pos, nrm);
    positions.set(r.position, v * 3);
    normals.set(r.normal, v * 3);
  }
  return { positions, normals };
}
