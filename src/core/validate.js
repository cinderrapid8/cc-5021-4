// Load-time validation and repair policy for skinned assets.
//
// Policy summary (also documented in README.md):
//   REJECT the asset (AssetError) when:
//     - any position/normal/weight/IBM/animation value is NaN or Infinity
//     - a joint index is negative or >= joint count
//     - a weight is negative beyond a tiny epsilon (< -1e-4)
//     - an inverse bind matrix is singular
//   REPAIR with a recorded warning when:
//     - a vertex weight sum differs from 1 (finite and > 0): renormalize
//     - a weight is in [-1e-4, 0): clamp to 0, then renormalize
//     - the same joint appears twice on one vertex: merge (sum) the weights
//   ZERO-INFLUENCE vertices (all four weights are 0):
//     - kept, rigidly attached to the mesh node (identity skin), warning lists them
// These rules guarantee no NaN ever reaches the GPU or the CPU skinner.

import { MAX_INFLUENCES } from './format.js';
import { mat4 } from './math.js';

export const NEG_WEIGHT_EPS = 1e-4;
export const SUM_WARN_EPS = 1e-3;
const MAX_LISTED = 16;

export class AssetError extends Error {
  constructor(message, problems) {
    super(message);
    this.name = 'AssetError';
    this.problems = problems || [];
  }
}

function fmtList(arr) {
  if (arr.length <= MAX_LISTED) return arr.join(', ');
  return arr.slice(0, MAX_LISTED).join(', ') + ` … (+${arr.length - MAX_LISTED} more)`;
}

// Validates and repairs skin weights in place (on a float64 copy).
// joints: Uint16Array-like [vertexCount*4], weights: Float32Array-like [vertexCount*4]
// Returns { weights: Float32Array, joints, warnings, zeroInfluence: number[] }
export function validateWeights(joints, weights, vertexCount, jointCount) {
  const errors = [];
  const warnings = [];
  const badIndexVerts = [];
  const nanVerts = [];
  const negVerts = [];
  const rescaledVerts = [];
  const mergedVerts = [];
  const zeroInfluence = [];
  let maxSumDeviation = 0;

  const w = new Float32Array(weights); // repaired output
  const j = new Uint16Array(joints);

  for (let v = 0; v < vertexCount; v++) {
    const o = v * MAX_INFLUENCES;
    let sum = 0;
    let hasNaN = false;
    let hasNeg = false;
    let hasBadIndex = false;

    for (let k = 0; k < MAX_INFLUENCES; k++) {
      const wv = w[o + k];
      if (!Number.isFinite(wv)) hasNaN = true;
      else if (wv < -NEG_WEIGHT_EPS) hasNeg = true;
      if (w[o + k] !== 0) {
        const ji = j[o + k];
        if (!Number.isInteger(ji) || ji < 0 || ji >= jointCount) hasBadIndex = true;
      }
    }
    if (hasNaN) { nanVerts.push(v); continue; }
    if (hasNeg) { negVerts.push(v); continue; }
    if (hasBadIndex) { badIndexVerts.push(v); continue; }

    // clamp tiny negatives
    for (let k = 0; k < MAX_INFLUENCES; k++) {
      if (w[o + k] < 0) w[o + k] = 0;
    }

    // merge duplicate joints (sum weights into first occurrence)
    let merged = false;
    for (let a = 0; a < MAX_INFLUENCES; a++) {
      if (w[o + a] === 0) continue;
      for (let b = a + 1; b < MAX_INFLUENCES; b++) {
        if (w[o + b] !== 0 && j[o + b] === j[o + a]) {
          w[o + a] += w[o + b];
          w[o + b] = 0;
          j[o + b] = 0;
          merged = true;
        }
      }
    }
    if (merged) mergedVerts.push(v);

    for (let k = 0; k < MAX_INFLUENCES; k++) sum += w[o + k];

    if (sum === 0) {
      zeroInfluence.push(v);
      continue; // stays all-zero: skinner treats as rigid to mesh node
    }

    const dev = Math.abs(sum - 1);
    if (dev > SUM_WARN_EPS) {
      rescaledVerts.push(v);
      if (dev > maxSumDeviation) maxSumDeviation = dev;
    }
    if (dev > 0) {
      for (let k = 0; k < MAX_INFLUENCES; k++) w[o + k] /= sum;
    }
  }

  if (nanVerts.length) errors.push(`vertices with NaN/Infinity weights: ${fmtList(nanVerts)}`);
  if (negVerts.length) errors.push(`vertices with negative weights (< -${NEG_WEIGHT_EPS}): ${fmtList(negVerts)}`);
  if (badIndexVerts.length) errors.push(`vertices with joint index out of range [0, ${jointCount}): ${fmtList(badIndexVerts)}`);

  if (errors.length) {
    throw new AssetError(`skin weight validation failed (${errors.length} problem group(s))`, errors);
  }

  if (rescaledVerts.length) {
    warnings.push(`renormalized weights on ${rescaledVerts.length} vertex(es) (max |sum-1| = ${maxSumDeviation.toExponential(2)}): ${fmtList(rescaledVerts)}`);
  }
  if (mergedVerts.length) {
    warnings.push(`merged duplicate joint influences on ${mergedVerts.length} vertex(es): ${fmtList(mergedVerts)}`);
  }
  if (zeroInfluence.length) {
    warnings.push(`${zeroInfluence.length} zero-influence vertex(es) stay rigid to the mesh node: ${fmtList(zeroInfluence)}`);
  }

  return { joints: j, weights: w, warnings, zeroInfluence };
}

// Rejects non-finite data in any numeric payload. name is used in the error.
export function assertFinite(arr, name) {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) {
      throw new AssetError(`${name} contains NaN/Infinity at element ${i}`, [`${name}[${i}] is not finite`]);
    }
  }
}

// Inverse bind matrices must be invertible and finite.
export function validateInverseBindMatrices(ibms, jointCount) {
  assertFinite(ibms, 'inverseBindMatrices');
  const errors = [];
  for (let jdx = 0; jdx < jointCount; jdx++) {
    const m = Array.from(ibms.subarray(jdx * 16, jdx * 16 + 16));
    const inv = mat4.invert(m);
    if (!inv) {
      errors.push(`inverse bind matrix of joint ${jdx} is singular`);
    }
  }
  if (errors.length) {
    throw new AssetError('invalid inverse bind matrices', errors);
  }
}
