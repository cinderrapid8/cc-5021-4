// Shared helpers: load the generated asset from disk.

import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadAsset, buildSkeleton, computeWorldMatrices, computeSkinMatrices } from '../src/core/loader.js';
import { bindPose, samplePose } from '../src/core/animation.js';

const ASSET_PATH = join(dirname(fileURLToPath(import.meta.url)), '..', 'assets', 'runway_character.rskin.json');

export function loadRunwayAsset() {
  const json = JSON.parse(readFileSync(ASSET_PATH, 'utf8'));
  return loadAsset(json);
}

// Full CPU pose evaluation: returns { world, skin, normals, meshWorld }.
export function evaluatePose(asset, skeleton, clipName, time) {
  const clip = clipName
    ? asset.animations.find((a) => a.name === clipName)
    : null;
  const pose = clip ? samplePose(asset, clip, time) : bindPose(asset);
  const world = computeWorldMatrices(asset, skeleton, pose);
  const { skin, normals, meshWorld } = computeSkinMatrices(asset, world);
  return { pose, world, skin, normals, meshWorld };
}

export function makeSkeleton(asset) {
  return buildSkeleton(asset);
}

export function maxAbsDiff(a, b, n) {
  let m = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}
