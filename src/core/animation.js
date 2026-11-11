// Animation clip sampling: STEP / LINEAR, quaternions use shortest-path slerp.

import { quat } from './math.js';

export function clipDuration(clip) {
  let d = 0;
  for (const ch of clip.channels) {
    const t = ch.times[ch.times.length - 1];
    if (t > d) d = t;
  }
  return d;
}

function sampleChannel(ch, time) {
  const times = ch.times;
  const comp = ch.components;
  if (time <= times[0]) return Array.from(ch.values.subarray(0, comp));
  const last = times.length - 1;
  if (time >= times[last]) {
    return Array.from(ch.values.subarray(last * comp, times.length * comp));
  }
  // find k such that times[k] <= time < times[k+1]
  let k = 0;
  while (k < last - 1 && times[k + 1] <= time) k++;
  const a = ch.values.subarray(k * comp, k * comp + comp);
  const b = ch.values.subarray((k + 1) * comp, (k + 1) * comp + comp);
  if (ch.interpolation === 'STEP') return Array.from(a);
  const t = (time - times[k]) / (times[k + 1] - times[k]);
  if (ch.path === 'rotation') {
    return quat.slerp(Array.from(a), Array.from(b), t);
  }
  const out = new Array(comp);
  for (let c = 0; c < comp; c++) out[c] = a[c] + (b[c] - a[c]) * t;
  return out;
}

// Builds the bind pose (per node {t,q,s}).
export function bindPose(asset) {
  return asset.nodes.map((n) => ({
    t: n.translation.slice(),
    q: n.rotation.slice(),
    s: n.scale.slice(),
  }));
}

// Samples a clip at `time` (seconds) over the bind pose. Unknown channels keep
// bind values. time is clamped to the clip range.
export function samplePose(asset, clip, time) {
  const pose = bindPose(asset);
  const dur = clipDuration(clip);
  const t = Math.min(Math.max(time, 0), dur);
  for (const ch of clip.channels) {
    const v = sampleChannel(ch, t);
    const p = pose[ch.node];
    if (ch.path === 'translation') p.t = v;
    else if (ch.path === 'rotation') p.q = quat.normalize(v);
    else p.s = v;
  }
  return pose;
}
