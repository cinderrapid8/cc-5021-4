// Inspector app wiring: asset loading, pose evaluation, GPU skinning,
// CPU<->GPU numeric check, vertex picking, and reproducible config save/load.

import {
  loadAsset, buildSkeleton, computeWorldMatrices, computeSkinMatrices,
} from '../core/loader.js';
import { bindPose, samplePose, clipDuration } from '../core/animation.js';
import { skinMeshCPU } from '../core/skinning.js';
import { mat4 } from '../core/math.js';
import { MAX_INFLUENCES } from '../core/format.js';
import { createContext, checkCapabilities } from '../gpu/webgl.js';
import { GpuSkinner } from '../gpu/skinner.js';
import { Renderer, OrbitCamera } from '../gpu/renderer.js';

const DEFAULT_ASSET_URL = new URL('../../assets/runway_character.rskin.json', import.meta.url);

const $ = (id) => document.getElementById(id);
const canvas = $('glcanvas');

const state = {
  asset: null,
  skeleton: null,
  assetName: null,
  gl: null,
  skinner: null,
  renderer: null,
  camera: new OrbitCamera([1.5, 0.95, -2.0], 4.5),
  clipName: null, // null = bind pose
  time: 0,
  playing: true,
  selectedVertex: null,
  colorMode: 0,
  heatJoint: 0,
  showSkeleton: true,
  continuousCheck: false,
  lastCheck: null, // { maxPos, maxNrm, worstVertex, perVertex: Float32Array }
  frame: null,    // per-frame evaluated pose data
};

// ---------- pose evaluation ----------

function evaluateFrame() {
  const { asset, skeleton, clipName, time } = state;
  const clip = clipName ? asset.animations.find((a) => a.name === clipName) : null;
  const pose = clip ? samplePose(asset, clip, time) : bindPose(asset);
  const world = computeWorldMatrices(asset, skeleton, pose);
  const sm = computeSkinMatrices(asset, world);
  state.frame = { world, skin: sm.skin, normals: sm.normals, meshWorld: sm.meshWorld };
  return state.frame;
}

// ---------- rendering ----------

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

let lastT = 0;
function frameLoop(tms) {
  requestAnimationFrame(frameLoop);
  if (!state.asset) return;
  const t = tms / 1000;
  const dt = Math.min(t - lastT, 0.1);
  lastT = t;

  const clip = state.clipName ? state.asset.animations.find((a) => a.name === state.clipName) : null;
  if (state.playing && clip) {
    const dur = clipDuration(clip);
    state.time = (state.time + dt) % dur;
    syncTimeUI();
  }

  resizeCanvas();
  const f = evaluateFrame();
  state.skinner.setPose(f.skin, f.normals);
  state.renderer.colorMode = state.colorMode;
  state.renderer.heatJoint = state.heatJoint;
  if (state.showSkeleton) state.renderer.updateSkeleton(f.world);
  else state.renderer.lineVertCount = 0;
  state.renderer.draw(state.camera, f.meshWorld, state.selectedVertex);

  if (state.continuousCheck) {
    runCheck(false);
    if (frameCount % 10 === 0) {
      renderCheckPanel();
      if (state.selectedVertex !== null) updateVertexPanel();
    }
  }
  frameCount++;
}

let frameCount = 0;

// ---------- numeric check ----------

function runCheck(announce = true) {
  if (!state.asset || !state.frame) return null;
  const f = state.frame;
  const cpu = skinMeshCPU(state.asset, f.skin, f.normals);
  const gpu = state.skinner.skinMeshGPU();
  const n = state.asset.mesh.vertexCount;
  const perVertex = new Float32Array(n);
  let maxPos = 0;
  let maxNrm = 0;
  let worst = 0;
  for (let v = 0; v < n; v++) {
    let e = 0;
    for (let i = 0; i < 3; i++) {
      e = Math.max(e, Math.abs(cpu.positions[v * 3 + i] - gpu.positions[v * 3 + i]));
      maxNrm = Math.max(maxNrm, Math.abs(cpu.normals[v * 3 + i] - gpu.normals[v * 3 + i]));
    }
    perVertex[v] = e;
    if (e > maxPos) { maxPos = e; worst = v; }
  }
  state.lastCheck = { maxPos, maxNrm, worstVertex: worst, perVertex, cpu, gpu };
  state.renderer.setErrors(perVertex);
  if (announce) renderCheckPanel();
  return state.lastCheck;
}

function renderCheckPanel() {
  const el = $('check-result');
  const c = state.lastCheck;
  if (!c) { el.textContent = '尚未运行'; return; }
  const cls = c.maxPos < 1e-3 ? 'ok' : 'err';
  el.innerHTML =
    `<div>最大位置误差: <span class="${cls}">${c.maxPos.toExponential(3)}</span> (模型空间单位)</div>` +
    `<div>最大法线误差: <span class="${c.maxNrm < 1e-3 ? 'ok' : 'err'}">${c.maxNrm.toExponential(3)}</span></div>` +
    `<div>最差顶点: #${c.worstVertex}</div>` +
    `<div>容差: 位置 1e-3 · 顶点数 ${c.perVertex.length}</div>`;
}

// ---------- vertex inspection ----------

function vertexInfluences(v) {
  const { asset } = state;
  const out = [];
  for (let k = 0; k < MAX_INFLUENCES; k++) {
    const w = asset.mesh.weights[v * MAX_INFLUENCES + k];
    const j = asset.mesh.joints[v * MAX_INFLUENCES + k];
    if (w > 0) out.push({ joint: j, name: asset.jointNames[j], weight: w });
  }
  return out;
}

function updateVertexPanel() {
  const el = $('vertex-info');
  const v = state.selectedVertex;
  if (v === null || !state.asset) { el.textContent = '未选择顶点'; return; }
  const infl = vertexInfluences(v);
  const rows = infl.length
    ? infl.map((i) => `<tr><th>${i.name}</th><td>${i.weight.toFixed(6)}</td></tr>`).join('')
    : '<tr><th colspan="2">零影响顶点 —— 刚性跟随网格节点</th></tr>';

  let cpuRow = '';
  let gpuRow = '';
  if (state.lastCheck) {
    const { cpu, gpu } = state.lastCheck;
    const d = Math.hypot(
      cpu.positions[v * 3] - gpu.positions[v * 3],
      cpu.positions[v * 3 + 1] - gpu.positions[v * 3 + 1],
      cpu.positions[v * 3 + 2] - gpu.positions[v * 3 + 2],
    );
    cpuRow = `<tr><th>CPU 位置</th><td>${fmt3(cpu.positions, v * 3)}</td></tr>`;
    gpuRow = `<tr><th>GPU 位置</th><td>${fmt3(gpu.positions, v * 3)}</td></tr>` +
             `<tr><th>|Δ|</th><td>${d.toExponential(3)}</td></tr>`;
  }
  el.innerHTML =
    `<table><tr><th>顶点</th><td>#${v}</td></tr>${rows}` +
    `<tr><th>绑定位置</th><td>${fmt3(state.asset.mesh.positions, v * 3)}</td></tr>` +
    `${cpuRow}${gpuRow}</table>`;
}

function fmt3(arr, o) {
  return `${arr[o].toFixed(3)}, ${arr[o + 1].toFixed(3)}, ${arr[o + 2].toFixed(3)}`;
}

function pickVertex(clientX, clientY) {
  if (!state.asset || !state.frame) return;
  const rect = canvas.getBoundingClientRect();
  const px = (clientX - rect.left) / rect.width;
  const py = (clientY - rect.top) / rect.height;
  const f = state.frame;
  const cpu = state.lastCheck?.cpu ?? skinMeshCPU(state.asset, f.skin, f.normals);
  const proj = mat4.perspective(Math.PI / 4, canvas.width / canvas.height, 0.05, 100);
  const view = mat4.lookAt(state.camera.eye(), state.camera.target, [0, 1, 0]);
  const vp = mat4.multiply(proj, view);
  const n = state.asset.mesh.vertexCount;
  let best = -1;
  let bestD = 0.02; // ndc distance threshold
  for (let v = 0; v < n; v++) {
    const p = [cpu.positions[v * 3], cpu.positions[v * 3 + 1], cpu.positions[v * 3 + 2]];
    const wp = mat4.transformPoint(f.meshWorld, p);
    const cx = vp[0] * wp[0] + vp[4] * wp[1] + vp[8] * wp[2] + vp[12];
    const cy = vp[1] * wp[0] + vp[5] * wp[1] + vp[9] * wp[2] + vp[13];
    const cw = vp[3] * wp[0] + vp[7] * wp[1] + vp[11] * wp[2] + vp[15];
    if (cw <= 0) continue;
    const dx = cx / cw - (px * 2 - 1);
    const dy = cy / cw - (1 - py * 2);
    const d = Math.hypot(dx, dy);
    if (d < bestD) { bestD = d; best = v; }
  }
  if (best >= 0) {
    state.selectedVertex = best;
    $('input-vertex').value = best;
    updateVertexPanel();
  }
}

// ---------- asset loading ----------

function showValidation(warnings, error) {
  const el = $('validation-report');
  if (error) {
    el.innerHTML = `<span class="err">拒绝加载: ${escapeHtml(error.message)}</span>` +
      (error.problems || []).map((p) => `\n<span class="err">· ${escapeHtml(p)}</span>`).join('');
    return;
  }
  if (!warnings || warnings.length === 0) {
    el.innerHTML = '<span class="ok">校验通过，无警告</span>';
    return;
  }
  el.innerHTML = warnings.map((w) => `<span class="warn">· ${escapeHtml(w)}</span>`).join('\n');
}

function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

function installAsset(json, name) {
  const asset = loadAsset(json); // throws AssetError on rejection
  state.asset = asset;
  state.skeleton = buildSkeleton(asset);
  state.assetName = name;
  state.clipName = asset.animations.length ? asset.animations[0].name : null;
  state.time = 0;
  state.selectedVertex = null;
  state.lastCheck = null;

  state.skinner = new GpuSkinner(state.gl, asset);
  state.renderer = new Renderer(state.gl, asset, state.skinner);

  // aim the camera at the character's world-space center
  const f = evaluateFrame();
  const hips = f.world[asset.jointNodes[0]].slice(12, 15);
  state.camera.target = [hips[0], hips[1] + 0.4, hips[2]];
  state.camera.dist = 4.5;

  // UI: clips + joints
  $('sel-clip').innerHTML =
    '<option value="">绑定姿势</option>' +
    asset.animations.map((a) => `<option value="${a.name}">${a.name}</option>`).join('');
  $('sel-clip').value = state.clipName ?? '';
  $('sel-heatjoint').innerHTML =
    asset.jointNames.map((n, i) => `<option value="${i}">${n}</option>`).join('');
  $('input-vertex').max = asset.mesh.vertexCount - 1;

  $('asset-info').textContent =
    `${asset.name} · 顶点 ${asset.mesh.vertexCount} · 关节 ${asset.jointNodes.length}` +
    ` · 动画 ${asset.animations.length} · 零影响顶点 ${asset.mesh.zeroInfluence.length}`;
  showValidation(asset.warnings, null);
  renderCheckPanel();
  updateVertexPanel();
  syncTimeUI();
}

async function loadDefaultAsset() {
  const res = await fetch(DEFAULT_ASSET_URL);
  if (!res.ok) throw new Error(`fetch ${DEFAULT_ASSET_URL}: ${res.status}`);
  const json = await res.json();
  installAsset(json, 'runway_character.rskin.json');
}

// ---------- config save / load ----------

function serializeConfig() {
  return {
    type: 'rskin-inspector-config',
    version: 1,
    asset: state.assetName,
    clip: state.clipName,
    time: state.time,
    playing: state.playing,
    selectedVertex: state.selectedVertex,
    colorMode: state.colorMode,
    heatJoint: state.heatJoint,
    showSkeleton: state.showSkeleton,
    continuousCheck: state.continuousCheck,
    camera: state.camera.serialize(),
  };
}

function applyConfig(cfg) {
  if (cfg.type !== 'rskin-inspector-config') throw new Error('不是有效的检视配置');
  if (!state.asset) throw new Error('请先加载资产');
  if (cfg.asset && cfg.asset !== state.assetName) {
    showValidation([], null);
    $('validation-report').innerHTML =
      `<span class="warn">配置来自资产 ${escapeHtml(cfg.asset)}，当前为 ${escapeHtml(state.assetName)}；已尽量应用</span>`;
  }
  if (cfg.clip !== undefined) {
    state.clipName = cfg.clip && state.asset.animations.some((a) => a.name === cfg.clip) ? cfg.clip : null;
    $('sel-clip').value = state.clipName ?? '';
  }
  if (typeof cfg.time === 'number') state.time = cfg.time;
  if (typeof cfg.playing === 'boolean') state.playing = cfg.playing;
  if (cfg.selectedVertex !== undefined) {
    state.selectedVertex = (cfg.selectedVertex === null || cfg.selectedVertex < 0
      || cfg.selectedVertex >= state.asset.mesh.vertexCount) ? null : cfg.selectedVertex;
    $('input-vertex').value = state.selectedVertex ?? '';
  }
  if (cfg.colorMode !== undefined) { state.colorMode = cfg.colorMode; $('sel-colormode').value = String(cfg.colorMode); }
  if (cfg.heatJoint !== undefined) {
    state.heatJoint = Math.min(Math.max(0, cfg.heatJoint | 0), state.asset.jointNodes.length - 1);
    $('sel-heatjoint').value = String(state.heatJoint);
  }
  if (cfg.showSkeleton !== undefined) { state.showSkeleton = cfg.showSkeleton; $('chk-skeleton').checked = cfg.showSkeleton; }
  if (cfg.continuousCheck !== undefined) { state.continuousCheck = cfg.continuousCheck; $('chk-continuous').checked = cfg.continuousCheck; }
  if (cfg.camera) state.camera = OrbitCamera.deserialize(cfg.camera);
  state.lastCheck = null;
  renderCheckPanel();
  updateVertexPanel();
  syncTimeUI();
}

// ---------- UI wiring ----------

function syncTimeUI() {
  const clip = state.clipName ? state.asset?.animations.find((a) => a.name === state.clipName) : null;
  const dur = clip ? clipDuration(clip) : 0;
  $('slider-time').max = dur;
  $('slider-time').value = state.time;
  $('slider-time').disabled = !clip;
  $('label-time').textContent = `${state.time.toFixed(3)} / ${dur.toFixed(3)}`;
  $('btn-play').textContent = state.playing ? '暂停' : '播放';
}

function wireUI() {
  $('btn-load-default').onclick = () => {
    loadDefaultAsset().catch((e) => showValidation([], e));
  };
  $('file-asset').onchange = async (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    try {
      const json = JSON.parse(await file.text());
      installAsset(json, file.name);
    } catch (e) {
      showValidation([], e);
    }
  };
  $('sel-clip').onchange = (ev) => {
    state.clipName = ev.target.value || null;
    state.time = 0;
    syncTimeUI();
  };
  $('btn-play').onclick = () => { state.playing = !state.playing; syncTimeUI(); };
  $('slider-time').oninput = (ev) => {
    state.time = parseFloat(ev.target.value);
    state.playing = false;
    syncTimeUI();
  };
  $('sel-colormode').onchange = (ev) => { state.colorMode = parseInt(ev.target.value, 10); };
  $('sel-heatjoint').onchange = (ev) => { state.heatJoint = parseInt(ev.target.value, 10); };
  $('chk-skeleton').onchange = (ev) => { state.showSkeleton = ev.target.checked; };
  $('chk-continuous').onchange = (ev) => { state.continuousCheck = ev.target.checked; };
  $('btn-goto-vertex').onclick = () => {
    if (!state.asset) return;
    const v = parseInt($('input-vertex').value, 10);
    if (Number.isInteger(v) && v >= 0 && v < state.asset.mesh.vertexCount) {
      state.selectedVertex = v;
      updateVertexPanel();
    }
  };
  $('btn-check').onclick = () => {
    if (runCheck(true)) updateVertexPanel();
  };
  $('btn-save-config').onclick = async () => {
    const text = JSON.stringify(serializeConfig(), null, 2);
    $('config-text').value = text;
    const blob = new Blob([text], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'inspector-config.json';
    a.click();
    URL.revokeObjectURL(a.href);
  };
  $('btn-load-config').onclick = () => {
    try {
      applyConfig(JSON.parse($('config-text').value));
    } catch (e) {
      $('validation-report').innerHTML = `<span class="err">配置导入失败: ${escapeHtml(e.message)}</span>`;
    }
  };

  // camera + picking
  let dragging = false;
  let moved = 0;
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    moved = 0;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    moved += Math.abs(e.movementX) + Math.abs(e.movementY);
    state.camera.rotate(e.movementX, e.movementY);
  });
  canvas.addEventListener('pointerup', (e) => {
    dragging = false;
    if (moved < 5) pickVertex(e.clientX, e.clientY);
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    state.camera.zoom(e.deltaY);
  }, { passive: false });
}

// ---------- boot ----------

function boot() {
  try {
    state.gl = createContext(canvas);
    checkCapabilities(state.gl);
  } catch (e) {
    $('validation-report').innerHTML = `<span class="err">WebGL 初始化失败: ${escapeHtml(e.message)}</span>`;
    return;
  }
  wireUI();
  loadDefaultAsset().catch((e) => showValidation([], e));
  requestAnimationFrame(frameLoop);
}

boot();
