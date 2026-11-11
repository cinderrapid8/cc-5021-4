// Distributable asset format "rskin" (runway skin), version 1.
// A single JSON document carrying skeleton, skinned mesh and animation clips.
// All bulk numeric data is base64-encoded little-endian typed arrays so the
// file is engine-independent and exact (no float text round-trips).

export const FORMAT_ID = 'rskin';
export const FORMAT_VERSION = 1;

// Maximum joint influences stored per vertex.
export const MAX_INFLUENCES = 4;

// ---- base64 <-> typed array helpers (work in Node and browsers) ----

export function bytesToBase64(bytes) {
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64');
  }
  let bin = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

export function base64ToBytes(b64) {
  if (typeof Buffer !== 'undefined') {
    return new Uint8Array(Buffer.from(b64, 'base64'));
  }
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export function f32ToBase64(arr) {
  return bytesToBase64(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
}

export function base64ToF32(b64) {
  const bytes = base64ToBytes(b64);
  if (bytes.byteLength % 4 !== 0) {
    throw new Error(`f32 payload byte length ${bytes.byteLength} is not a multiple of 4`);
  }
  // Copy to guarantee alignment.
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

export function u16ToBase64(arr) {
  return bytesToBase64(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
}

export function base64ToU16(b64) {
  const bytes = base64ToBytes(b64);
  if (bytes.byteLength % 2 !== 0) {
    throw new Error(`u16 payload byte length ${bytes.byteLength} is not a multiple of 2`);
  }
  return new Uint16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

export function u32ToBase64(arr) {
  return bytesToBase64(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
}

export function base64ToU32(b64) {
  const bytes = base64ToBytes(b64);
  if (bytes.byteLength % 4 !== 0) {
    throw new Error(`u32 payload byte length ${bytes.byteLength} is not a multiple of 4`);
  }
  return new Uint32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}
