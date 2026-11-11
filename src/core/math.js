// Column-major mat4 (WebGL layout): m[0..3] is column 0.
// CPU side uses plain number[] (float64) for reference precision.

export const vec3 = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ],
  length: (a) => Math.hypot(a[0], a[1], a[2]),
  normalize(a) {
    const l = Math.hypot(a[0], a[1], a[2]);
    if (l === 0) return [0, 0, 0];
    return [a[0] / l, a[1] / l, a[2] / l];
  },
};

export const mat4 = {
  identity: () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],

  // a * b (b applied first to column vectors)
  multiply(a, b) {
    const o = new Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        o[c * 4 + r] =
          a[r] * b[c * 4] +
          a[4 + r] * b[c * 4 + 1] +
          a[8 + r] * b[c * 4 + 2] +
          a[12 + r] * b[c * 4 + 3];
      }
    }
    return o;
  },

  translation: (x, y, z) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1],

  scaling: (x, y, z) => [x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1],

  fromQuat(q) {
    const [x, y, z, w] = q;
    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, xy = x * y2, xz = x * z2;
    const yy = y * y2, yz = y * z2, zz = z * z2;
    const wx = w * x2, wy = w * y2, wz = w * z2;
    return [
      1 - (yy + zz), xy + wz, xz - wy, 0,
      xy - wz, 1 - (xx + zz), yz + wx, 0,
      xz + wy, yz - wx, 1 - (xx + yy), 0,
      0, 0, 0, 1,
    ];
  },

  // T * R * S
  compose(t, q, s) {
    const r = mat4.fromQuat(q);
    const o = new Array(16);
    for (let c = 0; c < 3; c++) {
      o[c * 4] = r[c * 4] * s[0];
      o[c * 4 + 1] = r[c * 4 + 1] * s[1];
      o[c * 4 + 2] = r[c * 4 + 2] * s[2];
      o[c * 4 + 3] = 0;
    }
    o[12] = t[0]; o[13] = t[1]; o[14] = t[2]; o[15] = 1;
    return o;
  },

  transpose(m) {
    return [
      m[0], m[4], m[8], m[12],
      m[1], m[5], m[9], m[13],
      m[2], m[6], m[10], m[14],
      m[3], m[7], m[11], m[15],
    ];
  },

  // General 4x4 inverse (Gaussian elimination with partial pivoting).
  // Returns null when singular.
  invert(m) {
    const a = [];
    for (let i = 0; i < 4; i++) {
      a.push([m[i], m[4 + i], m[8 + i], m[12 + i], i === 0 ? 1 : 0, i === 1 ? 1 : 0, i === 2 ? 1 : 0, i === 3 ? 1 : 0]);
    }
    for (let col = 0; col < 4; col++) {
      let piv = col;
      for (let r = col + 1; r < 4; r++) {
        if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
      }
      if (Math.abs(a[piv][col]) < 1e-15) return null;
      [a[col], a[piv]] = [a[piv], a[col]];
      const d = a[col][col];
      for (let k = 0; k < 8; k++) a[col][k] /= d;
      for (let r = 0; r < 4; r++) {
        if (r === col) continue;
        const f = a[r][col];
        for (let k = 0; k < 8; k++) a[r][k] -= f * a[col][k];
      }
    }
    const o = new Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) o[c * 4 + r] = a[r][4 + c];
    }
    return o;
  },

  determinant3(m) {
    // upper 3x3 determinant
    return (
      m[0] * (m[5] * m[10] - m[6] * m[9]) -
      m[4] * (m[1] * m[10] - m[2] * m[9]) +
      m[8] * (m[1] * m[6] - m[2] * m[5])
    );
  },

  // Point transform (w=1), returns [x,y,z]
  transformPoint(m, p) {
    const [x, y, z] = p;
    return [
      m[0] * x + m[4] * y + m[8] * z + m[12],
      m[1] * x + m[5] * y + m[9] * z + m[13],
      m[2] * x + m[6] * y + m[10] * z + m[14],
    ];
  },

  // Direction transform (w=0)
  transformDir(m, p) {
    const [x, y, z] = p;
    return [
      m[0] * x + m[4] * y + m[8] * z,
      m[1] * x + m[5] * y + m[9] * z,
      m[2] * x + m[6] * y + m[10] * z,
    ];
  },

  // 3x3 normal matrix (inverse-transpose of upper 3x3), row-major 9 elements.
  // Exact for any invertible affine matrix, including non-uniform scale + shear.
  normalMatrix(m) {
    const inv = mat4.invert(m);
    if (!inv) return null;
    const t = mat4.transpose(inv);
    return [t[0], t[1], t[2], t[4], t[5], t[6], t[8], t[9], t[10]];
  },

  almostEqual(a, b, eps = 1e-9) {
    for (let i = 0; i < 16; i++) {
      if (Math.abs(a[i] - b[i]) > eps) return false;
    }
    return true;
  },

  perspective(fovY, aspect, near, far) {
    const f = 1 / Math.tan(fovY / 2);
    const nf = 1 / (near - far);
    return [
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, 2 * far * near * nf, 0,
    ];
  },

  lookAt(eye, center, up) {
    const z = vec3.normalize(vec3.sub(eye, center));
    const x = vec3.normalize(vec3.cross(up, z));
    const y = vec3.cross(z, x);
    return [
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -vec3.dot(x, eye), -vec3.dot(y, eye), -vec3.dot(z, eye), 1,
    ];
  },
};

export const quat = {
  identity: () => [0, 0, 0, 1],

  fromAxisAngle(axis, angle) {
    const n = vec3.normalize(axis);
    const h = angle / 2;
    const s = Math.sin(h);
    return [n[0] * s, n[1] * s, n[2] * s, Math.cos(h)];
  },

  normalize(q) {
    const l = Math.hypot(q[0], q[1], q[2], q[3]);
    if (l === 0) return [0, 0, 0, 1];
    return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
  },

  multiply(a, b) {
    const [ax, ay, az, aw] = a;
    const [bx, by, bz, bw] = b;
    return [
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz,
    ];
  },

  // Shortest-path normalized lerp.
  slerp(a, b, t) {
    let [bx, by, bz, bw] = b;
    let cos = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw;
    if (cos < 0) {
      cos = -cos;
      bx = -bx; by = -by; bz = -bz; bw = -bw;
    }
    if (cos > 0.9995) {
      return quat.normalize([
        a[0] + (bx - a[0]) * t,
        a[1] + (by - a[1]) * t,
        a[2] + (bz - a[2]) * t,
        a[3] + (bw - a[3]) * t,
      ]);
    }
    const th = Math.acos(cos);
    const sin = Math.sin(th);
    const wa = Math.sin((1 - t) * th) / sin;
    const wb = Math.sin(t * th) / sin;
    return [
      wa * a[0] + wb * bx,
      wa * a[1] + wb * by,
      wa * a[2] + wb * bz,
      wa * a[3] + wb * bw,
    ];
  },

  // Rotate a vector by quaternion.
  rotateVec3(q, v) {
    const [x, y, z, w] = q;
    const uvx = y * v[2] - z * v[1];
    const uvy = z * v[0] - x * v[2];
    const uvz = x * v[1] - y * v[0];
    const uuvx = y * uvz - z * uvy;
    const uuvy = z * uvx - x * uvz;
    const uuvz = x * uvy - y * uvx;
    return [
      v[0] + 2 * (w * uvx + uuvx),
      v[1] + 2 * (w * uvy + uuvy),
      v[2] + 2 * (w * uvz + uuvz),
    ];
  },
};

export function isFiniteArray(arr) {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) return false;
  }
  return true;
}
