// Minimal WebGL1 context helpers shared by the app and the Node GPU test.

export function createContext(canvas, width, height) {
  const attrs = {
    alpha: false,
    depth: true,
    stencil: false,
    antialias: true,
    preserveDrawingBuffer: true, // needed for reliable readback
  };
  const gl = canvas.getContext('webgl', attrs)
    || canvas.getContext('experimental-webgl', attrs);
  if (!gl) throw new Error('WebGL1 is not available in this environment');
  if (width !== undefined) {
    canvas.width = width;
    canvas.height = height;
  }
  return gl;
}

export function checkCapabilities(gl) {
  const floatTex = gl.getExtension('OES_texture_float');
  if (!floatTex) throw new Error('OES_texture_float is required for bone textures');
  const vtUnits = gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS);
  if (!vtUnits) throw new Error('vertex texture fetch is required for bone textures');
  // needed only when a mesh uses 32-bit indices (ours uses 16-bit)
  gl.getExtension('OES_element_index_uint');
  return { floatTex, vertexTextureUnits: vtUnits };
}

export function compileShader(gl, type, source) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, source);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`shader compile failed: ${log}\n--- source ---\n${source}`);
  }
  return sh;
}

export function linkProgram(gl, vsSource, fsSource) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`program link failed: ${log}`);
  }
  return prog;
}

// Collects all attribute/uniform locations up front.
export function programInfo(gl, prog, attribs, uniforms) {
  const info = { prog, attribs: {}, uniforms: {} };
  for (const name of attribs) {
    info.attribs[name] = gl.getAttribLocation(prog, name);
  }
  for (const name of uniforms) {
    info.uniforms[name] = gl.getUniformLocation(prog, name);
  }
  return info;
}

export function makeFloatTexture(gl, width, height) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.FLOAT, null);
  return tex;
}

export function uploadFloatTexture(gl, tex, width, height, data) {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.FLOAT, data);
}
