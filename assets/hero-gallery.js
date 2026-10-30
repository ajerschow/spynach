// Hero gallery of the landing page: a slidable pair of figures, the live "spin vortex" animation first and the
// Spynoza picture second. Without JavaScript, without WebGL, or with prefers-reduced-motion the page keeps its
// static figure, the markup of index.html, and no gallery is built.
//
// The animation: two logarithmic spirals of opposite handedness (phase fields exp(i(m*theta + k*ln r - w t))) and
// one Archimedean spiral interfere, drawn as a halftone dot screen. Dot size is |signal|, dot colour the signal
// phase. Pointer: the vortex centre drifts toward it. Click: a "pi pulse" reverses one spiral family smoothly.
// It draws only while its slide and the page are visible.
(() => {
  const figure = document.querySelector('.hero figure');
  const img = figure && figure.querySelector('img');
  const caption = figure && figure.querySelector('figcaption');
  if (!img || !caption) return;

  // The animation needs WebGL and a visitor who has not asked for reduced motion. Otherwise its slide shows a
  // still frame of it, assets/spin-vortex-frame.png, so the gallery keeps both figures.
  const canvas = document.createElement('canvas');
  const gl = matchMedia('(prefers-reduced-motion: reduce)').matches ? null : canvas.getContext('webgl', { antialias: false });
  let program = null;

  const vs = 'attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}';
  const fs = `
precision highp float;
uniform vec2 res; uniform float t; uniform float dpr;
uniform vec2 ctr;     // vortex centre, 0..1
uniform float hand;   // handedness of the second spiral family, -1..1

vec3 wheel(float x){                       // phase colour wheel, x in 0..1
  vec3 a = vec3(.42, .88, .76), b = vec3(.80, .26, .60), d = vec3(1., .66, .28);
  x = fract(x) * 3.;
  return x < 1. ? mix(a, b, x) : x < 2. ? mix(b, d, x - 1.) : mix(d, a, x - 2.);
}

void main(){
  vec2 fc = gl_FragCoord.xy;
  vec3 bg = vec3(.035, .04, .05);
  vec3 col = bg;

  // halftone field
  float cell = 7. * dpr;
  vec2 id = floor(fc / cell);
  vec2 cp = (id + .5) * cell;
  vec2 lp = fc - cp;
  vec2 p = (cp - ctr * res) / res.y;
  float r = length(p) + 1e-3, a = atan(p.y, p.x), L = log(r);

  float ph1 = 5. * a + 13. * L - 2.0 * t;            // log spiral -> endless zoom
  float ph2 = 7. * hand * a + 10. * L - 1.4 * t;     // second family, flips on click
  float ph3 = 2. * a + 34. * r - 1.1 * t;            // Archimedean spiral
  vec2 z = vec2(cos(ph1), sin(ph1)) + vec2(cos(ph2), sin(ph2)) + .6 * vec2(cos(ph3), sin(ph3));
  z /= 2.6;
  float I = dot(z, z);

  vec2 uv = cp / res;
  float vig = smoothstep(.015, .09, r)
            * smoothstep(0., .07, uv.x) * smoothstep(0., .07, 1. - uv.x)
            * smoothstep(0., .12, uv.y) * smoothstep(0., .14, 1. - uv.y);
  float rad = cell * .5 * sqrt(I) * vig;
  float dotA = 1. - smoothstep(rad - .9 * dpr, rad + .3 * dpr, length(lp));
  vec3 dc = wheel(atan(z.y, z.x) / 6.2832 + .5) * (.55 + .45 * sqrt(I));
  col = mix(col, dc, dotA);

  // top trace: an echo-shaped FID
  vec2 q = fc / res;
  float x = q.x - .5;
  float fid = cos(70. * x - 3. * t) * exp(-abs(x) * 7.) * (.8 + .2 * sin(t * .7));
  float yT = (.935 + .035 * fid) * res.y;
  float lt = 1. - smoothstep(.4 * dpr, 1.4 * dpr, abs(fc.y - yT));
  col = mix(col, vec3(.85), lt * .6 * smoothstep(.02, .1, q.x) * smoothstep(.02, .1, 1. - q.x));

  // right trace: a drifting spectrum of Lorentzian lines
  float s = 0.;
  for (int i = 0; i < 5; i++) {
    float fi = float(i);
    float cpos = .2 + .15 * fi + .02 * sin(t * (.3 + .1 * fi) + fi);
    float w = .008 + .004 * fi;
    float d = (q.y - cpos) / w;
    s += (.5 + .5 * sin(fi * 2.3 + t * .4)) / (1. + d * d);
  }
  float xR = (.955 - .035 * s) * res.x;
  float lr = 1. - smoothstep(.4 * dpr, 1.4 * dpr, abs(fc.x - xR));
  col = mix(col, vec3(.85), lr * .6 * smoothstep(.05, .14, q.y) * smoothstep(.1, .2, 1. - q.y));

  gl_FragColor = vec4(col, 1.);
}`;
  const compile = () => {
  const shader = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const prog = gl.createProgram();
  gl.attachShader(prog, shader(gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return false;
  program = prog;
  gl.useProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(program, 'p');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  return true;
  };
  const animated = Boolean(gl) && compile();

  // ---- the gallery: the animation first, the existing picture second ----
  const el = (tag, cls, attrs = {}) => Object.assign(document.createElement(tag), { className: cls }, attrs);
  const track = el('div', 'gallery-track', { tabIndex: 0 });
  track.setAttribute('role', 'group');
  track.setAttribute('aria-roledescription', 'carousel');
  track.setAttribute('aria-label', 'Figures: animation and picture');

  const animSlide = el('div', 'gallery-slide');
  const alt = 'Interference of rotating spin-phase spirals, drawn as coloured dots: dot size is the signal amplitude and colour its phase';
  let first;
  if (animated) {
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', alt);
    canvas.style.aspectRatio = '1397 / 1202';
    first = canvas;
  } else {
    first = el('img', 'gallery-still', { src: 'assets/spin-vortex-frame.png', alt, width: 960, height: 826 });
  }
  animSlide.append(first, el('div', 'gallery-caption', {
    textContent: animated
      ? 'Click for π pulse'
      : 'A frame of an animation of interfering spin-phase spirals; it moves in browsers with WebGL.',
  }));

  const imgSlide = el('div', 'gallery-slide');
  caption.className = 'gallery-caption';
  const stage = el('div', 'ripple-stage');
  stage.append(img);
  imgSlide.append(stage, caption);

  // ---- the picture: a click is an RF pulse. The excitation spreads over the picture and every point then rings down
  // like a free induction decay, a cosine under an exponential envelope, which refracts the image with a little
  // colour fringing; from the click point the receiver trace is drawn out to the right as it is recorded. Up to four
  // clicks at once. The WebGL canvas is built at the first click and sits over the image only while an FID is alive;
  // it is skipped for reduced motion, without WebGL, or when the image cannot be read as a texture. ----
  const RIPPLE = `
precision highp float;
uniform sampler2D tex; uniform vec2 res;
uniform vec3 clk[4];  // click x, y (0..1, y up) and age in s; age < 0 = unused
void main(){
  vec2 uv = gl_FragCoord.xy / res;
  float asp = res.x / res.y;
  vec2 shift = vec2(0.); float glint = 0.; float trace = 0.;
  for (int k = 0; k < 4; k++) {
    vec3 c = clk[k];
    if (c.z < 0.) continue;
    vec2 dv = (uv - c.xy) * vec2(asp, 1.);
    float d = length(dv) + 1e-4;
    float tau = c.z - d / .55;                              // time since the excitation reached this point
    if (tau > 0.) {
      float fid = cos(tau * 24.) * exp(-tau / .8) * smoothstep(0., .1, tau) * exp(-c.z / 3.);   // T2 decay
      shift += normalize(dv) / vec2(asp, 1.) * fid * .010;
      glint += fid * .08;
    }
    float u = dv.x / .55;                                   // the recorded trace, one time unit per width of .55
    if (dv.x > 0. && u < c.z) {
      float y = c.y + .07 * cos(u * 24.) * exp(-u / .8) * exp(-c.z / 3.) * min(1., c.z * 4.);
      trace = max(trace, (1. - smoothstep(.002, .006, abs(uv.y - y))) * exp(-c.z / 2.5));
    }
  }
  vec3 col;
  col.r = texture2D(tex, uv + shift * 1.35).r;
  col.g = texture2D(tex, uv + shift).g;
  col.b = texture2D(tex, uv + shift * .65).b;
  col += glint;
  col = mix(col, vec3(.55, .95, .85), clamp(trace, 0., 1.) * .9);
  gl_FragColor = vec4(col, 1.);
}`;
  const ripples = () => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    img.style.cursor = 'pointer';
    const clicks = [];
    let cv = null, g = null, uRes, uClk, loop = 0, dead = false;
    const build = () => {
      cv = el('canvas', 'ripple-canvas');
      cv.setAttribute('aria-hidden', 'true');
      g = cv.getContext('webgl', { antialias: false });
      if (!g) return false;
      const sh = (type, src) => { const x = g.createShader(type); g.shaderSource(x, src); g.compileShader(x); return x; };
      const pr = g.createProgram();
      g.attachShader(pr, sh(g.VERTEX_SHADER, vs));
      g.attachShader(pr, sh(g.FRAGMENT_SHADER, RIPPLE));
      g.linkProgram(pr);
      if (!g.getProgramParameter(pr, g.LINK_STATUS)) return false;
      g.useProgram(pr);
      g.bindBuffer(g.ARRAY_BUFFER, g.createBuffer());
      g.bufferData(g.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), g.STATIC_DRAW);
      const at = g.getAttribLocation(pr, 'p');
      g.enableVertexAttribArray(at);
      g.vertexAttribPointer(at, 2, g.FLOAT, false, 0, 0);
      g.bindTexture(g.TEXTURE_2D, g.createTexture());
      g.pixelStorei(g.UNPACK_FLIP_Y_WEBGL, true);
      for (const [k, v] of [[g.TEXTURE_MIN_FILTER, g.LINEAR], [g.TEXTURE_MAG_FILTER, g.LINEAR],
        [g.TEXTURE_WRAP_S, g.CLAMP_TO_EDGE], [g.TEXTURE_WRAP_T, g.CLAMP_TO_EDGE]]) g.texParameteri(g.TEXTURE_2D, k, v);
      g.texImage2D(g.TEXTURE_2D, 0, g.RGB, g.RGB, g.UNSIGNED_BYTE, img);   // throws if the image is cross-origin
      uRes = g.getUniformLocation(pr, 'res');
      uClk = g.getUniformLocation(pr, 'clk');
      stage.append(cv);
      return true;
    };
    const ck = new Float32Array(12);
    const paint = now => {
      loop = 0;
      while (clicks.length && now - clicks[0].t > 5500) clicks.shift();
      if (!clicks.length) { cv.style.visibility = 'hidden'; return; }
      const ratio = Math.min(devicePixelRatio || 1, 2);
      const w = Math.round(cv.clientWidth * ratio), h = Math.round(cv.clientHeight * ratio);
      if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; g.viewport(0, 0, w, h); }
      g.uniform2f(uRes, cv.width, cv.height);
      for (let k = 0; k < 4; k++) ck.set(clicks[k] ? [clicks[k].x, clicks[k].y, (now - clicks[k].t) / 1000] : [0, 0, -1], 3 * k);
      g.uniform3fv(uClk, ck);
      g.drawArrays(g.TRIANGLES, 0, 3);
      loop = requestAnimationFrame(paint);
    };
    stage.addEventListener('click', e => {      // click, not pointerdown: a swipe must not start a ripple
      if (dead || !img.complete || !img.naturalWidth) return;
      if (!cv) {
        try { dead = !build(); } catch (err) { dead = true; }
        if (dead) { img.style.cursor = ''; return; }
      }
      const r = img.getBoundingClientRect();
      clicks.push({ x: (e.clientX - r.left) / r.width, y: (r.bottom - e.clientY) / r.height, t: performance.now() });
      if (clicks.length > 4) clicks.shift();
      cv.style.visibility = 'visible';
      if (!loop) loop = requestAnimationFrame(paint);
    });
  };
  ripples();

  // ---- two more animations, each confined to its own slide on the same dark background ----
  const BG = 'vec3 bg = vec3(.035, .04, .05);';
  const KSPACE = `
precision highp float;
uniform vec2 res; uniform float t; uniform float dpr;
uniform vec2 mouse; uniform float mAmp;
uniform vec3 clk[4];  // click x, y (0..1) and age in s; age < 0 = unused
void main(){
  float cell = 10. * dpr;
  vec2 fc = gl_FragCoord.xy;
  vec2 id = floor(fc / cell);
  vec2 cp = (id + .5) * cell;            // sample at the dot centre
  vec2 lp = fc - cp;
  float asp = res.x / res.y;
  vec2 r = cp / res.y;                   // position in units of the height

  // sum of plane waves: k-vectors on slowly rotating spokes
  vec2 z = vec2(0.);
  for (int k = 0; k < 4; k++) {
    float fk = float(k);
    float dir = mod(fk, 2.) < .5 ? 1. : -1.;
    float th = dir * t * .045 + fk * 1.9;
    float km = 13. + 5. * fk;
    float ph = km * dot(r, vec2(cos(th), sin(th))) - t * (.5 + .15 * fk);
    z += .5 * vec2(cos(ph), sin(ph));
  }
  // pointer: an extra plane wave, k from the centre to the pointer
  vec2 kv = (mouse - .5) * vec2(asp, 1.) * 70.;
  float phm = dot(kv, r) - t * 1.2;
  z += mAmp * .8 * vec2(cos(phm), sin(phm));
  // clicks: outgoing circular waves with a soft front, decaying
  for (int k = 0; k < 4; k++) {
    vec3 ck = clk[k];
    float age = ck.z;
    if (age < 0. || age > 8.) continue;
    float d = length(r - ck.xy * vec2(asp, 1.));
    float front = 1. - smoothstep(age * .32 - .04, age * .32, d);
    float env = exp(-age / 2.) * front / (1. + 6. * d);
    float ph = 45. * d - 9. * age;
    z += 2.2 * env * vec2(cos(ph), sin(ph));
  }
  float I = clamp(dot(z, z) / 4., 0., 1.);
  float rad = cell * .46 * sqrt(I);
  float dotA = 1. - smoothstep(rad - 1. * dpr, rad + .2 * dpr, length(lp));
  vec3 col = z.x > 0. ? vec3(130., 230., 205.) / 255. : vec3(255., 150., 120.) / 255.;
  ${BG}
  gl_FragColor = vec4(mix(bg, col, dotA * .95), 1.);
}`;
  const COSY = `
#extension GL_OES_standard_derivatives : enable
precision highp float;
uniform vec2 res; uniform float t; uniform float dpr;
uniform vec2 mouse; uniform float mAmp;
uniform vec3 clk[4];  // click x, y (0..1) and age in s; age < 0 = unused
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
// 2D Lorentzian (the product form gives the familiar star-shaped contours)
float lor(vec2 q, vec2 c, float w, float asp){
  vec2 d = (q - c) * vec2(asp, 1.) / w;
  return 1. / ((1. + d.x*d.x) * (1. + d.y*d.y));
}
float shift(float i){ return .12 + .19*i + .025*sin(t*.11*(i + 1.) + 1.7*i); }
void main(){
  vec2 q = gl_FragCoord.xy / res;
  float asp = res.x / res.y;
  float w = .018;
  float F = 0.;
  for (int ii = 0; ii < 5; ii++) {
    float i = float(ii);
    float di = shift(i);
    F += lor(q, vec2(di), w, asp);                          // diagonal peak
    for (int jj = 0; jj < 5; jj++) {
      float j = float(jj);
      if (jj <= ii) continue;
      float h = hash(vec2(i, j));
      if (h < .45) continue;                                // only some spins are coupled
      float dj = shift(j);
      float a = .8 * sin(t*(.15 + .2*h) + 6.28*h);          // amplitude and sign oscillate
      F += a * (lor(q, vec2(di, dj), w, asp) + lor(q, vec2(dj, di), w, asp));
    }
  }
  F += mAmp * (lor(q, mouse, w, asp) + lor(q, mouse.yx, w, asp));
  // clicks: pulse, excitation, coherence transfer, relaxation
  for (int k = 0; k < 4; k++) {
    vec3 ck = clk[k];
    float age = ck.z;
    if (age < 0. || age > 9.) continue;
    float env = exp(-age / 2.2);                            // T2 decay
    float osc = cos(age * 3.);                              // evolution: sign flips
    F += 1.2 * env * (lor(q, ck.xy, w, asp) + lor(q, ck.yx, w, asp));
    for (int ii = 0; ii < 5; ii++) {
      float i = float(ii);
      float di = shift(i);
      float on = smoothstep(.18*i, .18*i + .35, age);       // transfers arrive in turn
      F += .9 * env * osc * on * (lor(q, vec2(ck.x, di), w, asp) + lor(q, vec2(di, ck.x), w, asp));
    }
    vec2 dd = (q - ck.xy) * vec2(asp, 1.);
    float rr = (length(dd) - age * .38) / .012;             // expanding RF ripple
    F += .5 * exp(-age * 1.1) * exp(-rr * rr);
  }
  // logarithmic contour levels, drawn with screen-space width
  float thr = .04;
  float A = abs(F);
  float v = log2(max(A, 1e-6) / thr) * 1.6;
  float g = abs(fract(v - .5) - .5) / max(fwidth(v), 1e-4);
  float line = (1. - min(g, 1.)) * step(thr, A);
  vec3 col = F > 0. ? vec3(130., 230., 205.) / 255. : vec3(255., 150., 120.) / 255.;
  ${BG}
  gl_FragColor = vec4(mix(bg, col, line * .85), 1.);
}`;

  // A slide with its own WebGL canvas, or null if this browser cannot run the shader. The pointer steers a
  // plane wave or a cross peak, a click starts a wave or an RF pulse; it draws only while visible.
  const shaderSlide = (source, extensions, text) => {
    const cv = el('canvas', '');
    const g = cv.getContext('webgl', { antialias: false });
    if (!g || !extensions.every(x => g.getExtension(x))) return null;
    const sh = (type, src) => { const s = g.createShader(type); g.shaderSource(s, src); g.compileShader(s); return s; };
    const pr = g.createProgram();
    g.attachShader(pr, sh(g.VERTEX_SHADER, vs));
    g.attachShader(pr, sh(g.FRAGMENT_SHADER, source));
    g.linkProgram(pr);
    if (!g.getProgramParameter(pr, g.LINK_STATUS)) return null;
    g.useProgram(pr);
    g.bindBuffer(g.ARRAY_BUFFER, g.createBuffer());
    g.bufferData(g.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), g.STATIC_DRAW);
    const at = g.getAttribLocation(pr, 'p');
    g.enableVertexAttribArray(at);
    g.vertexAttribPointer(at, 2, g.FLOAT, false, 0, 0);
    const u = n => g.getUniformLocation(pr, n);
    const [uRes, uT, uDpr, uMouse, uAmp, uClk] = ['res', 't', 'dpr', 'mouse', 'mAmp', 'clk'].map(u);
    cv.setAttribute('role', 'img');
    cv.setAttribute('aria-label', text);
    cv.style.aspectRatio = '1397 / 1202';
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const fit = () => { cv.width = cv.clientWidth * ratio; cv.height = cv.clientHeight * ratio; g.viewport(0, 0, cv.width, cv.height); };
    new ResizeObserver(fit).observe(cv);
    fit();

    const clicks = [];
    let mouse = [.5, .5], aim = [.5, .5], amp = 0, ampAim = 0;
    const local = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (r.bottom - e.clientY) / r.height]; };
    cv.addEventListener('pointermove', e => { aim = local(e); if (ampAim === 0) mouse = aim.slice(); ampAim = .9; });
    cv.addEventListener('pointerleave', () => { ampAim = 0; });
    cv.addEventListener('click', e => {
      const [x, y] = local(e);
      clicks.push({ x, y, t: performance.now() });
      if (clicks.length > 4) clicks.shift();
    });

    let shown = true, loop = 0;
    const begin = performance.now();
    const ck = new Float32Array(12);
    const paint = now => {
      loop = 0;
      if (!shown || document.hidden) return;
      mouse = mouse.map((m, i) => m + (aim[i] - m) * .05);
      amp += (ampAim - amp) * .03;
      g.uniform2f(uRes, cv.width, cv.height);
      g.uniform1f(uT, (now - begin) / 1000);
      g.uniform1f(uDpr, ratio);
      g.uniform2f(uMouse, mouse[0], mouse[1]);
      g.uniform1f(uAmp, amp);
      for (let k = 0; k < 4; k++) ck.set(clicks[k] ? [clicks[k].x, clicks[k].y, (now - clicks[k].t) / 1000] : [0, 0, -1], 3 * k);
      g.uniform3fv(uClk, ck);
      g.drawArrays(g.TRIANGLES, 0, 3);
      loop = requestAnimationFrame(paint);
    };
    new IntersectionObserver(([entry]) => {
      shown = entry.isIntersecting;
      if (shown && !loop) loop = requestAnimationFrame(paint);
    }, { threshold: 0.1 }).observe(cv);
    document.addEventListener('visibilitychange', () => { if (!document.hidden && !loop) loop = requestAnimationFrame(paint); });
    loop = requestAnimationFrame(paint);

    const slide = el('div', 'gallery-slide');
    slide.append(cv, el('div', 'gallery-caption', { textContent: text }));
    return slide;
  };
  const extra = animated ? [
    ['k-space', shaderSlide(KSPACE, [], 'Plane waves in k-space. Move the pointer; click for a wave')],
    ['COSY', shaderSlide(COSY, ['OES_standard_derivatives'], 'A drifting COSY-like spectrum. Move the pointer; click for an RF pulse')],
  ].filter(([, slide]) => slide) : [];

  // ---- the relaxation chord diagram: a static SVG (scripts/make_hero_chord.py), inlined so that a click can send a
  // pulse through it. Without the inlining (the fetch fails) the image stays and is a correct static figure. ----
  const chordSlide = () => {
    const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const alt = 'Chord diagram of the cross-relaxation pathways in the zero-quantum space of 11BF4-: 13 operators on a circle, ribbons as wide as the rates';
    const slide = el('div', 'gallery-slide');
    const stage = el('div', 'chord-stage');
    const still = el('img', 'chord', { src: 'assets/relaxation-chord.svg', alt, width: 900, height: 900 });
    stage.append(still);
    const note = el('div', 'gallery-caption');
    const hint = el('span', '', { textContent: 'Ribbon width is the rate.' });
    const link = el('a', '', { href: 'try.html#ex-pathways', textContent: 'Try it in the browser' });
    note.append('Cross-relaxation pathways of 11BF4-, contributed by Florin Teleanu. ', hint, ' ', link, '.');
    slide.append(stage, note);

    const wire = svg => {
      const ribbons = [...svg.querySelectorAll('.ribbon')].map(path => ({
        path, a: +path.dataset.source, b: +path.dataset.target, rate: Math.abs(+path.dataset.rate),
      }));
      const arcs = [...svg.querySelectorAll('.arc')];
      const widest = Math.max(...ribbons.map(r => r.rate));
      const HOP = calm ? 0 : 420, REST = calm ? 900 : 1900;
      let running = [], sweeps = null;
      const stop = () => { running.forEach(a => a.cancel()); running = []; if (sweeps) sweeps.remove(); sweeps = null; };
      const run = (node, keyframes, options) => running.push(node.animate(keyframes, { fill: 'both', ...options }));

      // The pulse spreads from the operator `start` over the ribbons, one hop after the other, fainter with every hop
      // and as bright as the rate; then everything relaxes back to rest.
      const pulse = start => {
        stop();
        const dist = new Map([[start, 0]]);
        for (let queue = [start]; queue.length;) {
          const u = queue.shift();
          for (const r of ribbons) {
            const v = r.a === u ? r.b : r.b === u ? r.a : -1;
            if (v >= 0 && !dist.has(v)) { dist.set(v, dist.get(u) + 1); queue.push(v); }
          }
        }
        const hops = Math.max(...dist.values());
        const total = hops * HOP + REST;
        sweeps = document.createElementNS(svg.namespaceURI, 'g');
        svg.insertBefore(sweeps, svg.querySelector('.arcs'));
        for (const r of ribbons) {
          const near = Math.min(dist.has(r.a) ? dist.get(r.a) : Infinity, dist.has(r.b) ? dist.get(r.b) : Infinity);
          if (near === Infinity) { run(r.path, [{ fillOpacity: .8 }, { fillOpacity: .1 }], { duration: 200 }); continue; }
          const t0 = near * HOP / total, level = Math.min(.95, (.3 + .65 * Math.sqrt(r.rate / widest)) * Math.pow(.86, near));
          run(r.path, [
            { fillOpacity: .8, offset: 0 }, { fillOpacity: .08, offset: Math.min(.06, t0) },
            { fillOpacity: .08, offset: t0 }, { fillOpacity: level, offset: t0 + .04 },
            { fillOpacity: level, offset: Math.min(.7, t0 + .3) }, { fillOpacity: .8, offset: 1 },
          ], { duration: total });
          if (!calm) {
            const trace = r.path.cloneNode(false);
            trace.removeAttribute('data-source'); trace.removeAttribute('data-target');
            Object.assign(trace.style, { fill: 'none', stroke: '#fff', strokeWidth: '0.014', pointerEvents: 'none' });
            trace.setAttribute('pathLength', '1');
            trace.setAttribute('stroke-dasharray', '.16 .84');
            sweeps.append(trace);
            run(trace, [{ strokeDashoffset: .16, opacity: .95 }, { strokeDashoffset: -.84, opacity: .2 }],
              { duration: 1.5 * HOP + 300, delay: near * HOP, easing: 'ease-out' });
          }
        }
        arcs.forEach((arc, k) => {
          const shape = arc.querySelector('path'), base = shape.getAttribute('fill');
          const d = dist.get(k);
          if (d === undefined) { run(shape, [{ fillOpacity: 1 }, { fillOpacity: .35 }, { fillOpacity: 1 }], { duration: total }); return; }
          run(shape, [{ fill: base }, { fill: '#fff', offset: .08 }, { fill: base }],
            { duration: REST * (d ? .6 : 1), delay: d * HOP, easing: 'ease-out' });
        });
        Promise.all(running.map(a => a.finished)).then(stop, () => {});
      };

      const name = arc => arc.querySelector('text').textContent;
      const home = arcs.findIndex(arc => name(arc) === 'Bz');
      svg.addEventListener('click', event => {
        const arc = event.target.closest('.arc');
        const ribbon = ribbons.find(r => r.path === event.target.closest('.ribbon'));
        pulse(arc ? arcs.indexOf(arc) : ribbon ? ribbon.a : Math.max(0, home));
      });
      arcs.forEach(arc => {
        arc.setAttribute('tabindex', '0');
        arc.setAttribute('role', 'button');
        arc.setAttribute('aria-label', 'Send a pulse from ' + name(arc));
        arc.addEventListener('keydown', e => {
          if (e.key === 'Enter' || e.key === ' ') { pulse(arcs.indexOf(arc)); e.preventDefault(); }
        });
      });
      hint.textContent = 'Ribbon width is the rate; click an operator to follow the transfer.';
    };

    fetch(still.src).then(r => (r.ok ? r.text() : Promise.reject(r.status))).then(text => {
      const svg = document.importNode(new DOMParser().parseFromString(text, 'image/svg+xml').documentElement, true);
      svg.classList.add('chord');
      svg.setAttribute('width', '900'); svg.setAttribute('height', '900');
      stage.replaceChildren(svg);
      wire(svg);
    }).catch(() => {});
    return slide;
  };

  const slides = [animSlide, chordSlide(), ...extra.map(([, slide]) => slide), imgSlide];
  const names = ['Animation', 'Relaxation', ...extra.map(([name]) => name), 'Picture'];
  track.append(...slides);

  const prev = el('button', 'gallery-btn prev', { type: 'button', textContent: '‹' });
  const next = el('button', 'gallery-btn next', { type: 'button', textContent: '›' });
  prev.setAttribute('aria-label', 'Previous figure');
  next.setAttribute('aria-label', 'Next figure');
  const dots = el('div', 'gallery-dots');
  const dotButtons = names.map((name, i) => {
    const b = el('button', 'gallery-dot', { type: 'button' });
    b.setAttribute('aria-label', `Show figure ${i + 1}: ${name.toLowerCase()}`);
    b.addEventListener('click', () => show(i));
    dots.append(b);
    return b;
  });

  const frame = el('div', 'gallery-frame');
  frame.append(track, prev, next);
  figure.className = 'gallery';
  figure.replaceChildren(frame, dots);

  let current = 0;
  const show = i => {
    current = Math.max(0, Math.min(slides.length - 1, i));
    track.scrollTo({ left: slides[current].offsetLeft - track.offsetLeft, behavior: 'smooth' });
    mark();
  };
  const mark = () => {
    dotButtons.forEach((b, i) => b.setAttribute('aria-current', i === current ? 'true' : 'false'));
    prev.disabled = current === 0;
    next.disabled = current === slides.length - 1;
  };
  prev.addEventListener('click', () => show(current - 1));
  next.addEventListener('click', () => show(current + 1));
  track.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight') { show(current + 1); e.preventDefault(); }
    if (e.key === 'ArrowLeft') { show(current - 1); e.preventDefault(); }
  });
  let settle = 0;
  track.addEventListener('scroll', () => {                // swipes and wheel scrolls update the dots
    clearTimeout(settle);
    settle = setTimeout(() => { current = Math.round(track.scrollLeft / track.clientWidth); mark(); }, 60);
  }, { passive: true });
  mark();

  // ---- the animation ----
  if (!animated) return;
  const U = n => gl.getUniformLocation(program, n);
  const uRes = U('res'), uT = U('t'), uDpr = U('dpr'), uCtr = U('ctr'), uHand = U('hand');
  const dpr = Math.min(devicePixelRatio || 1, 2);
  let ctr = [.5, .5], target = [.5, .5], hand = -1, handTarget = -1;

  const resize = () => {
    canvas.width = canvas.clientWidth * dpr; canvas.height = canvas.clientHeight * dpr;
    gl.viewport(0, 0, canvas.width, canvas.height);
  };
  new ResizeObserver(resize).observe(canvas);
  resize();

  canvas.addEventListener('pointermove', e => {
    const r = canvas.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width, y = (r.bottom - e.clientY) / r.height;
    target = [.5 + (x - .5) * .6, .5 + (y - .5) * .6];    // follow, but stay near the middle
  });
  canvas.addEventListener('pointerleave', () => { target = [.5, .5]; });
  canvas.addEventListener('click', () => { handTarget = -handTarget; });   // click, not pointerdown: a swipe must not flip it

  let visible = true, raf = 0;
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (visible && !raf) raf = requestAnimationFrame(draw);
  }, { root: null, threshold: 0.1 }).observe(canvas);       // also false while the slide is scrolled out of the track

  const t0 = performance.now();
  function draw(now) {
    raf = 0;
    if (!visible || document.hidden) return;
    ctr = ctr.map((m, i) => m + (target[i] - m) * .03);
    hand += (handTarget - hand) * .025;
    gl.uniform2f(uRes, canvas.width, canvas.height);
    gl.uniform1f(uT, (now - t0) / 1000);
    gl.uniform1f(uDpr, dpr);
    gl.uniform2f(uCtr, ctr[0], ctr[1]);
    gl.uniform1f(uHand, hand);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    raf = requestAnimationFrame(draw);
  }
  document.addEventListener('visibilitychange', () => { if (!document.hidden && !raf) raf = requestAnimationFrame(draw); });
  raf = requestAnimationFrame(draw);
})();
