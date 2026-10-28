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
  if (!img || !caption || matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl', { antialias: false });
  if (!gl) return;

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
  const shader = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, vs));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return;   // keep the static figure
  gl.useProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(program, 'p');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  // ---- the gallery: the animation first, the existing picture second ----
  const el = (tag, cls, attrs = {}) => Object.assign(document.createElement(tag), { className: cls }, attrs);
  const track = el('div', 'gallery-track', { tabIndex: 0 });
  track.setAttribute('role', 'group');
  track.setAttribute('aria-roledescription', 'carousel');
  track.setAttribute('aria-label', 'Figures: animation and picture');

  const animSlide = el('div', 'gallery-slide');
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', 'Animated interference of rotating spin-phase spirals');
  canvas.style.aspectRatio = String((img.naturalWidth && img.naturalHeight) ? img.naturalWidth / img.naturalHeight : 1397 / 1202);
  animSlide.append(canvas, el('div', 'gallery-caption', {
    textContent: 'Interference of rotating spin-phase spirals. Move the pointer to steer the centre; click to reverse one spiral.',
  }));

  const imgSlide = el('div', 'gallery-slide');
  caption.className = 'gallery-caption';
  imgSlide.append(img, caption);

  track.append(animSlide, imgSlide);
  const slides = [animSlide, imgSlide];
  const names = ['Animation', 'Picture'];

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
