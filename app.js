import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ───────────────────────── Sabitler ─────────────────────────
const JOB_SITES = [
  { name: "BİM",  url: "https://www.bim.com.tr/categories/676/hizli-basvuru.aspx" },
  { name: "A101", url: "https://www.a101.com.tr/kariyer" },
  { name: "ŞOK",  url: "https://kurumsal.sokmarket.com.tr/tr/is-basvurusu" },
];
const WASM_URL  = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const EYE_IRIS_GAIN_X = 115;   // iris yatay kayması → derece
const EYE_IRIS_GAIN_Y = 95;    // iris dikey kayması → derece
const BLEND_GAIN_X = 22;       // blendshape yatay bakış katkısı (derece)
const BLEND_GAIN_Y = 28;       // blendshape dikey bakış katkısı (derece)
const MAX_POINTS = 360;        // bölge başına saklanan maksimum nokta

const ALLOWED_REGIONS = {
  screen: { label: "Ekran",   color: "#4dabf7" },
  tablet: { label: "Tablet",  color: "#b197fc" },
  extra:  { label: "Ek alan", color: "#ffa94d" },
};
const ALL_REGIONS = {
  ...ALLOWED_REGIONS,
  phone:  { label: "Telefon", color: "#ff6b81" },
};

const DEFAULTS = { tol: 9, delay: 2.5, cooldown: 30, vol: 0.25, faceLost: true, ecoMode: false };
const settings = { ...DEFAULTS, ...JSON.parse(localStorage.getItem("ob_settings2") || "{}") };
const calib = {
  screen: [], tablet: [], extra: [], phone: [], openBase: null,
  ...JSON.parse(localStorage.getItem("ob_calib2") || "{}"),
};
if (!Array.isArray(calib.phone)) calib.phone = [];

let hulls = {};
let calibBounds = { x0: -35, x1: 35, y0: -25, y1: 45, hasPoints: false };

// ───────────────────────── 1-Euro Adaptif Filtre ─────────────────────────
// Sabit bakışta titremeyi sıfırlar, hızlı göz/kafa hareketinde gecikmeyi kaldırır.
class OneEuroFilter {
  constructor(minCutoff = 1.15, beta = 0.04, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }
  reset() {
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }
  alpha(cutoff, dt) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / Math.max(dt, 0.001));
  }
  filter(x, tMs) {
    if (this.xPrev === null || this.tPrev === null) {
      this.xPrev = x;
      this.tPrev = tMs;
      return x;
    }
    const dt = Math.max((tMs - this.tPrev) / 1000, 0.008);
    this.tPrev = tMs;
    const dx = (x - this.xPrev) / dt;
    const edx = this.dxPrev + this.alpha(this.dCutoff, dt) * (dx - this.dxPrev);
    this.dxPrev = edx;
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    const out = this.xPrev + this.alpha(cutoff, dt) * (x - this.xPrev);
    this.xPrev = out;
    return out;
  }
}

const filterX = new OneEuroFilter(1.2, 0.045, 1.0);
const filterY = new OneEuroFilter(1.1, 0.045, 1.0);

// ───────────────────────── DOM & Önbellekli Yazıcılar ─────────────────────────
const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), octx = overlay.getContext("2d", { alpha: true });
const mapCv = $("map"), mctx = mapCv.getContext("2d", { alpha: false });

const domCache = {};
function setText(id, val) {
  if (domCache[id] !== val) {
    domCache[id] = val;
    const el = $(id);
    if (el) el.textContent = val;
  }
}
function setBar(id, pct, color) {
  const wKey = id + "_w", cKey = id + "_c";
  const rounded = Math.round(pct * 2) / 2;
  const el = $(id);
  if (!el) return;
  if (domCache[wKey] !== rounded) {
    domCache[wKey] = rounded;
    el.style.width = `${rounded}%`;
  }
  if (color && domCache[cKey] !== color) {
    domCache[cKey] = color;
    el.style.background = color;
  }
}
function setStatus(cls, text) {
  const key = cls + "|" + text;
  if (domCache.status !== key) {
    domCache.status = key;
    const el = $("status");
    el.className = "status " + cls;
    el.textContent = text;
  }
}

// ───────────────────────── Durum ─────────────────────────
let landmarker = null, audioCtx = null, running = false, calibrating = false, cancelCalib = false;
let lastVideoTime = -1, lastTickAt = null;
let gaze = null;                 // yumuşatılmış bakış {x, y} (derece)
let lastValidRaw = null, closedMs = 0;
let isAway = false, awayMs = 0, lastTrigger = -Infinity, faceLostFired = false;
let count = 0, focusMs = 0, totalMs = 0;
let sampling = null;             // { samples: [[x,y]] }
let fpsFrames = 0, fpsLastAt = performance.now();

// ───────────────────────── Ayar Arayüzü ─────────────────────────
const getTickMs = () => (settings.ecoMode ? 90 : 45);
const save = () => localStorage.setItem("ob_settings2", JSON.stringify(settings));

function bindRange(id, key, fmt) {
  const el = $(id);
  el.value = settings[key];
  const show = () => setText(id + "Val", fmt(settings[key]));
  el.addEventListener("input", () => {
    settings[key] = parseFloat(el.value);
    save();
    show();
    if (key === "tol") drawMap(true);
  });
  show();
}
bindRange("tol", "tol", (v) => `${v}°`);
bindRange("delay", "delay", (v) => `${v} sn`);
bindRange("cooldown", "cooldown", (v) => `${v} sn`);
bindRange("vol", "vol", (v) => `%${Math.round(v * 100)}`);

$("faceLost").checked = settings.faceLost;
$("faceLost").addEventListener("change", (e) => { settings.faceLost = e.target.checked; save(); });

$("ecoMode").checked = !!settings.ecoMode;
$("ecoMode").addEventListener("change", (e) => {
  settings.ecoMode = e.target.checked;
  save();
  if (running) ticker.postMessage(getTickMs());
});

function saveCalib() {
  localStorage.setItem("ob_calib2", JSON.stringify(calib));
  rebuildHulls();
  for (const k in ALL_REGIONS) {
    const tag = $("tag-" + k);
    if (!tag) continue;
    const has = calib[k]?.length >= 3;
    tag.textContent = has ? "kayıtlı ✓" : "yok";
    tag.classList.toggle("done", has);
  }
}

// ───────────────────────── Ses ─────────────────────────
function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
}
function tone(freq, start, dur, vol) {
  if (vol <= 0.001) return;
  const o = audioCtx.createOscillator(), g = audioCtx.createGain();
  o.type = "sine";
  o.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, start);
  g.gain.linearRampToValueAtTime(vol, start + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
  o.connect(g).connect(audioCtx.destination);
  o.start(start);
  o.stop(start + dur + 0.05);
}
function chime() {
  ensureAudio();
  const t = audioCtx.currentTime;
  tone(659.25, t, 0.75, settings.vol);
  tone(987.77, t + 0.14, 0.75, settings.vol);
}
function beep(freq = 880) {
  ensureAudio();
  tone(freq, audioCtx.currentTime, 0.18, Math.max(settings.vol, 0.15));
}

// ───────────────────────── Ceza ─────────────────────────
function punish(reason) {
  const site = JOB_SITES[Math.floor(Math.random() * JOB_SITES.length)];
  chime();
  const w = window.open(site.url, "_blank");
  $("popupWarn").classList.toggle("hidden", !!w);
  setText("count", String(++count));
  const log = $("log");
  const empty = log.querySelector(".emptyLog");
  if (empty) empty.remove();
  const li = document.createElement("li");
  li.textContent = `${new Date().toLocaleTimeString("tr-TR")} — ${reason} → ${site.name} iş başvurusu açıldı 🛒`;
  log.prepend(li);
}

// ───────────────────────── Geometri & Üst-Yüz + Göz Analizi ─────────────────────────
const RAD = 180 / Math.PI;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 });
const cross = (a, b) => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
const len2 = (a) => Math.hypot(a.x, a.y);
const median = (arr) => {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

/**
 * Sadece ÜST YÜZ (4 göz köşesi + burun köprüsü + alın) ve İRİS + BLENDSHAPE kullanılır.
 * El çenedeyken, ağız kapalıyken veya ekranın en altına / tablete bakarken bozulmaz.
 */
function analyze(lm, blendCategories, W, H) {
  const P = (i) => ({ x: lm[i].x * W, y: lm[i].y * H, z: lm[i].z * W });

  // Çok noktalı üst-kafatası düzlemi (el çenede ve saç/kaş hareketine dayanıklı)
  const leftAnchor = mid(P(33), P(133));
  const rightAnchor = mid(P(263), P(362));
  const topAnchor = {
    x: (P(10).x + P(151).x + P(9).x + P(168).x) * 0.25,
    y: (P(10).y + P(151).y + P(9).y + P(168).y) * 0.25,
    z: (P(10).z + P(151).z + P(9).z + P(168).z) * 0.25,
  };
  const baseMid = {
    x: (leftAnchor.x + rightAnchor.x + P(6).x) / 3,
    y: (leftAnchor.y + rightAnchor.y + P(6).y) / 3,
    z: (leftAnchor.z + rightAnchor.z + P(6).z) / 3,
  };

  let n = cross(sub(rightAnchor, leftAnchor), sub(topAnchor, baseMid));
  const nl = Math.hypot(n.x, n.y, n.z) || 1;
  n = { x: n.x / nl, y: n.y / nl, z: n.z / nl };
  if (n.z > 0) n = { x: -n.x, y: -n.y, z: -n.z };

  const headX = Math.atan2(n.x, -n.z) * RAD;
  const headY = Math.asin(clamp(n.y, -1, 1)) * RAD;

  // İris konumu: 468 = sol göz irisi, 473 = sağ göz irisi
  const eyeDefs = [
    { a: 33, b: 133, up: 159, lo: 145, iris: 468 },
    { a: 362, b: 263, up: 386, lo: 374, iris: 473 },
  ];
  let ox = 0, oy = 0, open = 0;
  for (const e of eyeDefs) {
    const A = P(e.a), B = P(e.b), c = mid(A, B);
    const w = len2(sub(B, A)) || 1;
    let u = { x: (B.x - A.x) / w, y: (B.y - A.y) / w };
    if (u.x < 0) u = { x: -u.x, y: -u.y };
    const v = { x: -u.y, y: u.x };
    const irisPt = P(e.iris);
    const d = sub(irisPt, c);
    ox += (d.x * u.x + d.y * u.y) / w * 0.5;
    oy += (d.x * v.x + d.y * v.y) / w * 0.5;
    open += len2(sub(P(e.up), P(e.lo))) / w * 0.5;
  }

  // Blendshape tabanlı göz bakışı ve gerçek göz kırpma tespiti
  let bsX = 0, bsY = 0, blinkScore = 0;
  if (blendCategories && blendCategories.length) {
    const bs = {};
    for (let i = 0; i < blendCategories.length; i++) {
      const c = blendCategories[i];
      bs[c.categoryName] = c.score;
    }
    const lookDown = ((bs.eyeLookDownLeft || 0) + (bs.eyeLookDownRight || 0)) * 0.5;
    const lookUp   = ((bs.eyeLookUpLeft || 0)   + (bs.eyeLookUpRight || 0))   * 0.5;
    // Görüntü koordinatında sağ (+x) ve sol (-x)
    const lookRightImg = ((bs.eyeLookInLeft || 0)  + (bs.eyeLookOutRight || 0)) * 0.5;
    const lookLeftImg  = ((bs.eyeLookOutLeft || 0) + (bs.eyeLookInRight || 0))  * 0.5;

    bsX = (lookRightImg - lookLeftImg) * BLEND_GAIN_X;
    bsY = (lookDown - lookUp) * BLEND_GAIN_Y;
    blinkScore = ((bs.eyeBlinkLeft || 0) + (bs.eyeBlinkRight || 0)) * 0.5;
  }

  const eyeX = ox * EYE_IRIS_GAIN_X + bsX;
  const eyeY = oy * EYE_IRIS_GAIN_Y + bsY;
  return { headX, headY, eyeX, eyeY, open, blinkScore };
}

// ── Konveks Örtü (Monotone Chain) & Önbellekli Sınırlar ──
function convexHull(pts) {
  if (!pts || pts.length < 3) return pts ? pts.slice() : [];
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) {
    while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop();
    lo.push(q);
  }
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop();
    up.push(q);
  }
  lo.pop();
  up.pop();
  return lo.concat(up);
}

function segDist(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], l = dx * dx + dy * dy;
  const t = l ? clamp(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l, 0, 1) : 0;
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}

function hullDist(p, h) {
  if (!h || !h.length) return Infinity;
  if (h.length === 1) return Math.hypot(p[0] - h[0][0], p[1] - h[0][1]);
  let inside = h.length >= 3, best = Infinity;
  for (let i = 0; i < h.length; i++) {
    const a = h[i], b = h[(i + 1) % h.length];
    if ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < 0) inside = false;
    const d = segDist(p, a, b);
    if (d < best) best = d;
  }
  return inside ? 0 : best;
}

function rebuildHulls() {
  hulls = {};
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, countPts = 0;
  for (const k in ALL_REGIONS) {
    const pts = calib[k] || [];
    hulls[k] = convexHull(pts);
    for (let i = 0; i < pts.length; i++) {
      const px = -pts[i][0], py = pts[i][1];
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
      countPts++;
    }
  }
  calibBounds = countPts
    ? { x0: minX - 14, x1: maxX + 14, y0: minY - 14, y1: maxY + 14, hasPoints: true }
    : { x0: -35, x1: 35, y0: -25, y1: 45, hasPoints: false };
}

const hasRegions = () => Object.keys(ALLOWED_REGIONS).some((k) => (calib[k]?.length || 0) >= 3);

function focusDistance(g) {
  const p = [g.x, g.y];
  let bestAllowed = Infinity;
  for (const k in ALLOWED_REGIONS) {
    const d = hullDist(p, hulls[k]);
    if (d < bestAllowed) bestAllowed = d;
  }
  // Eğer kullanıcı telefon (yasak) alanını tanıttıysa ve bakışı telefon alanının içindeyse:
  if (hulls.phone && hulls.phone.length >= 3) {
    const phoneDist = hullDist(p, hulls.phone);
    if (phoneDist < 2.5 && bestAllowed > 1.5) {
      return Math.max(bestAllowed / settings.tol, 1.35);
    }
  }
  return bestAllowed / settings.tol; // ≤1 → odaklı
}

/** Sadece aşırı uç sıçramaları temizler, ekranın alt köşelerini korur */
function cleanPoints(pts) {
  if (pts.length < 12) return pts;
  const cx = median(pts.map((p) => p[0])), cy = median(pts.map((p) => p[1]));
  const d = pts.map((p) => Math.hypot(p[0] - cx, p[1] - cy));
  const lim = median(d) * 3.2 + 5;
  return pts.filter((_, i) => d[i] <= lim);
}
function capPoints(pts) {
  if (pts.length <= MAX_POINTS) return pts;
  const step = pts.length / MAX_POINTS;
  return Array.from({ length: MAX_POINTS }, (_, i) => pts[Math.floor(i * step)]);
}

// ───────────────────────── Kalibrasyon ─────────────────────────
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const ov = $("calOverlay"), dot = $("dot"), calText = $("calText"), calProg = $("calProgress");

$("calCancel").addEventListener("click", () => { cancelCalib = true; });
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && calibrating) cancelCalib = true;
});

function finishSampling(target, append) {
  const s = sampling;
  sampling = null;
  if (cancelCalib || !s) {
    setStatus("idle", "Kalibrasyon iptal edildi");
    return;
  }
  const pts = cleanPoints(s.samples);
  if (pts.length < 10) {
    setStatus("bad", "Yüz/gözler yeterince algılanamadı, kameraya bakarak tekrar dene");
    return;
  }
  calib[target] = capPoints(append ? (calib[target] || []).concat(pts) : pts);
  saveCalib();
  awayMs = 0;
  isAway = false;
  setStatus("ok", `${ALL_REGIONS[target].label} alanı başarıyla kaydedildi ✓`);
  drawMap(true);
}

function placeDot(p) {
  dot.style.left = `${p[0] * innerWidth}px`;
  dot.style.top = `${p[1] * innerHeight}px`;
}
function moveDot(a, b, ms, prog0, prog1) {
  return new Promise((res) => {
    const t0 = performance.now();
    const step = (now) => {
      if (cancelCalib) return res();
      const t = Math.min(1, (now - t0) / ms);
      const e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
      placeDot([a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e]);
      calProg.style.width = `${(prog0 + (prog1 - prog0) * t) * 100}%`;
      t < 1 ? requestAnimationFrame(step) : res();
    };
    requestAnimationFrame(step);
  });
}

async function calibrateScreen() {
  calibrating = true;
  cancelCalib = false;
  calProg.style.width = "0%";
  ov.classList.remove("hidden");
  dot.classList.remove("hidden");
  try { await document.documentElement.requestFullscreen(); } catch {}
  await sleep(400);
  // Ekranın üst, orta ve özellikle ALT kenarlarını eksiksiz tarayan rota
  const path = [
    [0.03, 0.04], [0.97, 0.04], [0.97, 0.50], [0.03, 0.50],
    [0.03, 0.96], [0.50, 0.97], [0.97, 0.96], [0.50, 0.04], [0.50, 0.50],
  ];
  calText.textContent = "Kırmızı noktayı GÖZLERİNLE takip et (normal duruşunu koru)";
  placeDot(path[0]);
  await sleep(1200);
  calText.textContent = "Ekran sınırları kaydediliyor…";
  sampling = { samples: [] };
  const segs = path.length - 1;
  for (let i = 1; i < path.length; i++) {
    if (cancelCalib) break;
    await moveDot(path[i - 1], path[i], 1050, (i - 1) / segs, i / segs);
  }
  await sleep(200);
  if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
  ov.classList.add("hidden");
  dot.classList.add("hidden");
  finishSampling("screen", false);
  calibrating = false;
}

async function calibrateFree(target) {
  calibrating = true;
  cancelCalib = false;
  calProg.style.width = "0%";
  const label = ALL_REGIONS[target].label.toLowerCase();
  ov.classList.remove("hidden");
  for (let i = 3; i > 0; i--) {
    if (cancelCalib) break;
    calText.textContent = `${i}… İlk bip sesinden sonra ${label} alanının köşelerine ve ortasına bak.`;
    await sleep(900);
  }
  if (!cancelCalib) {
    beep(660);
    sampling = { samples: [] };
    const steps = 60;
    for (let i = 0; i < steps; i++) {
      if (cancelCalib) break;
      const rem = Math.ceil(((steps - i) * 100) / 1000);
      calText.textContent = `${ALL_REGIONS[target].label} alanının tamamını gözlerinle tara… (${rem} sn)`;
      calProg.style.width = `${((i + 1) / steps) * 100}%`;
      await sleep(100);
    }
    beep(990);
  }
  ov.classList.add("hidden");
  finishSampling(target, target === "extra");
  calibrating = false;
}

// ───────────────────────── Hızlı Çizim (Sıfır Tahsisat) ─────────────────────────
const KEY_LANDMARKS = [33, 133, 362, 263, 159, 145, 386, 374, 10, 151, 9, 168, 6];

function drawOverlay(lm, color) {
  if (document.hidden && !document.pictureInPictureElement) return;
  const vw = video.videoWidth || 640, vh = video.videoHeight || 480;
  // Sadece boyut değiştiğinde canvas belleğini yeniden boyutlandır (GPU optimizasyonu)
  if (overlay.width !== vw || overlay.height !== vh) {
    overlay.width = vw;
    overlay.height = vh;
  }
  octx.clearRect(0, 0, vw, vh);
  if (!lm) return;

  octx.fillStyle = color;
  for (let i = 0; i < KEY_LANDMARKS.length; i++) {
    const p = lm[KEY_LANDMARKS[i]];
    octx.fillRect(p.x * vw - 2, p.y * vh - 2, 4, 4);
  }
  // İris merkezleri
  octx.fillStyle = "#ffffff";
  for (const idx of [468, 473]) {
    const p = lm[idx];
    octx.beginPath();
    octx.arc(p.x * vw, p.y * vh, 3.5, 0, Math.PI * 2);
    octx.fill();
  }
}

function drawMap(force = false) {
  if (document.hidden && !force) return;
  const W = mapCv.width, H = mapCv.height;
  mctx.fillStyle = "#0d1018";
  mctx.fillRect(0, 0, W, H);

  let { x0, x1, y0, y1 } = calibBounds;
  if (gaze) {
    const gx = -gaze.x, gy = gaze.y;
    if (gx - 10 < x0) x0 = gx - 10;
    if (gx + 10 > x1) x1 = gx + 10;
    if (gy - 10 < y0) y0 = gy - 10;
    if (gy + 10 > y1) y1 = gy + 10;
  }

  const sc = Math.min(W / Math.max(x1 - x0, 20), H / Math.max(y1 - y0, 20));
  const ox = (W - (x1 - x0) * sc) * 0.5;
  const oy = (H - (y1 - y0) * sc) * 0.5;
  const T = (p) => [ox + (-p[0] - x0) * sc, oy + (p[1] - y0) * sc];

  for (const k in ALL_REGIONS) {
    const h = hulls[k];
    if (!h || h.length < 3) continue;
    mctx.beginPath();
    for (let i = 0; i < h.length; i++) {
      const [x, y] = T(h[i]);
      i ? mctx.lineTo(x, y) : mctx.moveTo(x, y);
    }
    mctx.closePath();
    if (k !== "phone") {
      mctx.lineWidth = settings.tol * sc * 2;
      mctx.lineJoin = "round";
      mctx.strokeStyle = ALL_REGIONS[k].color + "20";
      mctx.stroke();
    }
    mctx.fillStyle = ALL_REGIONS[k].color + "33";
    mctx.fill();
    mctx.lineWidth = 2;
    mctx.strokeStyle = ALL_REGIONS[k].color;
    mctx.stroke();
  }

  if (gaze) {
    const [x, y] = T([gaze.x, gaze.y]);
    mctx.beginPath();
    mctx.arc(x, y, 7, 0, Math.PI * 2);
    mctx.fillStyle = isAway ? "#ff5c77" : "#38d996";
    mctx.fill();
    mctx.lineWidth = 2;
    mctx.strokeStyle = "#ffffff";
    mctx.stroke();
  }

  if (!hasRegions()) {
    mctx.fillStyle = "#6c768d";
    mctx.font = "14px -apple-system, sans-serif";
    mctx.textAlign = "center";
    mctx.fillText("Kalibrasyon yaptığında izinli odak bölgelerin burada görünecek", W / 2, H / 2);
  }
}

// ───────────────────────── Ana Analiz Döngüsü ─────────────────────────
function tick() {
  if (!running || video.readyState < 2) return;
  const now = performance.now();
  const dt = lastTickAt ? Math.min(now - lastTickAt, 1000) : 0;
  lastTickAt = now;

  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  // FPS sayacı
  fpsFrames++;
  if (now - fpsLastAt >= 1000) {
    const fps = Math.round((fpsFrames * 1000) / (now - fpsLastAt));
    setText("hudBadge", `${fps} FPS${settings.ecoMode ? " • Eko" : ""}`);
    fpsFrames = 0;
    fpsLastAt = now;
  }

  const res = landmarker.detectForVideo(video, now);
  const lm = res.faceLandmarks?.[0];
  const blends = res.faceBlendshapes?.[0]?.categories;

  if (lm && lm.length >= 478) {
    const f = analyze(lm, blends, video.videoWidth || 640, video.videoHeight || 480);

    // Gerçek göz kırpma tespiti (ekranın altına / tablete bakarken kısılan gözü kırpma saymaz!)
    const isBlinking = f.blinkScore > 0.62;
    if (isBlinking) {
      closedMs += dt;
    } else {
      closedMs = 0;
    }

    let raw;
    if (isBlinking && closedMs < 420 && lastValidRaw) {
      // Anlık göz kırpmada son geçerli bakışı koru
      raw = lastValidRaw;
    } else if (closedMs >= 1800 && lastValidRaw) {
      // Gözler 1.8 saniyeden uzun süre tamamen kapalıysa (uyuklama)
      raw = { x: lastValidRaw.x, y: lastValidRaw.y + 30 };
    } else {
      raw = { x: f.headX + f.eyeX, y: f.headY + f.eyeY };
      lastValidRaw = raw;
    }

    gaze = {
      x: filterX.filter(raw.x, now),
      y: filterY.filter(raw.y, now),
    };

    if (sampling && !isBlinking) {
      sampling.samples.push([gaze.x, gaze.y]);
    }
  } else {
    gaze = null;
    filterX.reset();
    filterY.reset();
  }

  drawMap();
  if (calibrating) {
    drawOverlay(lm, "#fcc419");
    return;
  }

  if (!hasRegions()) {
    drawOverlay(lm, "#8f98ad");
    setStatus("idle", lm ? "📺 Başlamak için önce ekranı kalibre et" : "Yüz aranıyor… Kameraya bak");
    return;
  }

  let away, reason;
  if (gaze) {
    faceLostFired = false;
    const d = focusDistance(gaze);
    away = isAway ? d > 0.82 : d > 1.0; // Histerezis: sınırda titremeyi önler
    reason = "Odak alanının dışına (telefona/etrafa) baktın";
    setText("distVal", d === 0 ? "alanın içinde ✓" : `${d.toFixed(2)}x`);
    setBar("distBar", Math.min(d / 2, 1) * 100, away ? "#ff5c77" : "#38d996");
  } else {
    away = settings.faceLost && !faceLostFired;
    reason = "Yüzün kameradan çıktı";
    setText("distVal", "yüz yok");
  }
  isAway = away;

  const delayMs = settings.delay * 1000;
  awayMs = away ? awayMs + dt : Math.max(0, awayMs - dt * 2.2);
  setText("awayVal", `${(awayMs / 1000).toFixed(1)} / ${settings.delay} sn`);
  setBar("awayBar", Math.min(awayMs / delayMs, 1) * 100);

  const coolingDown = now - lastTrigger < settings.cooldown * 1000;
  if (awayMs >= delayMs) {
    if (!coolingDown) {
      punish(reason);
      lastTrigger = now;
      if (!gaze) faceLostFired = true;
    }
    awayMs = 0;
  }

  totalMs += dt;
  if (away) {
    setStatus(
      awayMs > delayMs * 0.4 ? "bad" : "warn",
      coolingDown ? "🚨 Yakalandın! İşine odaklan." : gaze ? "👀 Odak alanından çıktın!" : "🙈 Yüzün görünmüyor"
    );
    drawOverlay(lm, "#ff5c77");
  } else {
    focusMs += dt;
    setStatus("ok", gaze ? "✅ Odaklısın, harika gidiyorsun!" : "Yüz yok (yakalama kapalı)");
    drawOverlay(lm, "#38d996");
  }

  const sec = Math.floor(focusMs / 1000);
  setText("focusTime", `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`);
  if (totalMs > 1000) {
    setText("focusPct", `%${Math.round((focusMs / totalMs) * 100)}`);
  }
}

// Arka plan sekmesinde zamanlayıcı durmasın diye Web Worker
const ticker = new Worker(
  URL.createObjectURL(
    new Blob(
      ["let id;onmessage=e=>{clearInterval(id);if(e.data>0)id=setInterval(()=>postMessage(0),e.data)}"],
      { type: "text/javascript" }
    )
  )
);
ticker.onmessage = () => {
  try { tick(); } catch (e) { console.error(e); }
};

// ───────────────────────── Model & Kamera Başlatma ─────────────────────────
async function loadModel() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    minFaceDetectionConfidence: 0.45,
    minFacePresenceConfidence: 0.45,
    minTrackingConfidence: 0.45,
  });
  try { return await FaceLandmarker.createFromOptions(fileset, opts("GPU")); }
  catch { return await FaceLandmarker.createFromOptions(fileset, opts("CPU")); }
}

const calBtns = () => document.querySelectorAll(".calBtn");

$("startBtn").addEventListener("click", async () => {
  ensureAudio();
  if (running) {
    running = false;
    ticker.postMessage(0);
    if (document.pictureInPictureElement) {
      await document.exitPictureInPicture().catch(() => {});
    }
    video.srcObject?.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
    $("placeholder").style.display = "";
    $("hudBadge").classList.add("hidden");
    $("startBtn").textContent = "▶️ Kamerayı Başlat";
    $("pipBtn").disabled = true;
    calBtns().forEach((b) => (b.disabled = true));
    setStatus("idle", "Durduruldu");
    return;
  }
  $("startBtn").disabled = true;
  try {
    // Model indirme ve kamera açılışını paralel başlat (çok daha hızlı açılış!)
    setText("loadMsg", "Yüz analiz motoru ve kamera hazırlanıyor…");
    const [model, stream] = await Promise.all([
      landmarker ? Promise.resolve(landmarker) : loadModel(),
      navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 24, max: 30 },
          facingMode: "user",
        },
        audio: false,
      }),
    ]);
    landmarker = model;
    video.srcObject = stream;
    await video.play();
    $("placeholder").style.display = "none";
    $("hudBadge").classList.remove("hidden");
    setText("loadMsg", "");
    running = true;
    lastTickAt = null;
    ticker.postMessage(getTickMs());
    $("startBtn").textContent = "⏹ Durdur";
    $("pipBtn").disabled = !document.pictureInPictureEnabled;
    calBtns().forEach((b) => (b.disabled = false));
  } catch (e) {
    console.error(e);
    setText("loadMsg", "Kamera/Model Hatası: " + (e.message || e));
  }
  $("startBtn").disabled = false;
});

// Mini Pencere (Picture-in-Picture) — arka planda kamera yavaşlamasını tamamen engeller
$("pipBtn").addEventListener("click", async () => {
  try {
    if (document.pictureInPictureElement) {
      await document.exitPictureInPicture();
    } else if (video.readyState >= 2) {
      await video.requestPictureInPicture();
    }
  } catch (e) {
    console.error(e);
  }
});

// Paylaşım butonu
$("shareBtn").addEventListener("click", async () => {
  const url = window.location.href;
  try {
    await navigator.clipboard.writeText(url);
    $("shareBtn").textContent = "✅ Link Kopyalandı!";
    setTimeout(() => ($("shareBtn").textContent = "🔗 Linki Kopyala"), 2000);
  } catch {
    prompt("Site bağlantısını kopyala:", url);
  }
});

$("cal-screen").addEventListener("click", () => !calibrating && calibrateScreen());
$("cal-tablet").addEventListener("click", () => !calibrating && calibrateFree("tablet"));
$("cal-extra").addEventListener("click", () => !calibrating && calibrateFree("extra"));
$("cal-phone").addEventListener("click", () => !calibrating && calibrateFree("phone"));

document.querySelectorAll(".clr").forEach((b) =>
  b.addEventListener("click", () => {
    calib[b.dataset.r] = [];
    saveCalib();
    drawMap(true);
  })
);
$("testSound").addEventListener("click", chime);
$("testOpen").addEventListener("click", () => punish("Test"));

saveCalib();
drawMap(true);
