import { FaceLandmarker, ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ───────────────────────── Sabitler ─────────────────────────
const JOB_SITES = [
  { name: "BİM",  url: "https://www.bim.com.tr/categories/676/hizli-basvuru.aspx" },
  { name: "A101", url: "https://www.a101.com.tr/kariyer" },
  { name: "ŞOK",  url: "https://kurumsal.sokmarket.com.tr/tr/is-basvurusu" },
];
const WASM_URL  = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const OBJ_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite";

// Kamerada elde tutulunca yakalanacak dikkat dağıtıcı nesneler
const BANNED_OBJECTS = new Set(["cell phone", "book", "remote", "scissors", "teddy bear", "toothbrush", "hair drier"]);

const MAX_POINTS = 360;

const ALLOWED_REGIONS = {
  screen: { label: "Ekran",   color: "#4dabf7" },
  tablet: { label: "Tablet",  color: "#b197fc" },
  extra:  { label: "Ek alan", color: "#ffa94d" },
};
const ALL_REGIONS = {
  ...ALLOWED_REGIONS,
  phone:  { label: "El / Telefon", color: "#ff5c77" },
};

// v3.0 sıkı varsayılanlar: tol=4.5°, handSens=1.3
const DEFAULTS = { tol: 4.5, handSens: 1.3, delay: 2.0, cooldown: 30, vol: 0.25, faceLost: true, ecoMode: false };
const settings = { ...DEFAULTS, ...JSON.parse(localStorage.getItem("ob_settings3") || "{}") };
const calib = {
  screen: [], tablet: [], extra: [], phone: [], openBase: null, vergBase: null,
  ...JSON.parse(localStorage.getItem("ob_calib3") || "{}"),
};
for (const k in ALL_REGIONS) if (!Array.isArray(calib[k])) calib[k] = [];

let hulls = {};
let calibBounds = { x0: -30, x1: 30, y0: -22, y1: 38, hasPoints: false };

// ───────────────────────── 1-Euro Adaptif Filtre ─────────────────────────
class OneEuroFilter {
  constructor(minCutoff = 1.25, beta = 0.05, dCutoff = 1.0) {
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

const filterX = new OneEuroFilter(1.25, 0.055, 1.0);
const filterY = new OneEuroFilter(1.20, 0.060, 1.0);
const filterDown = new OneEuroFilter(1.10, 0.04, 1.0);
const filterOpen = new OneEuroFilter(1.10, 0.04, 1.0);
const filterVerg = new OneEuroFilter(0.95, 0.03, 1.0);

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
let landmarker = null, objDetector = null, audioCtx = null;
let running = false, calibrating = false, cancelCalib = false;
let lastVideoTime = -1, lastTickAt = null, lastObjCheckAt = 0;
let detectedObject = null, detectedObjUntil = 0;
let gaze = null;                 // { x, y, eyeDown, open, verg }
let lastValidRaw = null, closedMs = 0, rollingOpen = 0.28;
let isAway = false, awayMs = 0, lastTrigger = -Infinity, faceLostFired = false;
let count = 0, focusMs = 0, totalMs = 0;
let sampling = null;             // { samples: [[x, y, eyeDown, open, verg]] }
let fpsFrames = 0, fpsLastAt = performance.now();

// ───────────────────────── Ayar Arayüzü ─────────────────────────
const getTickMs = () => (settings.ecoMode ? 90 : 45);
const save = () => localStorage.setItem("ob_settings3", JSON.stringify(settings));

function bindRange(id, key, fmt) {
  const el = $(id);
  if (!el) return;
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
bindRange("handSens", "handSens", (v) => `${v.toFixed(1)}x`);
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
  localStorage.setItem("ob_calib3", JSON.stringify(calib));
  rebuildHulls();
  for (const k in ALL_REGIONS) {
    const tag = $("tag-" + k);
    if (!tag) continue;
    const has = (calib[k]?.length || 0) >= 3;
    tag.textContent = has ? (k === "phone" ? "yasaklı ✓" : "kayıtlı ✓") : "yok";
    tag.classList.toggle("done", has && k !== "phone");
    tag.classList.toggle("dangerDone", has && k === "phone");
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

// ───────────────────────── 5D Geometri & Yüksek Hassasiyetli Analiz ─────────────────────────
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
 * 5 Boyutlu Odak Vektörü çıkarır:
 * 1) Şakak-Göz-Burun Köprüsü hibrit kafa açısı (headX, headY) -> En ufak kafa eğimini yakalar, el çenedeyken bozulmaz!
 * 2) İris + Blendshape + Göz Kapağı Eğimi (eyeX, eyeY, eyeDown) -> Kafa dikken gözle elde tutulan nesneye bakışı yakalar!
 * 3) Göz Bebeği Yakınsaması / Binoküler Vergence (verg) -> Ekrana (60-80cm) değil eldeki yakın nesneye (25-40cm) odaklanmayı yakalar!
 */
function analyze(lm, blendCategories, W, H) {
  const P = (i) => ({ x: lm[i].x * W, y: lm[i].y * H, z: lm[i].z * W });

  // 1) Şakak (127, 356) ve Göz/Burun Köprüsü (33, 133, 263, 362, 6, 168) ile yüksek hassasiyetli 2D/3D Kafa Açısı
  const templeL = P(127), templeR = P(356);
  const templeMid = mid(templeL, templeR);
  const faceW = len2(sub(templeR, templeL)) || 1;

  const leftEyeMid = mid(P(33), P(133));
  const rightEyeMid = mid(P(263), P(362));
  const noseBridge = mid(P(6), P(168));
  const forehead = {
    x: (P(10).x + P(151).x + P(9).x) / 3,
    y: (P(10).y + P(151).y + P(9).y) / 3,
    z: (P(10).z + P(151).z + P(9).z) / 3,
  };

  // Kafa aşağı eğildiğinde burun köprüsü ve gözler şakak çizgisine göre hızla aşağı iner (2D piksel kesinliği)
  const pitch2D = ((noseBridge.y - templeMid.y) / faceW) * 125;
  const yaw2D   = ((noseBridge.x - templeMid.x) / faceW) * 125;

  // 3D Üst-yüz normali
  let n = cross(sub(rightEyeMid, leftEyeMid), sub(forehead, noseBridge));
  const nl = Math.hypot(n.x, n.y, n.z) || 1;
  n = { x: n.x / nl, y: n.y / nl, z: n.z / nl };
  if (n.z > 0) n = { x: -n.x, y: -n.y, z: -n.z };
  const yaw3D   = Math.atan2(n.x, -n.z) * RAD;
  const pitch3D = Math.asin(clamp(n.y, -1, 1)) * RAD;

  const headX = yaw2D * 0.6 + yaw3D * 0.4;
  const headY = pitch2D * 0.65 + pitch3D * 0.35;

  // 2) İris konumu, Göz açıklığı (open) ve Binoküler Yakınsama (Vergence - yakın nesne odağı)
  // Sol göz: dış=33, iç(burun)=133, iris=468 | Sağ göz: iç(burun)=362, dış=263, iris=473
  const eyeDefs = [
    { a: 33, b: 133, outer: 33, inner: 133, up: 159, lo: 145, iris: 468 },
    { a: 362, b: 263, outer: 263, inner: 362, up: 386, lo: 374, iris: 473 },
  ];
  let ox = 0, oy = 0, open = 0, irisVerg = 0;
  for (const e of eyeDefs) {
    const A = P(e.a), B = P(e.b), c = mid(A, B);
    const w = len2(sub(B, A)) || 1;
    let u = { x: (B.x - A.x) / w, y: (B.y - A.y) / w };
    if (u.x < 0) u = { x: -u.x, y: -u.y };
    const v = { x: -u.y, y: u.x };
    const irisPt = P(e.iris);
    const d = sub(irisPt, c);
    ox += ((d.x * u.x + d.y * u.y) / w) * 0.5;
    oy += ((d.x * v.x + d.y * v.y) / w) * 0.5;
    open += (len2(sub(P(e.up), P(e.lo))) / w) * 0.5;

    // İrisin burun köküne (inner corner) yakınlığı -> Elde yakın nesneye bakarken iki göz bebeği içe yaklaşır
    const distToInner = len2(sub(irisPt, P(e.inner))) / w;
    const distToOuter = len2(sub(irisPt, P(e.outer))) / w;
    irisVerg += (distToOuter - distToInner) * 0.5;
  }

  // 3) Blendshape verileri
  let lookDown = 0, lookUp = 0, bsX = 0, bsVerg = 0, blinkScore = 0;
  if (blendCategories && blendCategories.length) {
    const bs = {};
    for (let i = 0; i < blendCategories.length; i++) {
      const c = blendCategories[i];
      bs[c.categoryName] = c.score;
    }
    lookDown = ((bs.eyeLookDownLeft || 0) + (bs.eyeLookDownRight || 0)) * 0.5;
    lookUp   = ((bs.eyeLookUpLeft || 0)   + (bs.eyeLookUpRight || 0))   * 0.5;
    const lookRightImg = ((bs.eyeLookInLeft || 0)  + (bs.eyeLookOutRight || 0)) * 0.5;
    const lookLeftImg  = ((bs.eyeLookOutLeft || 0) + (bs.eyeLookInRight || 0))  * 0.5;
    const lookInBoth   = ((bs.eyeLookInLeft || 0)  + (bs.eyeLookInRight || 0))  * 0.5;
    const lookOutBoth  = ((bs.eyeLookOutLeft || 0) + (bs.eyeLookOutRight || 0)) * 0.5;

    bsX = (lookRightImg - lookLeftImg) * 26;
    bsVerg = (lookInBoth - lookOutBoth) * 14;
    blinkScore = ((bs.eyeBlinkLeft || 0) + (bs.eyeBlinkRight || 0)) * 0.5;
  }

  // Göz kapağı düşüşü (Elde tutulan telefona/nesneye göz ucuyla bakarken üst göz kapağı iner)
  const baseOpen = calib.openBase || rollingOpen || 0.28;
  const eyelidDrop = Math.max(0, baseOpen - open);

  // Birleşik aşağı bakış skoru (derece cinsinden güçlü dikey tepki)
  const eyeDownDeg = oy * 145 + (lookDown - lookUp) * 44 + eyelidDrop * 165;
  const eyeX = ox * 125 + bsX;
  const eyeY = eyeDownDeg;
  const vergence = irisVerg * 45 + bsVerg;

  return { headX, headY, eyeX, eyeY, eyeDown: eyeDownDeg, open, vergence, blinkScore };
}

// ── Konveks Örtü & 5D Sıkı Alan Hesabı ──
function convexHull(pts) {
  if (!pts || pts.length < 3) return pts ? pts.map((p) => [p[0], p[1]]) : [];
  const p = pts.map((q) => [q[0], q[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
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
    ? { x0: minX - 12, x1: maxX + 12, y0: minY - 12, y1: maxY + 12, hasPoints: true }
    : { x0: -30, x1: 30, y0: -22, y1: 38, hasPoints: false };
}

const hasRegions = () => Object.keys(ALLOWED_REGIONS).some((k) => (calib[k]?.length || 0) >= 3);

/**
 * 5D Odak Uzaklığı:
 * Sadece (x, y) koordinatının poligon içinde olmasına bakmaz!
 * - Nokta bulutu yoğunluğu (kNN): Poligonun boşluklarını ve şişkinliğini reddeder.
 * - Göz Eğimi & Göz Kapağı Uyumu: Ekranın ortasına/altına bakarkenki doğal göz eğiminden fazla aşağı bakılıyorsa (eldeki nesneye bakış) yakalar!
 * - Yakın Odak (Vergence): Göz bebekleri yakındaki bir cisme (eldeki telefon/nesne) odaklandıysa yakalar!
 * - Yasak Bölge (Phone/Hand) Kesimi: Kullanıcı el/telefon bakışını tanıttıysa ekranla çakışsa bile yakalar!
 */
function evaluateFocus(g) {
  const p = [g.x, g.y];
  const tol = settings.tol;
  const sens = settings.handSens || 1.3;

  // 1) Kamera nesne tespiti aktifse
  if (detectedObject && performance.now() < detectedObjUntil) {
    return { d: 1.65, reason: `📱 Kamerada nesne algılandı (${detectedObject})` };
  }

  // 2) En yakın izinli bölgeyi ve o noktadaki beklenen 5D göz profilini bul
  let bestRegionDist = Infinity;
  let nearestSamples = [];

  for (const k in ALLOWED_REGIONS) {
    const pts = calib[k];
    if (!pts || pts.length < 3) continue;
    const hDist = hullDist(p, hulls[k]);

    // Aynı bölgedeki en yakın kalibrasyon örneklerini bul (kNN sıkı zarf kontrolü)
    let minPtDist = Infinity;
    for (let i = 0; i < pts.length; i++) {
      const q = pts[i];
      const sd = Math.hypot(p[0] - q[0], p[1] - q[1]);
      if (sd < minPtDist) minPtDist = sd;
    }
    // Poligon içinde olsa bile gerçek kalibrasyon noktalarından uzaksa (şişkinlik/boşluk) ceza ekle
    const tightDist = Math.max(hDist, Math.max(0, minPtDist - tol * 1.15));
    if (tightDist < bestRegionDist) {
      bestRegionDist = tightDist;
      nearestSamples = pts;
    }
  }

  let score = bestRegionDist / tol;
  let customReason = "Odak alanının dışına baktın";

  // 3) Ekran/Tablet alanı içindeyken ELDEKİ NESNEYE / TELEFONA bakış kontrolü (5D Mismatch)
  if (nearestSamples.length >= 5 && score < 1.25) {
    // Mevcut (x,y) konumuna en yakın 8 kalibrasyon noktasının göz eğimi, açıklık ve vergence ortalamasını al
    const sorted = nearestSamples
      .map((q) => ({ q, d: Math.hypot(p[0] - q[0], p[1] - q[1]) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, 8);

    let expEyeDown = 0, expOpen = 0, expVerg = 0, count5D = 0;
    for (const item of sorted) {
      if (item.q.length >= 5) {
        expEyeDown += item.q[2];
        expOpen += item.q[3];
        expVerg += item.q[4];
        count5D++;
      }
    }

    if (count5D > 0) {
      expEyeDown /= count5D;
      expOpen /= count5D;
      expVerg /= count5D;

      // a) Göz eğimi farkı: Kafa ekrana dönükken gözlerin eldeki nesneye kayması
      const excessEyeDown = Math.max(0, g.eyeDown - expEyeDown - (5.0 / sens));
      // b) Göz kapağı kısılması: Eldeki nesneye bakarken üst göz kapağının düşmesi
      const excessLidDrop = Math.max(0, (expOpen - g.open) - (0.032 / sens)) * 180;
      // c) Yakın odak (Vergence): Ekrana (70cm) değil eldeki cisme (30cm) odaklanma
      const excessVerg    = Math.max(0, (g.verg - expVerg) - (2.4 / sens)) * 2.2;

      const handPenalty = (excessEyeDown * 0.16 + excessLidDrop * 0.14 + excessVerg * 0.18) * sens;
      if (handPenalty > 0.35) {
        score += handPenalty;
        if (score > 1.0) {
          customReason = "👀 Ekran önünde elindeki nesneye / aşağıya odaklandın!";
        }
      }
    }
  }

  // 4) Eğer kullanıcı "🚫 Elde Telefon / Nesne" negatif kalibrasyonunu yaptıysa:
  // 5 boyutlu uzayda (x, y, eyeDown, open, verg) el bakışına ekrandan daha yakınsa anında yakala!
  if (calib.phone && calib.phone.length >= 3) {
    let minPhone5D = Infinity;
    for (let i = 0; i < calib.phone.length; i++) {
      const q = calib.phone[i];
      const d2D = Math.hypot(g.x - q[0], g.y - q[1]);
      const dDown = q.length >= 5 ? Math.abs(g.eyeDown - q[2]) * 0.7 : 0;
      const dOpen = q.length >= 5 ? Math.abs(g.open - q[3]) * 140 : 0;
      const dVerg = q.length >= 5 ? Math.abs(g.verg - q[4]) * 1.5 : 0;
      const d5 = Math.hypot(d2D, dDown, dOpen, dVerg);
      if (d5 < minPhone5D) minPhone5D = d5;
    }

    let minAllow5D = Infinity;
    for (const k in ALLOWED_REGIONS) {
      const pts = calib[k] || [];
      for (let i = 0; i < pts.length; i++) {
        const q = pts[i];
        const d2D = Math.hypot(g.x - q[0], g.y - q[1]);
        const dDown = q.length >= 5 ? Math.abs(g.eyeDown - q[2]) * 0.7 : 0;
        const dOpen = q.length >= 5 ? Math.abs(g.open - q[3]) * 140 : 0;
        const dVerg = q.length >= 5 ? Math.abs(g.verg - q[4]) * 1.5 : 0;
        const d5 = Math.hypot(d2D, dDown, dOpen, dVerg);
        if (d5 < minAllow5D) minAllow5D = d5;
      }
    }

    const phoneHullD = hullDist(p, hulls.phone);
    if (
      (minPhone5D < minAllow5D * 0.95 && minPhone5D < tol * 2.8) ||
      (phoneHullD <= tol * 0.7 && minPhone5D <= minAllow5D * 1.15)
    ) {
      score = Math.max(score, 1.45);
      customReason = "🚫 Elde telefon / nesne bakışı tespit edildi!";
    }
  }

  return { d: score, reason: customReason };
}

/** Uç sıçramaları temizler (%92 en sıkı küme) */
function cleanPoints(pts) {
  if (pts.length < 10) return pts;
  const cx = median(pts.map((p) => p[0])), cy = median(pts.map((p) => p[1]));
  const d = pts.map((p) => Math.hypot(p[0] - cx, p[1] - cy));
  const sortedD = [...d].sort((a, b) => a - b);
  const p90 = sortedD[Math.floor(sortedD.length * 0.90)] || 10;
  const lim = p90 * 1.25 + 2.5;
  return pts.filter((_, i) => d[i] <= lim);
}

function capPoints(pts) {
  if (pts.length <= MAX_POINTS) return pts;
  const step = pts.length / MAX_POINTS;
  return Array.from({ length: MAX_POINTS }, (_, i) => pts[Math.floor(i * step)]);
}

/** Negatif kalibrasyon (phone) eklendiğinde ekranın çakışan alt sızıntı noktalarını otomatik oyup çıkarır */
function carveOutPhoneFromAllowed() {
  if (!calib.phone || calib.phone.length < 5) return;
  for (const k of ["screen", "tablet"]) {
    if (!calib[k] || calib[k].length < 10) continue;
    const filtered = calib[k].filter((pt) => {
      let minP = Infinity;
      for (let i = 0; i < calib.phone.length; i++) {
        const q = calib.phone[i];
        const dist = Math.hypot(pt[0] - q[0], pt[1] - q[1], (pt[2] - q[2]) * 0.5);
        if (dist < minP) minP = dist;
      }
      return minP > settings.tol * 0.85;
    });
    if (filtered.length >= 8) calib[k] = filtered;
  }
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
  if (target === "screen") {
    calib.openBase = median(pts.map((p) => p[3]));
    calib.vergBase = median(pts.map((p) => p[4]));
  }
  if (target === "phone") {
    carveOutPhoneFromAllowed();
  }
  saveCalib();
  awayMs = 0;
  isAway = false;
  setStatus(
    "ok",
    target === "phone"
      ? "🚫 Elde telefon/nesne bakışı yasaklı bölge olarak kaydedildi ✓"
      : `${ALL_REGIONS[target].label} alanı 5D olarak kaydedildi ✓`
  );
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
  // 10 noktalı yoğun ızgara taraması (kenarlar + iç çaprazlar)
  const path = [
    [0.05, 0.06], [0.95, 0.06], [0.95, 0.50], [0.05, 0.50],
    [0.05, 0.92], [0.50, 0.92], [0.95, 0.92], [0.50, 0.50],
    [0.25, 0.25], [0.75, 0.75], [0.50, 0.50],
  ];
  calText.textContent = "Kırmızı noktayı GÖZLERİNLE takip et (normal çalışma duruşunu koru)";
  placeDot(path[0]);
  await sleep(1100);
  calText.textContent = "5D Ekran ve derinlik profili kaydediliyor…";
  sampling = { samples: [] };
  const segs = path.length - 1;
  for (let i = 1; i < path.length; i++) {
    if (cancelCalib) break;
    await moveDot(path[i - 1], path[i], 900, (i - 1) / segs, i / segs);
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
  const isPhone = target === "phone";
  const label = ALL_REGIONS[target].label.toLowerCase();
  ov.classList.remove("hidden");
  for (let i = 3; i > 0; i--) {
    if (cancelCalib) break;
    calText.textContent = isPhone
      ? `${i}… Şimdi elinde telefon veya bir nesne tutuyormuş gibi (ekranın önünde/altında) eline bak!`
      : `${i}… İlk bip sesinden sonra ${label} alanının köşelerine ve ortasına bak.`;
    await sleep(950);
  }
  if (!cancelCalib) {
    beep(660);
    sampling = { samples: [] };
    const steps = isPhone ? 45 : 55;
    for (let i = 0; i < steps; i++) {
      if (cancelCalib) break;
      const rem = Math.ceil(((steps - i) * 100) / 1000);
      calText.textContent = isPhone
        ? `🚫 Elindeki telefona/nesneye bakmaya devam et (sağa sola hafifçe gezdir)… (${rem} sn)`
        : `${ALL_REGIONS[target].label} alanının tamamını gözlerinle tara… (${rem} sn)`;
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
const KEY_LANDMARKS = [33, 133, 362, 263, 159, 145, 386, 374, 10, 151, 9, 168, 6, 127, 356];

function drawOverlay(lm, color) {
  if (document.hidden && !document.pictureInPictureElement) return;
  const vw = video.videoWidth || 640, vh = video.videoHeight || 480;
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
      mctx.strokeStyle = ALL_REGIONS[k].color + "1c";
      mctx.stroke();
    }
    mctx.fillStyle = ALL_REGIONS[k].color + (k === "phone" ? "44" : "30");
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
    mctx.fillText("1) Ekranı kalibre et  →  2) Elde Telefon/Nesne bakışını yasakla", W / 2, H / 2);
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

  fpsFrames++;
  if (now - fpsLastAt >= 1000) {
    const fps = Math.round((fpsFrames * 1000) / (now - fpsLastAt));
    setText("hudBadge", `${fps} FPS${settings.ecoMode ? " • Eko" : " • 5D"}`);
    fpsFrames = 0;
    fpsLastAt = now;
  }

  // Kamera görüntüsünde telefon / nesne tespiti (saniyede ~3-4 kez çalışır, CPU yormaz)
  if (objDetector && !calibrating && now - lastObjCheckAt > 280) {
    lastObjCheckAt = now;
    try {
      const objRes = objDetector.detectForVideo(video, now);
      let found = null;
      if (objRes?.detections) {
        for (const det of objRes.detections) {
          const cat = det.categories?.[0];
          if (cat && BANNED_OBJECTS.has(cat.categoryName) && cat.score >= 0.38) {
            found = cat.categoryName === "cell phone" ? "Telefon 📱" : cat.categoryName;
            break;
          }
        }
      }
      if (found) {
        detectedObject = found;
        detectedObjUntil = now + 900;
      }
    } catch {}
  }
  const objActive = detectedObject && now < detectedObjUntil;
  setText("telObj", objActive ? `🚨 ${detectedObject}` : "Temiz ✓");

  const res = landmarker.detectForVideo(video, now);
  const lm = res.faceLandmarks?.[0];
  const blends = res.faceBlendshapes?.[0]?.categories;

  if (lm && lm.length >= 478) {
    const f = analyze(lm, blends, video.videoWidth || 640, video.videoHeight || 480);

    const isBlinking = f.blinkScore > 0.62;
    if (isBlinking) {
      closedMs += dt;
    } else {
      closedMs = 0;
      rollingOpen = rollingOpen * 0.98 + f.open * 0.02;
    }

    let raw;
    if (isBlinking && closedMs < 380 && lastValidRaw) {
      raw = lastValidRaw;
    } else if (closedMs >= 1500 && lastValidRaw) {
      raw = { ...lastValidRaw, y: lastValidRaw.y + 35, eyeDown: lastValidRaw.eyeDown + 35 };
    } else {
      raw = {
        x: f.headX + f.eyeX,
        y: f.headY + f.eyeY,
        eyeDown: f.eyeDown,
        open: f.open,
        verg: f.vergence,
      };
      lastValidRaw = raw;
    }

    gaze = {
      x: filterX.filter(raw.x, now),
      y: filterY.filter(raw.y, now),
      eyeDown: filterDown.filter(raw.eyeDown, now),
      open: filterOpen.filter(raw.open, now),
      verg: filterVerg.filter(raw.verg, now),
    };

    setText("telEyeDown", `${gaze.eyeDown >= 0 ? "+" : ""}${gaze.eyeDown.toFixed(1)}°`);
    setText("telVerg", `${gaze.verg >= 0 ? "+" : ""}${gaze.verg.toFixed(1)}`);

    if (sampling && !isBlinking) {
      sampling.samples.push([gaze.x, gaze.y, gaze.eyeDown, gaze.open, gaze.verg]);
    }
  } else {
    gaze = null;
    filterX.reset();
    filterY.reset();
    filterDown.reset();
    filterOpen.reset();
    filterVerg.reset();
  }

  drawMap();
  if (calibrating) {
    drawOverlay(lm, "#fcc419");
    return;
  }

  if (!hasRegions()) {
    drawOverlay(lm, "#8f98ad");
    setStatus("idle", lm ? "📺 Başlamak için önce 1. Ekranı kalibre et" : "Yüz aranıyor… Kameraya bak");
    return;
  }

  let away, reason;
  if (gaze) {
    faceLostFired = false;
    const ev = evaluateFocus(gaze);
    const d = ev.d;
    away = isAway ? d > 0.84 : d > 1.0;
    reason = ev.reason;
    setText("distVal", d < 0.15 ? "tam odaklı ✓" : `${d.toFixed(2)}x`);
    setBar("distBar", Math.min(d / 2, 1) * 100, away ? "#ff5c77" : "#38d996");
  } else {
    away = settings.faceLost && !faceLostFired;
    reason = "Yüzün kameradan çıktı";
    setText("distVal", "yüz yok");
  }
  isAway = away;

  // Elde telefon/nesne yakalandığında sayaç 1.35x daha hızlı dolar!
  const speedMult = objActive ? 1.5 : 1.0;
  const delayMs = settings.delay * 1000;
  awayMs = away ? awayMs + dt * speedMult : Math.max(0, awayMs - dt * 1.8);
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
      coolingDown ? "🚨 Yakalandın! İşine odaklan." : gaze ? `⚠️ ${reason}` : "🙈 Yüzün görünmüyor"
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
async function loadModels() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const faceOpts = (delegate) => ({
    baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    minFaceDetectionConfidence: 0.45,
    minFacePresenceConfidence: 0.45,
    minTrackingConfidence: 0.45,
  });
  let face;
  try { face = await FaceLandmarker.createFromOptions(fileset, faceOpts("GPU")); }
  catch { face = await FaceLandmarker.createFromOptions(fileset, faceOpts("CPU")); }

  // Nesne dedektörünü (telefon/kitap/kumanda yakalayıcı) arka planda yükle
  if (!objDetector) {
    ObjectDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: OBJ_MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      scoreThreshold: 0.35,
      maxResults: 3,
    }).then((d) => { objDetector = d; }).catch((e) => console.warn("ObjDetector:", e));
  }
  return face;
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
    setText("loadMsg", "5D yüz, göz derinliği ve nesne motoru hazırlanıyor…");
    const [model, stream] = await Promise.all([
      landmarker ? Promise.resolve(landmarker) : loadModels(),
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
