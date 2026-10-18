import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ───────────────────────── Sabitler ─────────────────────────
const JOB_SITES = [
  { name: "BİM",  url: "https://www.bim.com.tr/categories/676/hizli-basvuru.aspx" },
  { name: "A101", url: "https://www.a101.com.tr/kariyer" },
  { name: "ŞOK",  url: "https://kurumsal.sokmarket.com.tr/tr/is-basvurusu" },
];
const WASM_URL       = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const MAX_POINTS = 360;

const ALLOWED_REGIONS = {
  screen: { label: "Ekran",   color: "#4dabf7" },
  tablet: { label: "Tablet",  color: "#b197fc" },
  extra:  { label: "Ek alan", color: "#ffa94d" },
};
const ALL_REGIONS = {
  ...ALLOWED_REGIONS,
  phone:  { label: "Telefon / El", color: "#ff5c77" },
};

const DEFAULTS = {
  tol: 5.5,
  downGain: 1.4,
  delay: 2.0,
  cooldown: 25,
  vol: 0.35,
  faceLost: true,
  ecoMode: false,
};
const settings = { ...DEFAULTS, ...JSON.parse(localStorage.getItem("kp_settings_v5") || "{}") };
const calib = {
  screen: [], tablet: [], extra: [], phone: [],
  ...JSON.parse(localStorage.getItem("kp_calib_v5") || "{}"),
};
for (const k in ALL_REGIONS) if (!Array.isArray(calib[k])) calib[k] = [];

let hulls = {};
let centers = {};
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

const fX = new OneEuroFilter(1.25, 0.055, 1.0);
const fY = new OneEuroFilter(1.20, 0.055, 1.0);

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
let landmarker = null, audioCtx = null;
let running = false, calibrating = false, cancelCalib = false;
let lastVideoTime = -1, lastTickAt = null;
let gaze = null;                 // { x, y, downDeg }
let lastValidRaw = null, closedMs = 0;
let isAway = false, awayMs = 0, lastTrigger = -Infinity, faceLostFired = false;
let count = 0, focusMs = 0, totalMs = 0;
let sampling = null;             // { samples: [[x, y]] }
let fpsFrames = 0, fpsLastAt = performance.now();

// ───────────────────────── Ayar Arayüzü ─────────────────────────
const getTickMs = () => (settings.ecoMode ? 85 : 42);
const save = () => localStorage.setItem("kp_settings_v5", JSON.stringify(settings));

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
bindRange("downGain", "downGain", (v) => `${v.toFixed(1)}x`);
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
  localStorage.setItem("kp_calib_v5", JSON.stringify(calib));
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

// ───────────────────────── Gerçekçi PCM Market Barkod Okuyucu Sesi ("DIT! ... DI-DIT!") ─────────────────────────
let activeScanSources = [];

function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
}

/**
 * Gerçek bir market kasası lazer barkod okuyucu (Zebra / Datalogic ~2720 Hz piezo buzzer)
 * ses dalgasını 44.1 kHz PCM AudioBuffer olarak doğrudan örnekler:
 * - İlk 3.5 ms piezo kristal "çıt" darbesi (chirp)
 * - Keskin tek/çift harmonikli kare-darbe (pulse) tınısı
 * - 13 ms kasa tezgahı akustik yansıması (slapback)
 */
function createBarcodeBuffer(durSec, freqHz, vol) {
  const sr = audioCtx.sampleRate;
  const totalSec = durSec + 0.025; // kasa akustik yansıma payı
  const len = Math.floor(sr * totalSec);
  const buf = audioCtx.createBuffer(1, len, sr);
  const data = buf.getChannelData(0);

  const mainSamples = Math.floor(sr * durSec);
  const echoDelay = Math.floor(sr * 0.013); // 13ms kasa yansıması
  const amp = Math.min(vol * 0.95, 1.0);

  let phase = 0;
  for (let i = 0; i < mainSamples; i++) {
    const t = i / sr;
    // Piezo kristalin ilk anda yüksek frekanstan oturması (3.5ms chirp)
    const instFreq = t < 0.0035 ? freqHz * (1.0 + 0.09 * (1 - t / 0.0035)) : freqHz;
    phase += (2 * Math.PI * instFreq) / sr;

    // Zebra/Datalogic barkod okuyucu harmonik yapısı (güçlü 1., 3. ve 5. harmonikler + hafif 2. harmonik)
    const raw =
      0.62 * Math.sin(phase) +
      0.16 * Math.sin(2 * phase) +
      0.34 * Math.sin(3 * phase) +
      0.14 * Math.sin(5 * phase);

    // Anında açılan (1.5ms) ve bıçak gibi kesilen (3ms) piezo zarfı
    let env = 1.0;
    if (t < 0.0015) env = t / 0.0015;
    else if (t > durSec - 0.003) env = Math.max(0, (durSec - t) / 0.003);

    data[i] = raw * env * amp;
  }

  // Kasa tezgahı kısa akustik yansıması (gerçek mekan hissi verir, "dandik" bilgisayar bip'i gibi duyulmaz)
  for (let i = 0; i < mainSamples; i++) {
    if (i + echoDelay < len) {
      data[i + echoDelay] += data[i] * 0.22;
    }
  }
  return buf;
}

/**
 * Ceza anında kasiyerin 3 ürünü rastgele hızlarla barkod okuyucudan geçirmesi ("Dıt! ... Dı-dıt!").
 * Bazen bir barkod sesinin bitmesine çok az kala diğeri üstüne biner.
 */
function chime() {
  ensureAudio();
  if (settings.vol <= 0.001) return;

  // Önceki çalan ses varsa durdur (sonraki cezayla asla karışmaz)
  for (const s of activeScanSources) {
    try { s.stop(); s.disconnect(); } catch {}
  }
  activeScanSources = [];

  const now = audioCtx.currentTime + 0.005;
  let offset = 0;
  const scanCount = 3; // 3 kez barkod okutma

  for (let i = 0; i < scanCount; i++) {
    // Her okutma 58ms - 82ms arası keskin market "DIT!" sesi
    const dur = 0.058 + Math.random() * 0.024;
    const freq = 2715 + (Math.random() - 0.5) * 30;
    const buf = createBarcodeBuffer(dur, freq, settings.vol);

    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(audioCtx.destination);
    src.start(now + offset);
    activeScanSources.push(src);

    // Bir sonraki okutmanın ne zaman başlayacağı tamamen rastgele:
    // %45 ihtimalle mevcut sesin bitmesine çok kısa süre kala (8ms - 22ms kala) öteki başlar ("dı-dıt!")
    // %55 ihtimalle 35ms - 260ms arası rastgele kasiyer boşluğu bırakır
    if (Math.random() < 0.45) {
      const earlyOverlap = 0.008 + Math.random() * 0.014;
      offset += Math.max(0.038, dur - earlyOverlap);
    } else {
      const randomGap = 0.035 + Math.random() * 0.23;
      offset += dur + randomGap;
    }
  }
}

function beep(freq = 880) {
  ensureAudio();
  if (settings.vol <= 0.001) return;
  const buf = createBarcodeBuffer(0.07, freq === 880 ? 2400 : freq * 2.5, Math.max(settings.vol, 0.2));
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);
  src.start();
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

// ───────────────────────── Geometri & Keskin Bakış Analizi ─────────────────────────
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
 * Yüzden 2D (x, y) bakış açısını çıkarır.
 * - Sadece ÜST YÜZ (şakaklar, göz köşeleri, burun köprüsü, alın) ve GÖZLER kullanılır (el çenedeyken bozulmaz).
 * - Hiçbir kalibrasyon değişkenine bağlı değildir (kalibrasyon öncesi ve sonrası koordinatlar %100 aynıdır, asla kaymaz!).
 * - Telefona / eldeki nesneye bakarken oluşan aşağı göz eğimi (eyeLookDown), göz kapağı kapanması ve yakın odak (eyeLookIn)
 *   doğrudan dikey bakış açısını (+y) aşağı doğru güçlendirir; böylece telefona baktığında haritadaki nokta
 *   mavi ekran kutusunun dışına net biçimde çıkar!
 */
function analyze(lm, blendCategories, W, H) {
  const P = (i) => ({ x: lm[i].x * W, y: lm[i].y * H, z: lm[i].z * W });

  // 1) Şakak (127, 356) ve Burun Köprüsü (6, 168) + Üst-Yüz Normali ile Kafa Açısı
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

  const pitch2D = ((noseBridge.y - templeMid.y) / faceW) * 115;
  const yaw2D   = ((noseBridge.x - templeMid.x) / faceW) * 115;

  let n = cross(sub(rightEyeMid, leftEyeMid), sub(forehead, noseBridge));
  const nl = Math.hypot(n.x, n.y, n.z) || 1;
  n = { x: n.x / nl, y: n.y / nl, z: n.z / nl };
  if (n.z > 0) n = { x: -n.x, y: -n.y, z: -n.z };
  const yaw3D   = Math.atan2(n.x, -n.z) * RAD;
  const pitch3D = Math.asin(clamp(n.y, -1, 1)) * RAD;

  const headX = yaw2D * 0.55 + yaw3D * 0.45;
  const headY = pitch2D * 0.60 + pitch3D * 0.40;

  // 2) İris konumu ve Göz Açıklığı
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
    ox += ((d.x * u.x + d.y * u.y) / w) * 0.5;
    oy += ((d.x * v.x + d.y * v.y) / w) * 0.5;
    open += (len2(sub(P(e.up), P(e.lo))) / w) * 0.5;
  }

  // 3) Blendshape Göz Bakışı
  let lookDown = 0, lookUp = 0, bsX = 0, lookInBoth = 0, blinkScore = 0;
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
    lookInBoth = Math.min(bs.eyeLookInLeft || 0, bs.eyeLookInRight || 0);

    bsX = (lookRightImg - lookLeftImg) * 24;
    blinkScore = ((bs.eyeBlinkLeft || 0) + (bs.eyeBlinkRight || 0)) * 0.5;
  }

  const dGain = settings.downGain || 1.4;
  // Aşağı / Eldeki telefona bakış güçlendirici (sabit fiziksel eşiklerle çalışır, kalibrasyonu asla bozmaz!)
  const extraDownBoost = Math.pow(Math.max(0, lookDown - 0.16), 1.3) * 75 * dGain;
  const nearHandBoost  = Math.max(0, lookInBoth - 0.12) * 28 * dGain;

  const eyeX = ox * 125 + bsX;
  const eyeY = (oy * 135 + (lookDown - lookUp) * 42) * dGain + extraDownBoost + nearHandBoost;

  return {
    x: headX + eyeX,
    y: headY + eyeY,
    downDeg: eyeY,
    open,
    blinkScore,
  };
}

// ── Konveks Örtü & Harita Geometrisi ──
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
  centers = {};
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, countPts = 0;
  for (const k in ALL_REGIONS) {
    const pts = calib[k] || [];
    hulls[k] = convexHull(pts);
    if (pts.length) {
      centers[k] = [median(pts.map((p) => p[0])), median(pts.map((p) => p[1]))];
    } else {
      centers[k] = null;
    }
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
 * Net, Şeffaf ve Hatasız Odak Hesabı:
 * Haritadaki nokta izinli mavi/mor/turuncu bölgenin içindeyse her zaman 0 (Odaklı) döner!
 * Eğer kullanıcı kırmızı "Telefon/El" bölgesini işaretlediyse ve nokta o kırmızı bölgenin içine
 * (ekran merkezinden uzağa) girdiyse anında yakalar!
 */
function evaluateFocus(g) {
  const p = [g.x, g.y];
  const tol = settings.tol;

  // 1) Eğer kırmızı "Telefon / El" yasak alanı tanımlıysa ve bakış noktası doğrudan kırmızı alana girdiyse:
  if (hulls.phone && hulls.phone.length >= 3 && centers.phone) {
    const phoneDist = hullDist(p, hulls.phone);
    const dToPhoneCenter = Math.hypot(p[0] - centers.phone[0], p[1] - centers.phone[1]);
    const dToScreenCenter = centers.screen
      ? Math.hypot(p[0] - centers.screen[0], p[1] - centers.screen[1])
      : Infinity;

    // Nokta kırmızı yasak bölgenin içindeyse VE ekranın merkezinden çok telefonun merkezine yakınsa:
    if (phoneDist <= 1.5 && dToPhoneCenter < dToScreenCenter * 0.92) {
      return { d: 1.5, reason: "🚫 Elindeki telefona / yasaklı alana baktın" };
    }
  }

  // 2) İzinli bölgelere (Ekran, Tablet, Ek Alan) olan uzaklık
  let best = Infinity;
  for (const k in ALLOWED_REGIONS) {
    const d = hullDist(p, hulls[k]);
    if (d < best) best = d;
  }
  return { d: best / tol, reason: "Odak alanının dışına (telefona/etrafa) baktın" };
}

/** Kalibrasyon sırasındaki uç sıçramaları temizler */
function cleanPoints(pts) {
  if (pts.length < 10) return pts;
  const cx = median(pts.map((p) => p[0])), cy = median(pts.map((p) => p[1]));
  const d = pts.map((p) => Math.hypot(p[0] - cx, p[1] - cy));
  const sortedD = [...d].sort((a, b) => a - b);
  const p90 = sortedD[Math.floor(sortedD.length * 0.90)] || 10;
  const lim = p90 * 1.2 + 2;
  return pts.filter((_, i) => d[i] <= lim);
}

function capPoints(pts) {
  if (pts.length <= MAX_POINTS) return pts;
  const step = pts.length / MAX_POINTS;
  return Array.from({ length: MAX_POINTS }, (_, i) => pts[Math.floor(i * step)]);
}

// ───────────────────────── Kalibrasyon ─────────────────────────
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const ov = $("calOverlay"), dot = $("dot"), calText = $("calText"), calProg = $("calProgress"), calBadge = $("calPhaseBadge");

$("calCancel").addEventListener("click", () => { cancelCalib = true; });
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && calibrating) cancelCalib = true;
});

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
  setStatus(
    "ok",
    target === "phone"
      ? "🚫 Telefon / El noktası kırmızı yasak bölge olarak haritaya eklendi ✓"
      : `✅ ${ALL_REGIONS[target].label} alanı başarıyla kaydedildi ✓`
  );
  drawMap(true);
}

async function calibrateScreen() {
  calibrating = true;
  cancelCalib = false;
  calProg.style.width = "0%";
  calBadge.textContent = "EKRAN KALİBRASYONU";
  calBadge.classList.remove("danger");
  ov.classList.remove("hidden");
  dot.classList.remove("hidden");
  try { await document.documentElement.requestFullscreen(); } catch {}
  await sleep(400);

  const path = [
    [0.05, 0.06], [0.95, 0.06], [0.95, 0.50], [0.05, 0.50],
    [0.05, 0.90], [0.50, 0.90], [0.95, 0.90], [0.50, 0.06], [0.50, 0.50],
  ];
  calText.textContent = "Kırmızı noktayı GÖZLERİNLE takip et (normal oturuşunu koru)";
  placeDot(path[0]);
  await sleep(1100);
  calText.textContent = "Ekran alanı kaydediliyor…";
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
  calBadge.textContent = isPhone ? "YASAKLI TELEFON / EL NOKTASI" : `${ALL_REGIONS[target].label.toUpperCase()} KALİBRASYONU`;
  calBadge.classList.toggle("danger", isPhone);
  const label = ALL_REGIONS[target].label.toLowerCase();
  ov.classList.remove("hidden");

  for (let i = 3; i > 0; i--) {
    if (cancelCalib) break;
    calText.textContent = isPhone
      ? `${i}… Şimdi elinde telefon tuttuğun noktaya bak (ekrana bakma, sadece telefona bak)!`
      : `${i}… İlk bip sesinden sonra ${label} alanının köşelerine ve ortasına bak.`;
    await sleep(900);
  }
  if (!cancelCalib) {
    beep(880);
    sampling = { samples: [] };
    const steps = isPhone ? 32 : 55;
    for (let i = 0; i < steps; i++) {
      if (cancelCalib) break;
      const rem = Math.ceil(((steps - i) * 100) / 1000);
      calText.textContent = isPhone
        ? `🚫 Sadece elindeki telefona bakmaya devam et… (${rem} sn)`
        : `${ALL_REGIONS[target].label} alanının tamamını gözlerinle tara… (${rem} sn)`;
      calProg.style.width = `${((i + 1) / steps) * 100}%`;
      await sleep(100);
    }
    beep(1100);
  }
  ov.classList.add("hidden");
  finishSampling(target, target === "extra");
  calibrating = false;
}

// ───────────────────────── Hızlı Çizim ─────────────────────────
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
      mctx.strokeStyle = ALL_REGIONS[k].color + "20";
      mctx.stroke();
    }
    mctx.fillStyle = ALL_REGIONS[k].color + (k === "phone" ? "48" : "30");
    mctx.fill();
    mctx.lineWidth = 2;
    mctx.strokeStyle = ALL_REGIONS[k].color;
    mctx.stroke();
  }

  if (gaze) {
    const [x, y] = T([gaze.x, gaze.y]);
    mctx.beginPath();
    mctx.arc(x, y, 7.5, 0, Math.PI * 2);
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
    mctx.fillText("Başlamak için “📺 Ekranı kalibre et” butonuna bas", W / 2, H / 2);
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
    setText("hudBadge", `${fps} FPS${settings.ecoMode ? " • Eko" : ""}`);
    fpsFrames = 0;
    fpsLastAt = now;
  }

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
    }

    let raw;
    if (isBlinking && closedMs < 400 && lastValidRaw) {
      raw = lastValidRaw;
    } else if (closedMs >= 1600 && lastValidRaw) {
      raw = { ...lastValidRaw, y: lastValidRaw.y + 35 };
    } else {
      raw = f;
      lastValidRaw = raw;
    }

    gaze = {
      x: fX.filter(raw.x, now),
      y: fY.filter(raw.y, now),
      downDeg: raw.downDeg,
    };

    setText("telGaze", `${(-gaze.x).toFixed(0)}° / ${gaze.y.toFixed(0)}°`);
    setText("telEyeDown", `${gaze.downDeg >= 0 ? "+" : ""}${gaze.downDeg.toFixed(1)}°`);

    if (sampling && !isBlinking) {
      sampling.samples.push([gaze.x, gaze.y]);
    }
  } else {
    gaze = null;
    fX.reset();
    fY.reset();
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
    const ev = evaluateFocus(gaze);
    const d = ev.d;
    away = isAway ? d > 0.84 : d > 1.0;
    reason = ev.reason;
    setText("distVal", d === 0 ? "alanın içinde ✓" : `${d.toFixed(2)}x`);
    setBar("distBar", Math.min(d / 2, 1) * 100, away ? "#ff5c77" : "#38d996");
    setText("telState", away ? "⚠️ Alan Dışı" : "✅ Odaklı");
  } else {
    away = settings.faceLost && !faceLostFired;
    reason = "Yüzün kameradan çıktı";
    setText("distVal", "yüz yok");
    setText("telState", "🙈 Yüz Yok");
  }
  isAway = away;

  const delayMs = settings.delay * 1000;
  awayMs = away ? awayMs + dt : Math.max(0, awayMs - dt * 2.0);
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
      coolingDown ? "🚨 Yakalandın! İşine odaklan." : gaze ? `👀 ${reason}` : "🙈 Yüzün görünmüyor"
    );
    drawOverlay(lm, "#ff5c77");
  } else {
    focusMs += dt;
    setStatus("ok", gaze ? "✅ Odaklısın, kasiyerlikten uzaklaşıyorsun!" : "Yüz yok (yakalama kapalı)");
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
async function loadModel() {
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
  try { return await FaceLandmarker.createFromOptions(fileset, faceOpts("GPU")); }
  catch { return await FaceLandmarker.createFromOptions(fileset, faceOpts("CPU")); }
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
