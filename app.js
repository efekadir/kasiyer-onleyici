import {
  FaceLandmarker,
  HandLandmarker,
  ObjectDetector,
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
const HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const OBJ_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite";

// Kamerada elde tutulunca yakalanacak tüm nesneler (person ve chair/couch hariç her türlü elde tutulabilir nesne)
const IGNORED_OBJECTS = new Set(["person", "chair", "couch", "bed", "dining table", "tv", "potted plant"]);

const MAX_POINTS = 360;

const ALLOWED_REGIONS = {
  screen: { label: "Ekran",   color: "#4dabf7" },
  tablet: { label: "Tablet",  color: "#b197fc" },
  extra:  { label: "Ek alan", color: "#ffa94d" },
};
const ALL_REGIONS = {
  ...ALLOWED_REGIONS,
  phone:  { label: "Elde Nesne/Telefon", color: "#ff5c77" },
};

// v4.0 yeni depolama anahtarı (eski hatalı kalibrasyonları otomatik temizler)
const DEFAULTS = {
  tol: 3.5,
  handSens: 1.6,
  delay: 2.0,
  cooldown: 25,
  vol: 0.25,
  faceLost: true,
  handDetect: true,
  ecoMode: false,
};
const settings = { ...DEFAULTS, ...JSON.parse(localStorage.getItem("kp_settings_v4") || "{}") };
const calib = {
  screen: [], tablet: [], extra: [], phone: [],
  ...JSON.parse(localStorage.getItem("kp_calib_v4") || "{}"),
};
for (const k in ALL_REGIONS) if (!Array.isArray(calib[k])) calib[k] = [];

let hulls = {};
let calibBounds = { x0: -28, x1: 28, y0: -20, y1: 32, hasPoints: false };

// ───────────────────────── 1-Euro Adaptif Filtre ─────────────────────────
class OneEuroFilter {
  constructor(minCutoff = 1.3, beta = 0.055, dCutoff = 1.0) {
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

const fX    = new OneEuroFilter(1.3, 0.06, 1.0);
const fY    = new OneEuroFilter(1.25, 0.06, 1.0);
const fHeadY = new OneEuroFilter(1.2, 0.05, 1.0);
const fDown = new OneEuroFilter(1.2, 0.05, 1.0);
const fOpen = new OneEuroFilter(1.1, 0.04, 1.0);
const fVerg = new OneEuroFilter(1.0, 0.03, 1.0);

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
let landmarker = null, handLandmarker = null, objDetector = null, audioCtx = null;
let running = false, calibrating = false, cancelCalib = false;
let lastVideoTime = -1, lastTickAt = null, lastAuxCheckAt = 0;
let activeThreat = null, activeThreatUntil = 0, activeBoxes = [];
let gaze = null;                 // { x, y, headY, lookDown, open, verg }
let lastValidRaw = null, closedMs = 0;
let isAway = false, awayMs = 0, lastTrigger = -Infinity, faceLostFired = false;
let count = 0, focusMs = 0, totalMs = 0;
let sampling = null;             // { samples: [[x, y, headY, lookDown, open, verg]] }
let fpsFrames = 0, fpsLastAt = performance.now();

// ───────────────────────── Ayar Arayüzü ─────────────────────────
const getTickMs = () => (settings.ecoMode ? 85 : 42);
const save = () => localStorage.setItem("kp_settings_v4", JSON.stringify(settings));

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

$("handDetect").checked = settings.handDetect !== false;
$("handDetect").addEventListener("change", (e) => { settings.handDetect = e.target.checked; save(); });

$("ecoMode").checked = !!settings.ecoMode;
$("ecoMode").addEventListener("change", (e) => {
  settings.ecoMode = e.target.checked;
  save();
  if (running) ticker.postMessage(getTickMs());
});

function saveCalib() {
  localStorage.setItem("kp_calib_v4", JSON.stringify(calib));
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

// ───────────────────────── Ses (Market Kasiyer Barkod Okuyucu "Dıt-Dıt") ─────────────────────────
let scanNodes = [];
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
  g.gain.linearRampToValueAtTime(vol, start + 0.015);
  g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
  o.connect(g).connect(audioCtx.destination);
  o.start(start);
  o.stop(start + dur + 0.04);
}
/** Tek bir gerçekçi süpermarket barkod okuyucu (piezo buzzer ~2580 Hz) "DIT!" sesi */
function barcodeBeep(start, dur, vol) {
  if (vol <= 0.001) return;
  const freq = 2580 + (Math.random() - 0.5) * 35; // gerçekçi mikro frekans farkı

  // Piezo tınısı: kare dalga + rezonant bant geçiren filtre + saf sinüs gövde
  const oscSq = audioCtx.createOscillator();
  const oscSin = audioCtx.createOscillator();
  const filter = audioCtx.createBiquadFilter();
  const gain = audioCtx.createGain();

  oscSq.type = "square";
  oscSq.frequency.setValueAtTime(freq, start);
  oscSin.type = "sine";
  oscSin.frequency.setValueAtTime(freq, start);

  filter.type = "bandpass";
  filter.frequency.setValueAtTime(freq, start);
  filter.Q.setValueAtTime(5.5, start);

  const sqGain = audioCtx.createGain();
  sqGain.gain.value = 0.45;
  const sinGain = audioCtx.createGain();
  sinGain.gain.value = 0.75;

  // Barkod okuyucu zarfı: anında keskin başlar, düz devam eder, aniden kesilir
  const peak = Math.min(vol * 1.15, 1.0);
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.linearRampToValueAtTime(peak, start + 0.003);
  gain.gain.setValueAtTime(peak, start + Math.max(0.01, dur - 0.006));
  gain.gain.exponentialRampToValueAtTime(0.0001, start + dur);

  oscSq.connect(filter).connect(sqGain).connect(gain);
  oscSin.connect(sinGain).connect(gain);
  gain.connect(audioCtx.destination);

  oscSq.start(start);
  oscSin.start(start);
  oscSq.stop(start + dur + 0.01);
  oscSin.stop(start + dur + 0.01);
  scanNodes.push(oscSq, oscSin, gain);
}

/**
 * Ceza sesi: Kasiyerin üst üste 3 ürünü rastgele aralıklarla barkod okuyucudan geçirmesi ("Dıt! ... Dı-dıt!").
 * Bazen bir "dıt" bitmesine çok az kala diğeri üstüne biner, toplam ~0.5-1.0 sn içinde biter.
 */
function chime() {
  ensureAudio();
  // Önceki çalan barkod sesleri varsa temizle (birbirine karışmasın)
  for (const n of scanNodes) {
    try { n.stop?.(); n.disconnect?.(); } catch {}
  }
  scanNodes = [];

  let cursor = audioCtx.currentTime + 0.01;
  const scanCount = 3;

  for (let i = 0; i < scanCount; i++) {
    const dur = 0.075 + Math.random() * 0.032; // 75ms - 107ms arası net barkod "dıt" süresi
    barcodeBeep(cursor, dur, settings.vol);

    // Sonraki barkod okutma zamanı: tamamen rastgele!
    // %40 ihtimalle mevcut sesin bitmesine çok az kala (10-25ms kala) üstüne binerek başlar ("dı-dıt!"),
    // %60 ihtimalle rastgele 45ms - 340ms arası kasiyer el çabukluğu boşluğu bırakır.
    if (Math.random() < 0.42) {
      const overlap = 0.010 + Math.random() * 0.016;
      cursor += Math.max(0.045, dur - overlap);
    } else {
      const gap = 0.045 + Math.random() * 0.30;
      cursor += dur + gap;
    }
  }
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

// ───────────────────────── 6D Saf Fiziksel Analiz (Kalibrasyondan Bağımsız!) ─────────────────────────
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
 * ÖNEMLİ: Bu fonksiyon hiçbir kalibrasyon değişkenine (calib.openBase vb.) bağlı DEĞİLDİR!
 * Böylece kalibrasyon sırasında kaydedilen koordinat sistemi ile canlı takip %100 aynı kalır.
 */
function analyze(lm, blendCategories, W, H) {
  const P = (i) => ({ x: lm[i].x * W, y: lm[i].y * H, z: lm[i].z * W });

  // 1) Şakak (127, 356) ve Göz/Burun Köprüsü (33, 133, 263, 362, 6, 168) hibrit kafa açısı
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

  const pitch2D = ((noseBridge.y - templeMid.y) / faceW) * 120;
  const yaw2D   = ((noseBridge.x - templeMid.x) / faceW) * 120;

  let n = cross(sub(rightEyeMid, leftEyeMid), sub(forehead, noseBridge));
  const nl = Math.hypot(n.x, n.y, n.z) || 1;
  n = { x: n.x / nl, y: n.y / nl, z: n.z / nl };
  if (n.z > 0) n = { x: -n.x, y: -n.y, z: -n.z };
  const yaw3D   = Math.atan2(n.x, -n.z) * RAD;
  const pitch3D = Math.asin(clamp(n.y, -1, 1)) * RAD;

  const headX = yaw2D * 0.6 + yaw3D * 0.4;
  const headY = pitch2D * 0.65 + pitch3D * 0.35;

  // 2) Saf İris konumu, Göz açıklığı (open) ve Yakınsama (Vergence)
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

    const distToInner = len2(sub(irisPt, P(e.inner))) / w;
    const distToOuter = len2(sub(irisPt, P(e.outer))) / w;
    irisVerg += (distToOuter - distToInner) * 0.5;
  }

  // 3) Saf Blendshape değerleri
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
    bsVerg = (lookInBoth - lookOutBoth) * 15;
    blinkScore = ((bs.eyeBlinkLeft || 0) + (bs.eyeBlinkRight || 0)) * 0.5;
  }

  // Saf dikey göz açısı (kalibrasyona bağımlı hiçbir sabit içermez!)
  const netLookDown = (lookDown - lookUp) * 42 + oy * 140;
  const eyeX = ox * 125 + bsX;
  const eyeY = netLookDown;
  const vergence = irisVerg * 45 + bsVerg;

  return {
    x: headX + eyeX,
    y: headY + eyeY,
    headY,
    lookDown: netLookDown,
    open,
    verg: vergence,
    blinkScore,
  };
}

// ───────────────────────── El & Nesne Görsel Tespiti (HandLandmarker + ObjectDetector) ─────────────────────────
function detectHandsAndObjects(faceLm, now) {
  if (!settings.handDetect || calibrating) {
    activeThreat = null;
    activeBoxes = [];
    return;
  }
  const boxes = [];
  let foundThreat = null;

  // 1) ObjectDetector: Kamerada telefon, kitap, kumanda, bardak, oyuncak vb. herhangi bir nesne var mı?
  if (objDetector) {
    try {
      const objRes = objDetector.detectForVideo(video, now);
      if (objRes?.detections) {
        const vw = video.videoWidth || 640, vh = video.videoHeight || 480;
        for (const det of objRes.detections) {
          const cat = det.categories?.[0];
          if (!cat) continue;
          const name = cat.categoryName.toLowerCase();
          if (IGNORED_OBJECTS.has(name)) continue;
          const bb = det.boundingBox;
          // Yüzün kendisini nesne sanmasın
          if (bb && bb.width < vw * 0.75 && cat.score >= 0.32) {
            const labelTr = name === "cell phone" ? "Telefon 📱" : `Nesne (${name})`;
            foundThreat = `Kamerada ${labelTr} tutulduğu algılandı`;
            boxes.push({
              x: bb.originX / vw,
              y: bb.originY / vh,
              w: bb.width / vw,
              h: bb.height / vh,
              label: labelTr,
            });
          }
        }
      }
    } catch {}
  }

  // 2) HandLandmarker: Kamerada el var mı? (Çenede sabit duran el hariç elde bir şey tutma / ele bakma)
  if (handLandmarker) {
    try {
      const handRes = handLandmarker.detectForVideo(video, now);
      const hands = handRes?.landmarks || [];
      if (hands.length > 0 && faceLm) {
        const chin = faceLm[152];
        const jawL = faceLm[234], jawR = faceLm[454];
        const faceH = Math.hypot(chin.x - faceLm[10].x, chin.y - faceLm[10].y) || 0.25;

        // İki el birden ekrandaysa kesinlikle bir şeyle (telefon/nesne) uğraşıyordur
        if (hands.length >= 2) {
          foundThreat = "İki elinle bir nesne/telefon tutuyorsun 🖐️";
        }

        for (const h of hands) {
          let minX = 1, maxX = 0, minY = 1, maxY = 0;
          for (const pt of h) {
            if (pt.x < minX) minX = pt.x;
            if (pt.x > maxX) maxX = pt.x;
            if (pt.y < minY) minY = pt.y;
            if (pt.y > maxY) maxY = pt.y;
          }
          const hCenter = { x: (minX + maxX) * 0.5, y: (minY + maxY) * 0.5 };
          const indexTip = h[8], thumbTip = h[4], wrist = h[0];

          // El çeneye / yanağa mı dayalı?
          const distToChin = Math.hypot(hCenter.x - chin.x, hCenter.y - chin.y);
          const distTipToChin = Math.hypot(indexTip.x - chin.x, indexTip.y - chin.y);
          const distToJawL = Math.hypot(hCenter.x - jawL.x, hCenter.y - jawL.y);
          const distToJawR = Math.hypot(hCenter.x - jawR.x, hCenter.y - jawR.y);
          const minFaceDist = Math.min(distToChin, distTipToChin, distToJawL, distToJawR);

          // Çeneden uzakta (göğüs/ekran önünde) tutulan el VEYA başparmak-işaret parmağı nesne kavrama pozisyonu
          const isRestingOnChin = minFaceDist < faceH * 0.58 && wrist.y > chin.y - 0.05;

          if (!isRestingOnChin) {
            foundThreat = foundThreat || "Ekran önünde elinde nesne/telefon tutuyorsun 🖐️";
            boxes.push({ x: minX, y: minY, w: maxX - minX, h: maxY - minY, label: "El / Nesne 🖐️" });
          } else if (gaze && calib.screen?.length >= 5) {
            // El çenede olsa bile gözler aşağıya (ele doğru) kaydıysa yakala!
            const screenOpenMed = median(calib.screen.map((p) => p[4]));
            if (gaze.open < screenOpenMed - 0.032) {
              foundThreat = "Eline / elindeki nesneye bakıyorsun 👀";
              boxes.push({ x: minX, y: minY, w: maxX - minX, h: maxY - minY, label: "Ele Bakış 👀" });
            }
          }
        }
      }
    } catch {}
  }

  activeBoxes = boxes;
  if (foundThreat) {
    activeThreat = foundThreat;
    activeThreatUntil = now + 750;
  }
}

// ── Konveks Örtü & 6D Sınıflandırıcı ──
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
    ? { x0: minX - 10, x1: maxX + 10, y0: minY - 10, y1: maxY + 10, hasPoints: true }
    : { x0: -28, x1: 28, y0: -20, y1: 32, hasPoints: false };
}

const hasRegions = () => Object.keys(ALLOWED_REGIONS).some((k) => (calib[k]?.length || 0) >= 3);

/**
 * 6 Boyutlu Mesafe (x, y, headY, lookDown, open, verg)
 * Ekran ile Eldeki Nesne/Telefon aynı (x, y) doğrultusunda olsa bile:
 * - Göz kapağı açıklığı (open),
 * - Göz küresi aşağı eğimi (lookDown),
 * - Kafa eğimi (headY) ve
 * - Göz bebeği yakınsaması (verg)
 * üzerinden iki durumu birbirinden kesin olarak ayırır!
 */
function dist6D(g, q, sens) {
  const dx = g.x - q[0];
  const dy = (g.y - q[1]) * 1.25;
  if (q.length < 6) return Math.hypot(dx, dy);
  const dHeadY = (g.headY - q[2]) * 0.9 * sens;
  const dLook  = (g.lookDown - q[3]) * 1.35 * sens;
  const dOpen  = (g.open - q[4]) * 260 * sens;
  const dVerg  = (g.verg - q[5]) * 2.1 * sens;
  return Math.hypot(dx, dy, dHeadY, dLook, dOpen, dVerg);
}

function evaluateFocus(g, now) {
  const p = [g.x, g.y];
  const tol = settings.tol;
  const sens = settings.handSens || 1.6;

  // 1) Kamerada el / nesne / telefon yakalandıysa anında ceza!
  if (activeThreat && now < activeThreatUntil) {
    return { d: 1.75, reason: activeThreat };
  }

  // 2) İzinli bölgelere olan 2D + 6D mesafeyi hesapla
  let best2DDist = Infinity;
  let bestAllowed6D = Infinity;
  let localNeighbors = [];

  for (const k in ALLOWED_REGIONS) {
    const pts = calib[k];
    if (!pts || pts.length < 3) continue;
    const hDist = hullDist(p, hulls[k]);

    let minPt2D = Infinity;
    for (let i = 0; i < pts.length; i++) {
      const q = pts[i];
      const d2 = Math.hypot(p[0] - q[0], p[1] - q[1]);
      if (d2 < minPt2D) minPt2D = d2;
      const d6 = dist6D(g, q, sens);
      if (d6 < bestAllowed6D) bestAllowed6D = d6;
    }

    // Konveks örtü içinde olsa bile gerçek kalibrasyon noktalarından uzaksa (şişkinlik) izin verme
    const effective2D = Math.max(hDist, Math.max(0, minPt2D - tol * 1.05));
    if (effective2D < best2DDist) {
      best2DDist = effective2D;
      localNeighbors = pts;
    }
  }

  let score = best2DDist / tol;
  let customReason = "Odak alanının dışına baktın";

  // 3) Eğer kullanıcı "Elde Nesne/Telefon" kalibrasyonunu yaptıysa (2. Aşama):
  // 6 boyutlu uzayda el/telefon profiline ekran profilinden daha yakınsa (veya telefon bölgesindeyse) ANINDA YAKALA!
  if (calib.phone && calib.phone.length >= 3) {
    let bestPhone6D = Infinity;
    let bestPhone2D = Infinity;
    for (let i = 0; i < calib.phone.length; i++) {
      const q = calib.phone[i];
      const d2 = Math.hypot(p[0] - q[0], p[1] - q[1]);
      if (d2 < bestPhone2D) bestPhone2D = d2;
      const d6 = dist6D(g, q, sens);
      if (d6 < bestPhone6D) bestPhone6D = d6;
    }
    const phoneHullD = hullDist(p, hulls.phone);

    // 6D sınıflandırıcı: Ekranın tam önünde olsa bile göz/kapak/derinlik el profiline daha yakınsa yakala!
    const ratio = bestPhone6D / Math.max(bestAllowed6D, 0.1);
    if (ratio < 0.96 * (sens / 1.4) || (phoneHullD <= tol * 0.6 && ratio < 1.12)) {
      const phoneScore = Math.max(1.45, (1.15 / Math.max(ratio, 0.3)));
      if (phoneScore > score) {
        score = phoneScore;
        customReason = "🚨 Ekran önünde elindeki telefona / nesneye baktın!";
      }
    }
  }

  // 4) Ekran içindeyken yerel göz kapağı / göz eğimi / yakın odak sapması kontrolü
  if (localNeighbors.length >= 5) {
    const closest = localNeighbors
      .map((q) => ({ q, d: Math.hypot(p[0] - q[0], p[1] - q[1]) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, 6);

    let expHeadY = 0, expLookDown = 0, expOpen = 0, expVerg = 0, cnt = 0;
    for (const item of closest) {
      if (item.q.length >= 6) {
        expHeadY    += item.q[2];
        expLookDown += item.q[3];
        expOpen     += item.q[4];
        expVerg     += item.q[5];
        cnt++;
      }
    }
    if (cnt > 0) {
      expHeadY /= cnt;
      expLookDown /= cnt;
      expOpen /= cnt;
      expVerg /= cnt;

      // O ekran noktasında olması gerekenden daha kısık göz kapağı, daha aşağı göz eğimi veya daha yüksek yakın odak
      const lidDropExcess  = Math.max(0, (expOpen - g.open) - (0.022 / sens)) * 220;
      const lookDownExcess = Math.max(0, (g.lookDown - expLookDown) - (3.2 / sens)) * 1.1;
      const vergExcess     = Math.max(0, (g.verg - expVerg) - (1.8 / sens)) * 2.5;

      const mismatchPenalty = ((lidDropExcess + lookDownExcess + vergExcess) / tol) * 0.45 * sens;
      if (mismatchPenalty > 0.25) {
        score += mismatchPenalty;
        if (score > 1.0) {
          customReason = "👀 Gözlerin ekran yerine elindeki nesneye odaklandı!";
        }
      }
    }
  }

  return { d: score, reason: customReason };
}

/** Kalibrasyon sırasındaki uç sıçramaları (%86 en temiz merkez) kırpar */
function cleanPoints(pts) {
  if (pts.length < 10) return pts;
  const cx = median(pts.map((p) => p[0])), cy = median(pts.map((p) => p[1]));
  const d = pts.map((p) => Math.hypot(p[0] - cx, p[1] - cy));
  const sortedD = [...d].sort((a, b) => a - b);
  const p86 = sortedD[Math.floor(sortedD.length * 0.86)] || 8;
  const lim = p86 * 1.15 + 1.5;
  return pts.filter((_, i) => d[i] <= lim);
}

function capPoints(pts) {
  if (pts.length <= MAX_POINTS) return pts;
  const step = pts.length / MAX_POINTS;
  return Array.from({ length: MAX_POINTS }, (_, i) => pts[Math.floor(i * step)]);
}

/** Yasaklı el/telefon noktalarıyla çakışan ekran alt kenar noktalarını oyup çıkarır */
function carveOutPhoneFromAllowed() {
  if (!calib.phone || calib.phone.length < 5) return;
  for (const k of ["screen", "tablet"]) {
    if (!calib[k] || calib[k].length < 12) continue;
    const filtered = calib[k].filter((pt) => {
      const gPt = { x: pt[0], y: pt[1], headY: pt[2], lookDown: pt[3], open: pt[4], verg: pt[5] };
      let minD6 = Infinity;
      for (let i = 0; i < calib.phone.length; i++) {
        const d6 = dist6D(gPt, calib.phone[i], 1.4);
        if (d6 < minD6) minD6 = d6;
      }
      return minD6 > settings.tol * 1.15;
    });
    if (filtered.length >= 8) calib[k] = filtered;
  }
}

// ───────────────────────── Çift Aşamalı Keskin Kalibrasyon ─────────────────────────
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

/**
 * 📺 Ekran Kalibrasyonu:
 * AŞAMA 1: Kırmızı noktayı takip ederek ekranın 6D profilini çıkarır.
 * AŞAMA 2 (Otomatik): Hemen ardından 4 saniye boyunca ekranın önünde elinde telefon/nesne tutuyormuş gibi baktırır
 * ve o bakışı "Yasaklı Bölge" olarak kaydedip ekrandan kesip çıkarır!
 */
async function calibrateScreenAndHand() {
  calibrating = true;
  cancelCalib = false;
  calProg.style.width = "0%";
  calBadge.textContent = "AŞAMA 1 / 2 • EKRAN ALANI";
  calBadge.classList.remove("danger");
  ov.classList.remove("hidden");
  dot.classList.remove("hidden");
  try { await document.documentElement.requestFullscreen(); } catch {}
  await sleep(400);

  const path = [
    [0.06, 0.08], [0.94, 0.08], [0.94, 0.48], [0.06, 0.48],
    [0.06, 0.86], [0.50, 0.86], [0.94, 0.86], [0.50, 0.48],
    [0.25, 0.25], [0.75, 0.70], [0.50, 0.48],
  ];
  calText.textContent = "AŞAMA 1: Kırmızı noktayı GÖZLERİNLE takip et (normal çalışma duruşunu koru)";
  placeDot(path[0]);
  await sleep(1100);
  calText.textContent = "Ekran bakış ve göz açıklığı profili kaydediliyor…";
  sampling = { samples: [] };
  const segs = path.length - 1;
  for (let i = 1; i < path.length; i++) {
    if (cancelCalib) break;
    await moveDot(path[i - 1], path[i], 820, (i - 1) / segs, i / segs);
  }
  const screenSamples = sampling ? cleanPoints(sampling.samples) : [];
  sampling = null;
  dot.classList.add("hidden");

  if (!cancelCalib && screenSamples.length >= 10) {
    calib.screen = capPoints(screenSamples);
    beep(660);

    // AŞAMA 2: Ekran önünde elde telefon / nesne tutma simülasyonu
    calBadge.textContent = "AŞAMA 2 / 2 • ELDE TELEFON / NESNE YASAĞI";
    calBadge.classList.add("danger");
    calProg.style.width = "0%";
    for (let i = 3; i > 0; i--) {
      if (cancelCalib) break;
      calText.textContent = `AŞAMA 2 (${i}): Şimdi ekranın önünde ELİNDE TELEFON veya NESNE tutuyormuş gibi eline bak!`;
      await sleep(1000);
    }
    if (!cancelCalib) {
      beep(520);
      sampling = { samples: [] };
      const steps = 40;
      for (let i = 0; i < steps; i++) {
        if (cancelCalib) break;
        const rem = Math.ceil(((steps - i) * 100) / 1000);
        calText.textContent = `🚫 Ekran önünde elindeki telefona/nesneye bakmaya devam et… (${rem} sn)`;
        calProg.style.width = `${((i + 1) / steps) * 100}%`;
        await sleep(100);
      }
      const phoneSamples = sampling ? cleanPoints(sampling.samples) : [];
      sampling = null;
      if (phoneSamples.length >= 8) {
        calib.phone = capPoints(phoneSamples);
        carveOutPhoneFromAllowed();
      }
      beep(990);
    }
  }

  if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
  ov.classList.add("hidden");
  dot.classList.add("hidden");

  if (cancelCalib || screenSamples.length < 10) {
    setStatus("bad", cancelCalib ? "Kalibrasyon iptal edildi" : "Yüz/gözler yeterince algılanamadı, tekrar dene");
  } else {
    saveCalib();
    awayMs = 0;
    isAway = false;
    setStatus("ok", "✅ Ekran ve Elde Telefon/Nesne ayrımı başarıyla kalibre edildi!");
    drawMap(true);
  }
  calibrating = false;
}

async function calibrateFree(target) {
  calibrating = true;
  cancelCalib = false;
  calProg.style.width = "0%";
  const isPhone = target === "phone";
  calBadge.textContent = isPhone ? "YASAKLI AÇI TANITIMI" : `${ALL_REGIONS[target].label.toUpperCase()} KALİBRASYONU`;
  calBadge.classList.toggle("danger", isPhone);
  const label = ALL_REGIONS[target].label.toLowerCase();
  ov.classList.remove("hidden");

  for (let i = 3; i > 0; i--) {
    if (cancelCalib) break;
    calText.textContent = isPhone
      ? `${i}… Şimdi elinde telefon veya nesne tuttuğun yere bak!`
      : `${i}… İlk bip sesinden sonra ${label} alanının köşelerine ve ortasına bak.`;
    await sleep(950);
  }
  if (!cancelCalib) {
    beep(660);
    sampling = { samples: [] };
    const steps = isPhone ? 40 : 55;
    for (let i = 0; i < steps; i++) {
      if (cancelCalib) break;
      const rem = Math.ceil(((steps - i) * 100) / 1000);
      calText.textContent = isPhone
        ? `🚫 Elindeki telefona/nesneye bakmaya devam et… (${rem} sn)`
        : `${ALL_REGIONS[target].label} alanının tamamını gözlerinle tara… (${rem} sn)`;
      calProg.style.width = `${((i + 1) / steps) * 100}%`;
      await sleep(100);
    }
    beep(990);
  }
  ov.classList.add("hidden");
  const s = sampling;
  sampling = null;
  if (!cancelCalib && s) {
    const pts = cleanPoints(s.samples);
    if (pts.length >= 8) {
      calib[target] = capPoints(target === "extra" || isPhone ? (calib[target] || []).concat(pts) : pts);
      if (isPhone) carveOutPhoneFromAllowed();
      saveCalib();
      awayMs = 0;
      isAway = false;
      setStatus("ok", `${ALL_REGIONS[target].label} kaydedildi ✓`);
      drawMap(true);
    } else {
      setStatus("bad", "Yeterli veri alınamadı, kameraya bakarak tekrar dene");
    }
  }
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

  // Yakalanan el veya nesne kutularını çiz
  if (activeBoxes.length) {
    octx.strokeStyle = "#ff5c77";
    octx.lineWidth = 3;
    octx.fillStyle = "rgba(255, 92, 119, 0.18)";
    for (const b of activeBoxes) {
      const rx = b.x * vw, ry = b.y * vh, rw = b.w * vw, rh = b.h * vh;
      octx.fillRect(rx, ry, rw, rh);
      octx.strokeRect(rx, ry, rw, rh);
    }
  }

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
    mctx.fillStyle = ALL_REGIONS[k].color + (k === "phone" ? "48" : "2e");
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
    mctx.fillText("Başlamak için “📺 Ekran + El Ayrımını Kalibre Et” butonuna bas", W / 2, H / 2);
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
    setText("hudBadge", `${fps} FPS • 6D`);
    fpsFrames = 0;
    fpsLastAt = now;
  }

  const res = landmarker.detectForVideo(video, now);
  const lm = res.faceLandmarks?.[0];
  const blends = res.faceBlendshapes?.[0]?.categories;

  // Saniyede ~6 kez El & Nesne Tespiti çalıştır
  if (now - lastAuxCheckAt > 165) {
    lastAuxCheckAt = now;
    detectHandsAndObjects(lm, now);
  }
  const threatActive = activeThreat && now < activeThreatUntil;
  setText("telObj", threatActive ? `🚨 ${activeThreat}` : "Temiz ✓");

  if (lm && lm.length >= 478) {
    const f = analyze(lm, blends, video.videoWidth || 640, video.videoHeight || 480);

    const isBlinking = f.blinkScore > 0.62;
    if (isBlinking) {
      closedMs += dt;
    } else {
      closedMs = 0;
    }

    let raw;
    if (isBlinking && closedMs < 380 && lastValidRaw) {
      raw = lastValidRaw;
    } else if (closedMs >= 1500 && lastValidRaw) {
      raw = { ...lastValidRaw, y: lastValidRaw.y + 35, lookDown: lastValidRaw.lookDown + 35 };
    } else {
      raw = f;
      lastValidRaw = raw;
    }

    gaze = {
      x: fX.filter(raw.x, now),
      y: fY.filter(raw.y, now),
      headY: fHeadY.filter(raw.headY, now),
      lookDown: fDown.filter(raw.lookDown, now),
      open: fOpen.filter(raw.open, now),
      verg: fVerg.filter(raw.verg, now),
    };

    setText("telEyeDown", `${gaze.lookDown >= 0 ? "+" : ""}${gaze.lookDown.toFixed(1)}° (${Math.round(gaze.open * 100)}%)`);
    setText("telVerg", `${gaze.verg >= 0 ? "+" : ""}${gaze.verg.toFixed(1)}`);

    if (sampling && !isBlinking) {
      sampling.samples.push([gaze.x, gaze.y, gaze.headY, gaze.lookDown, gaze.open, gaze.verg]);
    }
  } else {
    gaze = null;
    fX.reset(); fY.reset(); fHeadY.reset(); fDown.reset(); fOpen.reset(); fVerg.reset();
  }

  drawMap();
  if (calibrating) {
    drawOverlay(lm, "#fcc419");
    return;
  }

  if (!hasRegions()) {
    drawOverlay(lm, "#8f98ad");
    setStatus("idle", lm ? "📺 Başlamak için “Ekran + El Ayrımını Kalibre Et” butonuna bas" : "Yüz aranıyor… Kameraya bak");
    return;
  }

  let away, reason;
  if (gaze) {
    faceLostFired = false;
    const ev = evaluateFocus(gaze, now);
    const d = ev.d;
    away = isAway ? d > 0.85 : d > 1.0;
    reason = ev.reason;
    setText("distVal", d < 0.2 ? "tam odaklı ✓" : `${d.toFixed(2)}x`);
    setBar("distBar", Math.min(d / 2, 1) * 100, away ? "#ff5c77" : "#38d996");
  } else {
    away = settings.faceLost && !faceLostFired;
    reason = "Yüzün kameradan çıktı";
    setText("distVal", "yüz yok");
  }
  isAway = away;

  const speedMult = threatActive ? 1.5 : 1.0;
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

  // El dedektörünü (HandLandmarker) arka planda yükle
  if (!handLandmarker) {
    HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.35,
      minHandPresenceConfidence: 0.35,
      minTrackingConfidence: 0.35,
    }).then((h) => { handLandmarker = h; }).catch(() => {
      HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "CPU" },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.35,
      }).then((h) => { handLandmarker = h; }).catch((e) => console.warn("HandLandmarker:", e));
    });
  }

  // Nesne dedektörünü (ObjectDetector) arka planda yükle
  if (!objDetector) {
    ObjectDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: OBJ_MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      scoreThreshold: 0.32,
      maxResults: 4,
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
    setText("loadMsg", "Yüz, el ve nesne yapay zeka motorları hazırlanıyor…");
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

$("cal-screen").addEventListener("click", () => !calibrating && calibrateScreenAndHand());
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
