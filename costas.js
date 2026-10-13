'use strict';

const SPS = 64;            // próbek na jeden symbol
const NBUF = 4096;         // długość historii (potęga 2)
const MASK = NBUF - 1;
const TAU = Math.PI * 2;
const RANGE = 1.6;         // zakres osi I/Q
const CYC = 4;             // cykli nośnej na symbol
const SHAPE = 16;          // wygładzanie zboczy I/Q w nadajniku [próbki]
const DEG = 180 / Math.PI;

const SIGNALS = {
  carrier: { label: 'Sama nośna', pts: [{ i: 1, q: 0, bits: '' }] },
  bpsk: { label: 'BPSK', pts: [{ i: -1, q: 0, bits: '0' }, { i: 1, q: 0, bits: '1' }] },
  qpsk: {
    label: 'QPSK',
    pts: [{ i: -1, q: -1, bits: '00' }, { i: -1, q: 1, bits: '01' }, { i: 1, q: -1, bits: '10' }, { i: 1, q: 1, bits: '11' }],
  },
};

// period: co ile stopni powtarzają się punkty zatrzasku
const DETECTORS = {
  pll: { label: 'Zwykła PLL', period: 360 },
  bpsk: { label: 'Costas BPSK', period: 180 },
  qpsk: { label: 'Costas QPSK', period: 90 },
};

const DEFAULTS = {
  signal: 'bpsk',
  det: 'bpsk',
  closed: true,           // pętla zamknięta
  integ: true,            // człon całkujący (pamięć częstotliwości)
  df: 0.02,               // odstrojenie nadajnika [cykle na symbol]
  noise: 0,
  bw: 0.08,               // szybkość pętli [cykle na symbol]
  zeta: 0.7,              // tłumienie
  lpf: 32,                // uśrednianie w gałęziach I/Q [próbki]
  speed: 4,               // próbek na klatkę
  win: 1024,              // szerokość okna czasu [próbki]
};

const wrapDeg = (d, period = 360) => d - period * Math.round(d / period);

/* ------------------------------------------------------------------ */
/* Symulacja: nadajnik → kanał → odbiornik z pętlą, próbka po próbce   */
/* ------------------------------------------------------------------ */

// Krótka historia z uśrednianiem ostatnich `len` próbek (filtr dolnoprzepustowy).
class Hist {
  constructor(size = 128) { this.a = new Float64Array(size); this.p = 0; }
  push(x) { this.p = (this.p + 1) % this.a.length; this.a[this.p] = x; }
  avg(len) {
    let s = 0, p = this.p;
    for (let k = 0; k < len; k++) { s += this.a[p]; p = p === 0 ? this.a.length - 1 : p - 1; }
    return s / len;
  }
}

function createSim(P, rand = Math.random) {
  const buf = {};
  for (const name of ['i', 'q', 'rI', 'rQ', 'e', 'es', 'fc', 'psi']) buf[name] = new Float32Array(NBUF);

  const s = {
    buf, n: 0,
    symLog: [],       // nadane symbole: { n, bits }
    decisions: [],    // decyzje odbiornika: { n, i, q, bits, ok }
    curve: [],        // punkty charakterystyki detektora: { x: błąd fazy [°], y: uśredniony sygnał błędu }
    stats: { symbols: 0, bits: 0, errors: 0 },
    psiDeg: 0, locked: false, lockAt: 0,
  };

  const txI = new Hist(), txQ = new Hist(), rxI = new Hist(), rxQ = new Hist();
  const symHist = new Array(64).fill(null);
  let phC = 0, cur = null;
  let psi = 0;        // rzeczywisty błąd fazy: faza nośnej nadajnika minus faza NCO [rad]
  let f = 0;          // stan członu całkującego: poprawka częstotliwości NCO [rad/próbkę]
  let es = 0, lockCount = 0;

  const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(TAU * rand());

  s.resetStats = () => { s.stats.symbols = s.stats.bits = s.stats.errors = 0; };
  s.kick = deg => { psi += deg / DEG; };
  s.resetLoop = (deg = null) => { f = 0; es = 0; lockCount = 0; if (deg !== null) psi = deg / DEG; };

  s.step = () => {
    const n = s.n, pts = SIGNALS[P.signal].pts;

    // --- nadajnik
    if (!cur || cur.pts !== pts || n % SPS === 0) {
      const k = Math.floor(n / SPS);
      cur = { k, pts, pt: pts[Math.floor(rand() * pts.length)], partial: n % SPS !== 0 };
      symHist[k & 63] = cur;
      s.symLog.push({ n, bits: cur.pt.bits });
      if (s.symLog[0].n < n - NBUF) s.symLog.shift();
    }
    txI.push(cur.pt.i); txQ.push(cur.pt.q);
    const bi = txI.avg(SHAPE), bq = txQ.avg(SHAPE);
    phC = (phC + TAU * CYC / SPS) % TAU;
    const rf = bi * Math.cos(phC) - bq * Math.sin(phC) + P.noise * gauss();

    // --- odbiornik: mieszanie z własną nośną (NCO) i uśrednianie
    const th = phC - psi;
    rxI.push(rf * Math.cos(th)); rxQ.push(-rf * Math.sin(th));
    const M = P.lpf;
    const rI = 2 * rxI.avg(M), rQ = 2 * rxQ.avg(M);

    // --- detektor błędu fazy
    let e;
    if (P.det === 'pll') e = rQ;
    else if (P.det === 'bpsk') e = rI * rQ;
    else e = 0.5 * (Math.sign(rI) * rQ - Math.sign(rQ) * rI);
    es += (e - es) / 32;

    // --- filtr pętli i NCO
    const wn = TAU * P.bw / SPS;
    let corr = 0;
    if (P.closed) {
      const fMax = TAU * 0.5 / SPS;
      f = P.integ ? Math.max(-fMax, Math.min(fMax, f + wn * wn * e)) : 0;
      corr = 2 * P.zeta * wn * e + f;
    } else {
      f = 0;
    }
    psi += TAU * P.df / SPS - corr;
    if (Math.abs(psi) > 1e6) psi %= TAU;

    // --- wskaźnik zatrzasku (symulacja zna prawdziwy błąd fazy)
    const period = DETECTORS[P.det].period;
    s.psiDeg = wrapDeg(psi * DEG);
    lockCount = P.closed && Math.abs(wrapDeg(s.psiDeg, period)) < Math.min(45, period / 4) ? lockCount + 1 : 0;
    s.locked = lockCount > 3 * SPS;
    s.lockAt = ((Math.round(s.psiDeg / period) * period) % 360 + 360) % 360;

    const x = n & MASK;
    buf.i[x] = bi; buf.q[x] = bq; buf.rI[x] = rI; buf.rQ[x] = rQ;
    buf.e[x] = e; buf.es[x] = es;
    buf.fc[x] = f * SPS / TAU;
    buf.psi[x] = s.psiDeg / 180;
    if (n % 4 === 0) {
      s.curve.push({ x: s.psiDeg, y: es });
      if (s.curve.length > 1500) s.curve.shift();
    }

    // --- decyzja w środku symbolu (z uwzględnieniem opóźnienia obu filtrów)
    if (pts.length > 1) {
      const m = n - Math.round((SHAPE - 1) / 2 + (M - 1) / 2) - SPS / 2;
      const sym = m >= 0 && m % SPS === 0 ? symHist[(m / SPS) & 63] : null;
      if (sym && sym.k === m / SPS && sym.pts === pts && !sym.partial) {
        let best = pts[0], bd = Infinity;
        for (const p of pts) {
          const d = (p.i - rI) ** 2 + (p.q - rQ) ** 2;
          if (d < bd) { bd = d; best = p; }
        }
        let err = 0;
        for (let c = 0; c < best.bits.length; c++) if (best.bits[c] !== sym.pt.bits[c]) err++;
        s.stats.symbols++; s.stats.bits += best.bits.length; s.stats.errors += err;
        s.decisions.push({ n, i: rI, q: rQ, bits: best.bits, ok: err === 0 });
        if (s.decisions.length > 400) s.decisions.shift();
      }
    }
    s.n++;
  };

  return s;
}

/* ------------------------------------------------------------------ */
/* Interfejs                                                           */
/* ------------------------------------------------------------------ */

const fmtSigned = v => (v > 0 ? '+' : '') + v;

const SLIDERS = [
  { group: 'Kanał', key: 'df', label: 'Odstrojenie nadajnika', min: -0.3, max: 0.3, step: 0.005, fmt: v => fmtSigned(+v.toFixed(3)) + ' cyklu / symbol' },
  { key: 'noise', label: 'Szum', min: 0, max: 1.5, step: 0.01, fmt: v => v.toFixed(2) },
  { group: 'Filtr pętli', key: 'bw', label: 'Szybkość pętli', min: 0.01, max: 0.6, step: 0.01, fmt: v => v.toFixed(2) },
  { key: 'zeta', label: 'Tłumienie', min: 0.1, max: 2, step: 0.05, fmt: v => v.toFixed(2) },
  { key: 'integ', label: 'Pamięć częstotliwości (człon całkujący)', check: true },
  { group: 'Gałęzie I/Q', key: 'lpf', label: 'Filtr (uśrednianie)', min: 1, max: 128, step: 1, fmt: v => v <= 1 ? 'wyłączony' : v + ' próbek' },
  { group: 'Podgląd', key: 'speed', label: 'Tempo', min: 0.25, max: 16, step: 0.25, fmt: v => v + ' próbek / klatkę' },
  { key: 'win', label: 'Okno czasu', min: 256, max: 4096, step: 128, fmt: v => v / SPS + ' symboli' },
];

const EXPERIMENTS = [
  {
    title: 'Pętla otwarta: wszystko się kręci',
    set: { closed: false, df: 0.03 },
    text: 'Pętla jest wyłączona, a nadajnik ma odrobinę inną częstotliwość niż NCO odbiornika. Konstelacja powoli wiruje, dane przelewają się z I do Q i z powrotem, a odczytane bity są na zmianę dobre i odwrócone. Spójrz na dolny wykres: gdy błąd fazy przechodzi przez wszystkie kąty, sygnał błędu kreśli falę — jego znak mówi, w którą stronę trzeba dokręcić NCO. Tę informację zaraz wykorzystamy.',
  },
  {
    title: 'Zamknij pętlę',
    set: { df: 0.03 },
    psi: 70,
    text: 'Ta sama sytuacja, ale sygnał błędu steruje teraz NCO. Błąd fazy (przebieg nr 5 i wskazówka) zjeżdża do zielonej linii, konstelacja nieruchomieje na osi I, a poprawka częstotliwości (nr 4) dochodzi do przerywanej linii rzeczywistego odstrojenia. Klikaj „skok fazy” i „Wyzeruj NCO”, żeby oglądać łapanie od nowa. Przesuń odstrojenie nadajnika — pętla pójdzie za nim.',
  },
  {
    title: 'Dlaczego nie zwykła PLL?',
    set: { det: 'pll', df: 0.02 },
    psi: 40,
    text: 'Zwykła pętla fazowa używa samego Q jako sygnału błędu. Dla gołej nośnej to działa (przełącz sygnał na „Sama nośna” i zobacz). Ale w BPSK każdy bit odwraca znak nośnej, więc Q co chwilę zmienia znak razem z danymi i pętla jest ciągnięta raz w jedną, raz w drugą stronę — nigdy się nie zatrzaskuje. Przełącz detektor na „Costas BPSK”: pomnożenie Q przez I kasuje znak danych i pętla łapie.',
  },
  {
    title: 'Dwuznaczność fazy',
    set: { df: 0 },
    text: 'Pętla jest zatrzaśnięta. Kliknij skok fazy „+180°”: pętla nawet nie drgnie, bo dla detektora konstelacja obrócona o pół obrotu wygląda identycznie — za to wszystkie bity są odwrócone. Kliknij „+90°”: punkt ląduje dokładnie między dwoma punktami zatrzasku i po chwili wahania zsuwa się do jednego z nich. Pętla Costasa odtwarza fazę tylko z dokładnością do tej dwuznaczności; resztę trzeba rozstrzygnąć inaczej, np. kodowaniem różnicowym.',
  },
  {
    title: 'Bez pamięci częstotliwości',
    set: { integ: false, df: 0.03 },
    text: 'Człon całkujący jest wyłączony. Pętla trzyma częstotliwość, ale tylko dlatego, że detektor stale „pcha” — a pchać może tylko wtedy, gdy błąd fazy nie jest zerem. Konstelacja stoi, lecz jest przekrzywiona, i im większe odstrojenie, tym bardziej (przy zbyt dużym pętla puszcza). Włącz „Pamięć częstotliwości”: poprawka przenosi się do przebiegu nr 4, a błąd fazy schodzi do zera.',
  },
  {
    title: 'Szybka czy wolna pętla?',
    set: { noise: 0.6, df: 0.02 },
    text: 'Do sygnału dodano szum. Przesuwaj „Szybkość pętli”: wolna pętla łapie długo, ale potem stoi spokojnie; szybka łapie od razu, lecz przepuszcza szum do NCO i konstelacja drga. Przy skrajnie dużej szybkości pętla staje się niestabilna, bo filtry w gałęziach opóźniają informację o błędzie. Zmieniaj też „Tłumienie”: małe daje kołysanie wokół punktu zatrzasku, duże — powolne dochodzenie.',
  },
  {
    title: 'Za duże odstrojenie',
    set: { df: 0.25, speed: 8, win: 4096 },
    text: 'Nadajnik jest odstrojony dużo bardziej, niż wynosi szybkość pętli. Błąd fazy przeskakuje kolejne obroty, a poprawka częstotliwości tylko powoli pełznie w stronę przerywanej linii — aż w końcu pętla chwyta. Zwiększ „Szybkość pętli”, żeby złapała wcześniej, albo wyłącz pamięć częstotliwości, żeby zobaczyć, że bez niej nie złapie wcale.',
  },
  {
    title: 'QPSK: cztery punkty zatrzasku',
    set: { signal: 'qpsk', det: 'qpsk' },
    psi: 30,
    text: 'W QPSK dane są i w I, i w Q, więc sztuczka „I × Q” nie wystarcza — detektor porównuje każdą gałąź ze znakiem drugiej. Konstelacja wygląda tak samo po obrocie o ćwierć obrotu, dlatego punktów zatrzasku jest cztery. Klikaj skok fazy „+90°”: pętla pozostaje zatrzaśnięta, ale bity są odczytywane błędnie. Przełącz detektor na „Costas BPSK” i zobacz, że nie potrafi utrzymać tego sygnału.',
  },
];

function initUI() {
  const P = { ...DEFAULTS };
  const sim = createSim(P);
  const $ = id => document.getElementById(id);
  const el = {
    scope: $('scope'), rx: $('rxPlane'), dial: $('dial'), curve: $('curve'),
    play: $('btnPlay'), loop: $('btnLoop'), sliders: $('sliders'),
    expButtons: $('expButtons'), expText: $('expText'), stats: $('stats'), lock: $('lock'),
  };
  const C = {};
  let running = true, acc = 0, frame = 0;

  /* --- kolory z CSS (jasny / ciemny motyw) --- */
  const readColors = () => {
    const cs = getComputedStyle(document.documentElement);
    for (const k of ['grid', 'axis', 'text', 'muted', 'ci', 'cq', 'rf', 'err', 'ok', 'ref'])
      C[k] = cs.getPropertyValue('--' + k).trim();
  };
  readColors();
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', readColors);

  /* --- kontrolki --- */
  const sliderRows = [];
  for (const sl of SLIDERS) {
    if (sl.group) {
      const h = document.createElement('h3');
      h.textContent = sl.group;
      el.sliders.append(h);
    }
    const row = document.createElement('div');
    const label = document.createElement('label');
    const name = document.createElement('span');
    name.textContent = sl.label;
    const input = document.createElement('input');
    input.id = 'sl-' + sl.key;
    label.htmlFor = input.id;
    let out = null;
    if (sl.check) {
      row.className = 'row check';
      input.type = 'checkbox';
      label.append(input, name);
      row.append(label);
      input.addEventListener('change', () => { P[sl.key] = input.checked; syncControls(); });
    } else {
      row.className = 'row';
      out = document.createElement('output');
      Object.assign(input, { type: 'range', min: sl.min, max: sl.max, step: sl.step });
      label.append(name, out);
      row.append(label, input);
      input.addEventListener('input', () => { P[sl.key] = +input.value; syncControls(); });
    }
    el.sliders.append(row);
    sliderRows.push({ sl, input, out });
  }

  const choiceButtons = (box, options, key) => {
    const map = {};
    for (const [val, o] of Object.entries(options)) {
      const b = document.createElement('button');
      b.textContent = o.label;
      b.addEventListener('click', () => { P[key] = val; clearResults(); syncControls(); });
      box.append(b);
      map[val] = b;
    }
    return map;
  };
  const signalButtons = choiceButtons($('signalButtons'), SIGNALS, 'signal');
  const detButtons = choiceButtons($('detButtons'), DETECTORS, 'det');

  function syncControls() {
    for (const { sl, input, out } of sliderRows) {
      if (sl.check) input.checked = P[sl.key];
      else { input.value = P[sl.key]; out.textContent = sl.fmt(P[sl.key], P); }
    }
    for (const [k, b] of Object.entries(signalButtons)) b.classList.toggle('active', k === P.signal);
    for (const [k, b] of Object.entries(detButtons)) b.classList.toggle('active', k === P.det);
    el.loop.textContent = P.closed ? 'Pętla: zamknięta' : 'Pętla: otwarta';
    el.loop.classList.toggle('active', P.closed);
  }

  function clearResults() {
    sim.resetStats();
    sim.decisions.length = 0;
    sim.curve.length = 0;
  }

  const expButtons = EXPERIMENTS.map((ex, idx) => {
    const b = document.createElement('button');
    b.textContent = (idx + 1) + '. ' + ex.title;
    b.addEventListener('click', () => applyExperiment(idx));
    el.expButtons.append(b);
    return b;
  });

  function applyExperiment(idx) {
    const ex = EXPERIMENTS[idx];
    Object.assign(P, DEFAULTS, ex.set);
    sim.resetLoop(ex.psi ?? 0);
    expButtons.forEach((b, k) => b.classList.toggle('active', k === idx));
    el.expText.textContent = ex.text;
    clearResults();
    syncControls();
    setRunning(true);
  }

  function setRunning(on) {
    running = on;
    el.play.textContent = on ? '⏸ Pauza' : '▶ Start';
  }

  el.play.addEventListener('click', () => setRunning(!running));
  $('btnStep').addEventListener('click', () => { setRunning(false); for (let k = 0; k < 8; k++) sim.step(); });
  $('btnDefaults').addEventListener('click', () => { Object.assign(P, DEFAULTS); sim.resetLoop(); clearResults(); syncControls(); });
  $('btnResetStats').addEventListener('click', clearResults);
  el.loop.addEventListener('click', () => { P.closed = !P.closed; sim.resetStats(); syncControls(); });
  $('btnResetLoop').addEventListener('click', () => { sim.resetLoop(); sim.resetStats(); });
  for (const b of document.querySelectorAll('[data-kick]'))
    b.addEventListener('click', () => { sim.kick(+b.dataset.kick); sim.resetStats(); });
  document.addEventListener('keydown', e => {
    if (e.code === 'Space' && !/^(BUTTON|INPUT)$/.test(e.target.tagName)) { e.preventDefault(); setRunning(!running); }
  });

  /* --- rysowanie --- */
  function fit(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
    if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph; }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  // zakresy pionowe kolejnych przebiegów
  const LANE_RANGE = [RANGE, RANGE, 1.3, 0.36, 1.15];

  function drawScope() {
    const { ctx, w, h } = fit(el.scope);
    const b = sim.buf, laneH = h / 5, W = P.win, n0 = sim.n - W;
    const period = DETECTORS[P.det].period;
    const X = s => (s - n0) / W * w;
    const geo = l => {
      const top = l * laneH + 28, bot = (l + 1) * laneH - 20;
      return { top, bot, mid: (top + bot) / 2, sc: (bot - top) / 2 / LANE_RANGE[l] };
    };
    const hline = (g, v) => { const y = g.mid - v * g.sc; ctx.moveTo(0, y); ctx.lineTo(w, y); };

    // siatka: granice symboli, zero, poziomy odniesienia
    ctx.lineWidth = 1;
    if (w / W * SPS >= 8) {
      ctx.strokeStyle = C.grid;
      ctx.beginPath();
      for (let s = Math.ceil(n0 / SPS) * SPS; s < sim.n; s += SPS) {
        const x = Math.round(X(s)) + 0.5;
        ctx.moveTo(x, 0); ctx.lineTo(x, h);
      }
      ctx.stroke();
    }
    for (let l = 0; l < 5; l++) {
      const g = geo(l);
      if (l < 3) {
        ctx.strokeStyle = C.grid;
        ctx.setLineDash([2, 4]);
        ctx.beginPath(); hline(g, 1); hline(g, -1); ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.strokeStyle = C.axis;
      ctx.beginPath();
      ctx.moveTo(0, Math.round(g.mid) + 0.5); ctx.lineTo(w, Math.round(g.mid) + 0.5);
      if (l) { ctx.moveTo(0, Math.round(l * laneH) + 0.5); ctx.lineTo(w, Math.round(l * laneH) + 0.5); }
      ctx.stroke();
    }

    const g3 = geo(3), g4 = geo(4);
    // nr 4: rzeczywiste odstrojenie
    ctx.strokeStyle = C.ref;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath(); hline(g3, P.df); ctx.stroke();
    // nr 5: punkty zatrzasku
    ctx.strokeStyle = C.ok;
    ctx.globalAlpha = 0.55;
    ctx.setLineDash([]);
    ctx.beginPath();
    for (let d = -180; d <= 180; d += period) hline(g4, d / 180);
    ctx.stroke();
    ctx.globalAlpha = 1;

    const trace = (l, arr, color, o = {}) => {
      const g = geo(l);
      ctx.save();
      ctx.beginPath(); ctx.rect(0, l * laneH + 1, w, laneH - 2); ctx.clip();
      ctx.beginPath();
      let prev = null;
      for (let s = Math.max(n0, sim.n - NBUF, 0); s < sim.n; s++) {
        const v = arr[s & MASK], x = X(s), y = g.mid - v * g.sc;
        // przebieg fazy "zawija się" na ±180° — nie łącz tych skoków kreską
        if (prev === null || (o.wrap && Math.abs(v - prev) > 1)) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        prev = v;
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = o.width || 1.6;
      ctx.globalAlpha = o.alpha || 1;
      ctx.lineJoin = 'round';
      ctx.stroke();
      ctx.restore();
    };

    trace(0, b.i, C.ci, { width: 2 });
    trace(0, b.q, C.cq, { width: 2 });
    trace(1, b.rI, C.ci, { width: 2 });
    trace(1, b.rQ, C.cq, { width: 2 });
    trace(2, b.e, C.rf, { alpha: 0.3, width: 1 });
    trace(2, b.es, C.rf, { width: 2.2 });
    trace(3, b.fc, C.ok, { width: 2.2 });
    trace(4, b.psi, C.text, { width: 2, wrap: true });

    // podziałka przy prawej krawędzi
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillStyle = C.muted;
    for (const d of [-180, -90, 90, 180]) ctx.fillText(fmtSigned(d) + '°', w - 5, g4.mid - d / 180 * g4.sc - 3);
    for (const v of [-0.3, 0.3]) ctx.fillText(fmtSigned(v), w - 5, g3.mid - v * g3.sc + 3);

    // nadane i odebrane bity (gdy mieszczą się na ekranie)
    const showBits = w / W * SPS >= 18 * SIGNALS[P.signal].pts[0].bits.length;
    ctx.font = '11px ui-monospace, Consolas, monospace';
    ctx.textAlign = 'center';
    if (showBits) {
      ctx.fillStyle = C.muted;
      for (const e of sim.symLog)
        if (e.n + SPS > n0) ctx.fillText(e.bits, X(e.n + SPS / 2), laneH - 6);
    }
    const g1 = geo(1);
    for (const d of sim.decisions) {
      if (d.n < n0) continue;
      const x = X(d.n);
      for (const [v, col] of [[d.i, C.ci], [d.q, C.cq]]) {
        ctx.fillStyle = d.ok ? col : C.err;
        ctx.beginPath();
        ctx.arc(x, Math.max(g1.top - 6, Math.min(g1.bot + 6, g1.mid - v * g1.sc)), 3, 0, TAU);
        ctx.fill();
      }
      if (showBits) {
        ctx.fillStyle = d.ok ? C.text : C.err;
        ctx.fillText(d.bits, x, 2 * laneH - 6);
      }
    }
  }

  function drawRx(i, q) {
    const { ctx, w, h } = fit(el.rx);
    const cx = w / 2, cy = h / 2, sc = Math.min(w, h) / 2 / RANGE;
    const px = v => cx + v * sc, py = v => cy - v * sc;
    ctx.lineWidth = 1;
    ctx.strokeStyle = C.grid;
    ctx.beginPath();
    ctx.arc(cx, cy, sc, 0, TAU);
    ctx.rect(cx - sc, cy - sc, 2 * sc, 2 * sc);
    ctx.stroke();
    ctx.strokeStyle = C.axis;
    ctx.beginPath();
    ctx.moveTo(0, Math.round(cy) + 0.5); ctx.lineTo(w, Math.round(cy) + 0.5);
    ctx.moveTo(Math.round(cx) + 0.5, 0); ctx.lineTo(Math.round(cx) + 0.5, h);
    ctx.stroke();
    ctx.font = '600 12px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillStyle = C.ci; ctx.fillText('I', w - 6, cy - 6);
    ctx.textAlign = 'left';
    ctx.fillStyle = C.cq; ctx.fillText('Q', cx + 6, 14);

    // punkty idealne
    ctx.strokeStyle = C.muted;
    ctx.beginPath();
    for (const p of SIGNALS[P.signal].pts) {
      const x = px(p.i), y = py(p.q);
      ctx.moveTo(x - 4, y); ctx.lineTo(x + 4, y);
      ctx.moveTo(x, y - 4); ctx.lineTo(x, y + 4);
    }
    ctx.stroke();

    // ślad
    const from = Math.max(0, sim.n - 256);
    ctx.beginPath();
    for (let s = from; s < sim.n; s++) {
      const x = px(sim.buf.rI[s & MASK]), y = py(sim.buf.rQ[s & MASK]);
      if (s === from) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = C.rf;
    ctx.globalAlpha = 0.35;
    ctx.lineWidth = 1.2;
    ctx.stroke();

    const nd = sim.decisions.length;
    sim.decisions.forEach((d, k) => {
      ctx.globalAlpha = 0.15 + 0.85 * (k + 1) / nd;
      ctx.fillStyle = d.ok ? C.rf : C.err;
      ctx.beginPath(); ctx.arc(px(d.i), py(d.q), 2.5, 0, TAU); ctx.fill();
    });
    ctx.globalAlpha = 1;
    ctx.strokeStyle = C.text;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(px(i), py(q), 5, 0, TAU); ctx.stroke();
  }

  function drawDial() {
    const { ctx, w, h } = fit(el.dial);
    const cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 22;
    const period = DETECTORS[P.det].period, a = sim.psiDeg / DEG;
    ctx.lineWidth = 1;
    ctx.strokeStyle = C.axis;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, TAU); ctx.stroke();
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let d = 0; d < 360; d += 45) {
      const c = Math.cos(d / DEG), s = Math.sin(d / DEG), stable = d % period === 0;
      ctx.strokeStyle = C.axis;
      ctx.beginPath();
      ctx.moveTo(cx + c * (r - 5), cy - s * (r - 5)); ctx.lineTo(cx + c * r, cy - s * r);
      ctx.stroke();
      if (stable) {
        ctx.fillStyle = C.ok;
        ctx.beginPath(); ctx.arc(cx + c * r, cy - s * r, 5, 0, TAU); ctx.fill();
      }
      if (d % 90 === 0) {
        ctx.fillStyle = C.muted;
        ctx.fillText(wrapDeg(d) === -180 ? '±180°' : fmtSigned(wrapDeg(d)) + '°', cx + c * (r + 6) + (c > 0.5 ? 6 : c < -0.5 ? -6 : 0) * 2, cy - s * (r + 12));
      }
    }
    ctx.textBaseline = 'alphabetic';
    ctx.strokeStyle = sim.locked ? C.ok : C.rf;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(a) * (r - 10), cy - Math.sin(a) * (r - 10)); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, TAU); ctx.fill();
  }

  function drawCurve() {
    const { ctx, w, h } = fit(el.curve);
    const top = 6, bot = h - 16, mid = (top + bot) / 2, sc = (bot - top) / 2 / 1.2;
    const period = DETECTORS[P.det].period;
    const X = d => (d + 180) / 360 * w;
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.lineWidth = 1;
    ctx.strokeStyle = C.grid;
    ctx.fillStyle = C.muted;
    ctx.beginPath();
    for (let d = -180; d <= 180; d += 45) {
      const x = Math.round(X(d)) + 0.5;
      ctx.moveTo(x, top); ctx.lineTo(x, bot);
      if (d % 90 === 0 && Math.abs(d) < 180) ctx.fillText(fmtSigned(d) + '°', x, h - 3);
    }
    ctx.stroke();
    ctx.strokeStyle = C.axis;
    ctx.beginPath(); ctx.moveTo(0, Math.round(mid) + 0.5); ctx.lineTo(w, Math.round(mid) + 0.5); ctx.stroke();

    ctx.fillStyle = C.ok;
    for (let d = -180; d <= 180; d += period) { ctx.beginPath(); ctx.arc(X(d), mid, 5, 0, TAU); ctx.fill(); }

    const pts = sim.curve, np = pts.length;
    ctx.fillStyle = C.rf;
    for (let k = 0; k < np; k++) {
      ctx.globalAlpha = 0.08 + 0.6 * (k + 1) / np;
      ctx.fillRect(X(pts[k].x) - 1, mid - pts[k].y * sc - 1, 2, 2);
    }
    ctx.globalAlpha = 1;
    if (np) {
      const p = pts[np - 1];
      ctx.strokeStyle = C.text;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(X(p.x), Math.max(top, Math.min(bot, mid - p.y * sc)), 5, 0, TAU); ctx.stroke();
    }
  }

  function updateReadouts(ri, rq, fc) {
    const f = v => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(2);
    $('rxI').textContent = f(ri);
    $('rxQ').textContent = f(rq);
    $('roPsi').textContent = fmtSigned(Math.round(sim.psiDeg)) + '°';
    $('roFreq').textContent = (fc < 0 ? '−' : '+') + Math.abs(fc).toFixed(3);
    const st = sim.stats;
    el.stats.textContent = SIGNALS[P.signal].pts.length > 1
      ? `Symbole: ${st.symbols} · błędne bity: ${st.errors} z ${st.bits}` + (st.bits ? ` (${(100 * st.errors / st.bits).toFixed(1)}%)` : '')
      : 'Sama nośna — brak bitów.';
    el.lock.textContent = !P.closed ? 'Pętla otwarta — NCO biegnie swobodnie.'
      : sim.locked ? `Zatrzaśnięta przy ${sim.lockAt}°.` : 'Niezatrzaśnięta — szuka.';
    el.lock.classList.toggle('on', sim.locked);
  }

  function tick() {
    if (running) {
      acc += P.speed;
      while (acc >= 1) { sim.step(); acc--; }
    }
    const last = (sim.n - 1) & MASK, b = sim.buf;
    drawScope();
    drawRx(b.rI[last], b.rQ[last]);
    drawDial();
    drawCurve();
    if (frame % 4 === 0) updateReadouts(b.rI[last], b.rQ[last], b.fc[last]);
    frame++;
    requestAnimationFrame(tick);
  }

  // rozbieg, żeby ekran od razu był wypełniony
  applyExperiment(1);
  for (let k = 0; k < 3 * SPS; k++) sim.step();
  requestAnimationFrame(tick);
}

if (typeof document !== 'undefined' && document.getElementById('scope')) initUI();
