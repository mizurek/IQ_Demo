'use strict';

const SPS = 64;            // próbek na jeden symbol
const NBUF = 2048;         // długość historii (potęga 2 — także rozmiar FFT)
const MASK = NBUF - 1;
const TAU = Math.PI * 2;
const RANGE = 1.6;         // zakres osi na przebiegach i płaszczyznach IQ

/* ------------------------------------------------------------------ */
/* Konstelacje                                                         */
/* ------------------------------------------------------------------ */

function gray(j, bits) {
  return (j ^ (j >> 1)).toString(2).padStart(bits, '0');
}

function qam(levels) {
  const nb = Math.log2(levels), pts = [];
  const lv = a => (2 * a - (levels - 1)) / (levels - 1);
  for (let a = 0; a < levels; a++)
    for (let b = 0; b < levels; b++)
      pts.push({ i: lv(a), q: lv(b), bits: gray(a, nb) + gray(b, nb) });
  return pts;
}

function psk(m) {
  const nb = Math.log2(m), pts = [];
  for (let j = 0; j < m; j++)
    pts.push({ i: Math.cos(TAU * j / m), q: Math.sin(TAU * j / m), bits: gray(j, nb) });
  return pts;
}

const MODES = {
  manual: { label: 'Ręczny' },
  rotate: { label: 'Wirujący wektor' },
  bpsk: { label: 'BPSK', pts: [{ i: -1, q: 0, bits: '0' }, { i: 1, q: 0, bits: '1' }] },
  qpsk: { label: 'QPSK', pts: qam(2) },
  psk8: { label: '8-PSK', pts: psk(8) },
  qam16: { label: '16-QAM', pts: qam(4) },
  qam64: { label: '64-QAM', pts: qam(8) },
};

const DEFAULTS = {
  mode: 'qpsk',
  manI: 0.8, manQ: 0.5,   // punkt w trybie ręcznym
  cycles: 4,              // cykli nośnej na symbol
  shape: 16,              // wygładzanie zboczy I/Q [próbki]
  rot: 1,                 // obroty wektora na symbol (tryb "wirujący")
  gainErr: 0,             // różnica wzmocnień I/Q [%]
  quadErr: 0,             // błąd przesunięcia 90° [stopnie]
  noise: 0,
  loPhase: 0,             // faza nośnej lokalnej [stopnie]
  loDetune: 0,            // odstrojenie nośnej lokalnej [cykle na symbol]
  lpf: 32,                // długość uśredniania w odbiorniku [próbki]
  speed: 2,               // próbek na klatkę
  win: 512,               // szerokość okna czasu [próbki]
};

/* ------------------------------------------------------------------ */
/* Symulacja: nadajnik → kanał → odbiornik, próbka po próbce           */
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
  for (const name of ['i', 'q', 'pI', 'pQ', 'rf', 'mI', 'mQ', 'rI', 'rQ']) buf[name] = new Float32Array(NBUF);

  const s = {
    buf, n: 0, refDelay: 0,
    symLog: [],       // nadane symbole: { n, bits }
    decisions: [],    // decyzje odbiornika: { n, i, q, bits, ok }
    stats: { symbols: 0, bits: 0, errors: 0 },
  };

  const txI = new Hist(), txQ = new Hist(), rxI = new Hist(), rxQ = new Hist();
  const symHist = new Array(64).fill(null);
  let phC = 0, phOff = 0, rotPh = 0, cur = null;

  const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(TAU * rand());

  s.resetStats = () => { s.stats.symbols = s.stats.bits = s.stats.errors = 0; };

  s.step = () => {
    const n = s.n, pts = MODES[P.mode].pts;

    // --- źródło: skąd biorą się I i Q
    let si, sq;
    if (P.mode === 'manual') {
      si = P.manI; sq = P.manQ; cur = null;
    } else if (P.mode === 'rotate') {
      rotPh = (rotPh + TAU * P.rot / SPS) % TAU;
      si = Math.cos(rotPh); sq = Math.sin(rotPh); cur = null;
    } else {
      if (!cur || cur.pts !== pts || n % SPS === 0) {
        const k = Math.floor(n / SPS);
        cur = { k, pts, pt: pts[Math.floor(rand() * pts.length)], partial: n % SPS !== 0 };
        symHist[k & 63] = cur;
        s.symLog.push({ n, bits: cur.pt.bits });
        if (s.symLog[0].n < n - NBUF) s.symLog.shift();
      }
      si = cur.pt.i; sq = cur.pt.q;
    }

    // --- wygładzenie zboczy
    txI.push(si); txQ.push(sq);
    const L = P.mode === 'rotate' ? 1 : P.shape;
    const bi = txI.avg(L), bq = txQ.avg(L);

    // --- modulator: dwie nośne przesunięte o ćwierć okresu, potem suma
    phC = (phC + TAU * P.cycles / SPS) % TAU;
    const g = P.gainErr / 100, e = P.quadErr * Math.PI / 180;
    const pI = bi * (1 + g / 2) * Math.cos(phC);
    const pQ = -bq * (1 - g / 2) * Math.sin(phC + e);

    // --- kanał
    const rf = pI + pQ + P.noise * gauss();

    // --- demodulator: mnożenie przez nośną lokalną i uśrednianie
    phOff = (phOff + TAU * P.loDetune / SPS) % TAU;
    const phL = phC + P.loPhase * Math.PI / 180 + phOff;
    const mI = rf * Math.cos(phL), mQ = -rf * Math.sin(phL);
    rxI.push(mI); rxQ.push(mQ);
    const M = P.lpf;
    const rI = 2 * rxI.avg(M), rQ = 2 * rxQ.avg(M);   // ×2: średnia iloczynu to połowa amplitudy

    const x = n & MASK;
    buf.i[x] = bi; buf.q[x] = bq; buf.pI[x] = pI; buf.pQ[x] = pQ; buf.rf[x] = rf;
    buf.mI[x] = mI; buf.mQ[x] = mQ; buf.rI[x] = rI; buf.rQ[x] = rQ;
    s.refDelay = Math.round((M - 1) / 2);

    // --- decyzja w środku symbolu (z uwzględnieniem opóźnienia obu filtrów)
    if (pts) {
      const m = n - Math.round((L - 1) / 2 + (M - 1) / 2) - SPS / 2;
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
/* Widmo                                                               */
/* ------------------------------------------------------------------ */

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -TAU / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* Interfejs                                                           */
/* ------------------------------------------------------------------ */

const fmtSigned = v => (v > 0 ? '+' : '') + v;

const SLIDERS = [
  { group: 'Nadajnik', key: 'cycles', label: 'Częstotliwość nośnej', min: 1, max: 10, step: 0.5, fmt: v => v + ' cykli / symbol' },
  { key: 'rot', label: 'Prędkość wirowania', min: -3, max: 3, step: 0.25, fmt: v => fmtSigned(v) + ' obr. / symbol', only: 'rotate' },
  { key: 'shape', label: 'Wygładzanie zboczy I/Q', min: 1, max: 64, step: 1, fmt: v => v <= 1 ? 'wyłączone' : Math.round(v / SPS * 100) + '% symbolu' },
  { key: 'gainErr', label: 'Nierówne wzmocnienie I i Q', min: -50, max: 50, step: 1, fmt: v => fmtSigned(v) + ' %' },
  { key: 'quadErr', label: 'Błąd przesunięcia o ćwierć okresu', min: -45, max: 45, step: 1, fmt: v => fmtSigned(v) + '°' },
  { group: 'Kanał', key: 'noise', label: 'Szum', min: 0, max: 1.5, step: 0.01, fmt: v => v.toFixed(2) },
  { group: 'Odbiornik', key: 'loPhase', label: 'Faza nośnej lokalnej', min: -180, max: 180, step: 1, fmt: v => fmtSigned(v) + '°' },
  { key: 'loDetune', label: 'Odstrojenie odbiornika', min: -0.2, max: 0.2, step: 0.005, fmt: v => fmtSigned(+v.toFixed(3)) + ' cyklu / symbol' },
  { key: 'lpf', label: 'Filtr (uśrednianie)', min: 1, max: 128, step: 1, fmt: (v, P) => v <= 1 ? 'wyłączony' : v + ' próbek · ' + (v * P.cycles / SPS).toFixed(1) + ' okr. nośnej' },
  { group: 'Podgląd', key: 'speed', label: 'Tempo', min: 0.25, max: 8, step: 0.25, fmt: v => v + ' próbek / klatkę' },
  { key: 'win', label: 'Okno czasu', min: 128, max: 1024, step: 64, fmt: v => v / SPS + ' symboli' },
];

const EXPERIMENTS = [
  {
    title: 'Dwie gałki, jedna nośna',
    set: { mode: 'manual', manI: 0.9, manQ: 0, win: 256 },
    text: 'Przeciągaj punkt na płaszczyźnie IQ nadajnika. Ruch w poziomie (I) zmienia wysokość niebieskiej fali, ruch w pionie (Q) — pomarańczowej, czyli tej samej nośnej przesuniętej o ćwierć okresu. Fioletowa suma to wciąż jedna sinusoida: odległość punktu od środka ustawia jej amplitudę, a kąt — jej fazę (porównaj odczyty pod wykresem z przebiegiem nr 3). Odbiornik odczytuje z niej z powrotem te same dwie liczby.',
  },
  {
    title: 'Wirujący wektor = inna częstotliwość',
    set: { mode: 'rotate', rot: 1 },
    text: 'Punkt krąży po okręgu, więc I i Q to dwie wolne sinusoidy. Spójrz na widmo: prążek nie leży na nośnej, tylko obok niej. Przesuń „Prędkość wirowania” na wartość ujemną — prążek przeskoczy na drugą stronę nośnej. Kierunek obrotu decyduje, czy częstotliwość rośnie, czy maleje; do tego potrzebne są obie składowe naraz.',
  },
  {
    title: 'Od bitów do bitów (QPSK)',
    set: { mode: 'qpsk' },
    text: 'Każdy symbol niesie dwa bity: pierwszy ustawia znak I, drugi znak Q (cyfry pod przebiegiem nr 1). W sygnale RF widać to jako skoki fazy. Odbiornik mnoży RF przez własną nośną, uśrednia i w środku każdego symbolu (kropki) sprawdza, który punkt konstelacji jest najbliżej — odczytane bity są na dole. Przełącz na 16-QAM lub 64-QAM: ta sama maszyneria, tylko więcej poziomów na każdej osi.',
  },
  {
    title: 'Po co filtr w odbiorniku?',
    set: { mode: 'qpsk', lpf: 1 },
    text: 'Filtr jest wyłączony. Surowy iloczyn RF × nośna lokalna faluje z podwojoną częstotliwością nośnej, ale jego średnia to właśnie szukane I (lub Q). Bez uśredniania odzyskane IQ rozmazuje się w pętle. Przesuwaj „Filtr (uśrednianie)” w prawo i patrz, jak pętle kurczą się do punktów — najczyściej, gdy okno obejmuje całkowitą liczbę półokresów nośnej. Za długi filtr zaczyna zlewać sąsiednie symbole.',
  },
  {
    title: 'Zła faza w odbiorniku',
    set: { mode: 'qpsk', loPhase: 30 },
    text: 'Nośna lokalna odbiornika jest przesunięta względem nadajnika o 30°. Konstelacja odbiornika obróciła się o ten sam kąt, a część I „przecieka” do Q i odwrotnie. Zwiększaj fazę: przy 45° odbiornik zaczyna się mylić, przy 90° I i Q zamieniają się rolami, przy 180° oba mają odwrócony znak. Dlatego prawdziwy odbiornik musi odtworzyć fazę nośnej.',
  },
  {
    title: 'Odstrojony odbiornik',
    set: { mode: 'qpsk', loDetune: 0.02 },
    text: 'Nośna lokalna ma odrobinę inną częstotliwość niż nadajnik. Błąd fazy stale narasta, więc konstelacja się kręci, a odzyskane I i Q powoli przelewają się jedno w drugie. Zmień znak odstrojenia — obrót zmieni kierunek. Im większe odstrojenie, tym szybciej wiruje.',
  },
  {
    title: 'Szum i gęste konstelacje',
    set: { mode: 'qam16', noise: 0.5 },
    text: 'Szum dodany do RF zamienia punkty konstelacji w chmurki. Dopóki chmurki się nie stykają, bity są odczytywane bezbłędnie. Przełączaj QPSK → 16-QAM → 64-QAM przy tym samym szumie i obserwuj licznik błędów: gęstsza konstelacja przenosi więcej bitów na symbol, ale punkty leżą bliżej siebie. Dłuższy filtr zmniejsza chmurki.',
  },
  {
    title: 'Niedoskonały modulator',
    set: { mode: 'rotate', rot: 1, gainErr: 30 },
    text: 'Gałęzie I i Q mają różne wzmocnienie. W widmie po drugiej stronie nośnej wyrósł niechciany, lustrzany prążek, a okrąg w odbiorniku stał się elipsą. Wyzeruj wzmocnienie i ustaw zamiast niego „Błąd przesunięcia o ćwierć okresu” — efekt jest podobny, tylko elipsa jest pochylona. Przełącz na 16-QAM, żeby zobaczyć, jak te same błędy zniekształcają siatkę punktów.',
  },
  {
    title: 'Dlaczego dwie gałęzie? (BPSK)',
    set: { mode: 'bpsk' },
    text: 'BPSK używa tylko składowej I — Q jest stale zerem. Teraz przesuwaj „Fazę nośnej lokalnej” w stronę 90°: sygnał w gałęzi I odbiornika maleje do zera i w całości pojawia się w gałęzi Q. Odbiornik z jedną gałęzią przy pechowej fazie nie widziałby nic; z dwiema nic nie ginie — punkt po prostu leży pod innym kątem.',
  },
];

function initUI() {
  const P = { ...DEFAULTS };
  const sim = createSim(P);
  const $ = id => document.getElementById(id);
  const el = {
    scope: $('scope'), tx: $('txPlane'), rx: $('rxPlane'), spec: $('spectrum'),
    play: $('btnPlay'), modes: $('modeButtons'), sliders: $('sliders'),
    expButtons: $('expButtons'), expText: $('expText'), stats: $('stats'),
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
    row.className = 'row';
    const label = document.createElement('label');
    const name = document.createElement('span');
    name.textContent = sl.label;
    const out = document.createElement('output');
    const input = document.createElement('input');
    Object.assign(input, { type: 'range', min: sl.min, max: sl.max, step: sl.step, id: 'sl-' + sl.key });
    label.htmlFor = input.id;
    label.append(name, out);
    row.append(label, input);
    el.sliders.append(row);
    input.addEventListener('input', () => { P[sl.key] = +input.value; syncControls(); });
    sliderRows.push({ sl, row, input, out });
  }

  const modeButtons = {};
  for (const [key, m] of Object.entries(MODES)) {
    const b = document.createElement('button');
    b.textContent = m.label;
    b.addEventListener('click', () => setMode(key));
    el.modes.append(b);
    modeButtons[key] = b;
  }

  function syncControls() {
    for (const { sl, row, input, out } of sliderRows) {
      input.value = P[sl.key];
      out.textContent = sl.fmt(P[sl.key], P);
      row.hidden = !!sl.only && sl.only !== P.mode;
    }
    for (const [key, b] of Object.entries(modeButtons)) b.classList.toggle('active', key === P.mode);
  }

  function clearResults() {
    sim.resetStats();
    sim.decisions.length = 0;
  }

  function setMode(mode) {
    P.mode = mode;
    clearResults();
    syncControls();
  }

  const expButtons = EXPERIMENTS.map((ex, idx) => {
    const b = document.createElement('button');
    b.textContent = (idx + 1) + '. ' + ex.title;
    b.addEventListener('click', () => applyExperiment(idx));
    el.expButtons.append(b);
    return b;
  });

  function applyExperiment(idx) {
    Object.assign(P, DEFAULTS, EXPERIMENTS[idx].set);
    expButtons.forEach((b, k) => b.classList.toggle('active', k === idx));
    el.expText.textContent = EXPERIMENTS[idx].text;
    clearResults();
    syncControls();
    setRunning(true);
  }

  function setRunning(on) {
    running = on;
    el.play.textContent = on ? '⏸ Pauza' : '▶ Start';
  }

  el.play.addEventListener('click', () => setRunning(!running));
  $('btnStep').addEventListener('click', () => { setRunning(false); for (let k = 0; k < 4; k++) sim.step(); });
  $('btnDefaults').addEventListener('click', () => { Object.assign(P, DEFAULTS); clearResults(); syncControls(); });
  $('btnResetStats').addEventListener('click', clearResults);
  document.addEventListener('keydown', e => {
    if (e.code === 'Space' && !/^(BUTTON|INPUT)$/.test(e.target.tagName)) { e.preventDefault(); setRunning(!running); }
  });

  /* --- przeciąganie punktu na płaszczyźnie nadajnika --- */
  let dragging = false;
  const dragTo = e => {
    const r = el.tx.getBoundingClientRect();
    const sc = Math.min(r.width, r.height) / 2 / RANGE;
    const clamp = v => Math.max(-1.5, Math.min(1.5, v));
    P.manI = clamp((e.clientX - r.left - r.width / 2) / sc);
    P.manQ = clamp(-(e.clientY - r.top - r.height / 2) / sc);
    if (P.mode !== 'manual') setMode('manual');
  };
  el.tx.addEventListener('pointerdown', e => { dragging = true; el.tx.setPointerCapture(e.pointerId); dragTo(e); });
  el.tx.addEventListener('pointermove', e => { if (dragging) dragTo(e); });
  el.tx.addEventListener('pointerup', () => { dragging = false; });
  el.tx.addEventListener('pointercancel', () => { dragging = false; });

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

  function drawScope() {
    const { ctx, w, h } = fit(el.scope);
    const b = sim.buf, laneH = h / 5, W = P.win, n0 = sim.n - W;
    const X = s => (s - n0) / W * w;
    const geo = l => {
      const top = l * laneH + 28, bot = (l + 1) * laneH - 20;
      return { top, bot, mid: (top + bot) / 2, sc: (bot - top) / 2 / RANGE };
    };

    // siatka: granice symboli, zero i poziomy ±1
    ctx.lineWidth = 1;
    ctx.strokeStyle = C.grid;
    ctx.beginPath();
    for (let s = Math.ceil(n0 / SPS) * SPS; s < sim.n; s += SPS) {
      const x = Math.round(X(s)) + 0.5;
      ctx.moveTo(x, 0); ctx.lineTo(x, h);
    }
    ctx.stroke();
    for (let l = 0; l < 5; l++) {
      const g = geo(l);
      ctx.strokeStyle = C.grid;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      for (const v of [-1, 1]) { ctx.moveTo(0, g.mid - v * g.sc); ctx.lineTo(w, g.mid - v * g.sc); }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = C.axis;
      ctx.beginPath();
      ctx.moveTo(0, Math.round(g.mid) + 0.5); ctx.lineTo(w, Math.round(g.mid) + 0.5);
      if (l) { ctx.moveTo(0, Math.round(l * laneH) + 0.5); ctx.lineTo(w, Math.round(l * laneH) + 0.5); }
      ctx.stroke();
    }

    const trace = (l, arr, color, o = {}) => {
      const g = geo(l), d = o.delay || 0;
      ctx.save();
      ctx.beginPath(); ctx.rect(0, l * laneH + 1, w, laneH - 2); ctx.clip();
      ctx.beginPath();
      let first = true;
      for (let s = Math.max(n0, d, sim.n - NBUF + d); s < sim.n; s++) {
        const x = X(s), y = g.mid - arr[(s - d) & MASK] * g.sc;
        if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = o.width || 1.6;
      ctx.globalAlpha = o.alpha || 1;
      ctx.setLineDash(o.dash || []);
      ctx.lineJoin = 'round';
      ctx.stroke();
      ctx.restore();
    };

    trace(0, b.i, C.ci, { width: 2 });
    trace(0, b.q, C.cq, { width: 2 });

    trace(1, b.i, C.ci, { alpha: 0.4, width: 1, dash: [3, 3] });
    trace(1, b.q, C.cq, { alpha: 0.4, width: 1, dash: [3, 3] });
    trace(1, b.pI, C.ci);
    trace(1, b.pQ, C.cq);

    trace(2, b.rf, C.rf);

    trace(3, b.mI, C.ci, { alpha: 0.3, width: 1 });
    trace(3, b.i, C.ref, { delay: sim.refDelay, width: 1.2, dash: [5, 4] });
    trace(3, b.rI, C.ci, { width: 2.2 });

    trace(4, b.mQ, C.cq, { alpha: 0.3, width: 1 });
    trace(4, b.q, C.ref, { delay: sim.refDelay, width: 1.2, dash: [5, 4] });
    trace(4, b.rQ, C.cq, { width: 2.2 });

    // nadane bity
    ctx.font = '11px ui-monospace, Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.fillStyle = C.muted;
    for (const e of sim.symLog)
      if (e.n + SPS > n0) ctx.fillText(e.bits, X(e.n + SPS / 2), laneH - 6);

    // chwile decyzji i odebrane bity
    const g3 = geo(3), g4 = geo(4);
    for (const d of sim.decisions) {
      if (d.n < n0) continue;
      const x = X(d.n);
      for (const [g, v, col] of [[g3, d.i, C.ci], [g4, d.q, C.cq]]) {
        ctx.fillStyle = d.ok ? col : C.err;
        ctx.beginPath();
        ctx.arc(x, Math.max(g.top - 6, Math.min(g.bot + 6, g.mid - v * g.sc)), 3.5, 0, TAU);
        ctx.fill();
      }
      ctx.fillStyle = d.ok ? C.text : C.err;
      ctx.fillText(d.bits, x, h - 6);
    }
  }

  function planeBase(canvas) {
    const { ctx, w, h } = fit(canvas);
    const cx = w / 2, cy = h / 2, sc = Math.min(w, h) / 2 / RANGE;
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
    return { ctx, px: v => cx + v * sc, py: v => cy - v * sc };
  }

  const trail = (ctx, pl, ai, aq, len, color) => {
    ctx.beginPath();
    for (let s = Math.max(0, sim.n - len); s < sim.n; s++) {
      const x = pl.px(ai[s & MASK]), y = pl.py(aq[s & MASK]);
      if (s === Math.max(0, sim.n - len)) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.35;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.globalAlpha = 1;
  };

  function drawTx(i, q) {
    const pl = planeBase(el.tx), { ctx } = pl, pts = MODES[P.mode].pts;
    if (pts) {
      ctx.strokeStyle = C.muted;
      for (const p of pts) { ctx.beginPath(); ctx.arc(pl.px(p.i), pl.py(p.q), 4, 0, TAU); ctx.stroke(); }
    }
    trail(ctx, pl, sim.buf.i, sim.buf.q, 96, C.rf);

    const x = pl.px(i), y = pl.py(q), x0 = pl.px(0), y0 = pl.py(0);
    // rzuty na osie: "gałka I" i "gałka Q"
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = C.muted;
    ctx.beginPath();
    ctx.moveTo(x, y); ctx.lineTo(x, y0);
    ctx.moveTo(x, y); ctx.lineTo(x0, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.strokeStyle = C.ci; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x, y0); ctx.stroke();
    ctx.strokeStyle = C.cq; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y); ctx.stroke();
    // wektor
    ctx.lineWidth = 2;
    ctx.strokeStyle = C.rf;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x, y); ctx.stroke();
    ctx.fillStyle = C.rf;
    ctx.beginPath(); ctx.arc(x, y, 6, 0, TAU); ctx.fill();
  }

  function drawRx(i, q) {
    const pl = planeBase(el.rx), { ctx } = pl, pts = MODES[P.mode].pts;
    if (pts) {
      ctx.strokeStyle = C.muted;
      ctx.beginPath();
      for (const p of pts) {
        const x = pl.px(p.i), y = pl.py(p.q);
        ctx.moveTo(x - 4, y); ctx.lineTo(x + 4, y);
        ctx.moveTo(x, y - 4); ctx.lineTo(x, y + 4);
      }
      ctx.stroke();
    }
    trail(ctx, pl, sim.buf.rI, sim.buf.rQ, 192, C.rf);
    const nd = sim.decisions.length;
    sim.decisions.forEach((d, k) => {
      ctx.globalAlpha = 0.15 + 0.85 * (k + 1) / nd;
      ctx.fillStyle = d.ok ? C.rf : C.err;
      ctx.beginPath(); ctx.arc(pl.px(d.i), pl.py(d.q), 2.5, 0, TAU); ctx.fill();
    });
    ctx.globalAlpha = 1;
    ctx.strokeStyle = C.text;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(pl.px(i), pl.py(q), 5, 0, TAU); ctx.stroke();
  }

  const NB = NBUF / 4, DB_MIN = -70;
  const specRe = new Float64Array(NBUF), specIm = new Float64Array(NBUF);
  const specDb = new Float32Array(NB).fill(DB_MIN);

  function updateSpectrum() {
    for (let k = 0; k < NBUF; k++) {
      const s = sim.n - NBUF + k;
      specRe[k] = s < 0 ? 0 : sim.buf.rf[s & MASK] * (0.5 - 0.5 * Math.cos(TAU * k / (NBUF - 1)));
      specIm[k] = 0;
    }
    fft(specRe, specIm);
    for (let k = 0; k < NB; k++) {
      const db = Math.max(DB_MIN, 20 * Math.log10(Math.hypot(specRe[k], specIm[k]) / (NBUF / 4) + 1e-9));
      specDb[k] += 0.35 * (db - specDb[k]);
    }
  }

  function drawSpectrum() {
    const { ctx, w, h } = fit(el.spec);
    const top = 14, ph = h - 16 - top, maxC = NB * SPS / NBUF;
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.lineWidth = 1;
    ctx.strokeStyle = C.grid;
    ctx.fillStyle = C.muted;
    ctx.beginPath();
    for (let c = 0; c <= maxC; c += 2) {
      const x = Math.round(c / maxC * w) + 0.5;
      ctx.moveTo(x, top); ctx.lineTo(x, top + ph);
      if (c > 0 && c < maxC) ctx.fillText(c, x, h - 3);
    }
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, top + ph);
    for (let k = 0; k < NB; k++) ctx.lineTo(k / NB * w, top + ph * specDb[k] / DB_MIN);
    ctx.lineTo(w, top + ph);
    ctx.closePath();
    ctx.fillStyle = C.rf;
    ctx.globalAlpha = 0.2;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = C.rf;
    ctx.lineWidth = 1.4;
    ctx.stroke();

    const xc = P.cycles / maxC * w;
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = C.muted;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(xc, top); ctx.lineTo(xc, top + ph); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.muted;
    ctx.fillText('nośna', Math.max(18, xc), 10);
  }

  function updateReadouts(ti, tq, ri, rq) {
    const f = v => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(2);
    $('txI').textContent = f(ti);
    $('txQ').textContent = f(tq);
    $('txA').textContent = Math.hypot(ti, tq).toFixed(2);
    $('txP').textContent = Math.hypot(ti, tq) < 0.01 ? '—' : Math.round(Math.atan2(tq, ti) * 180 / Math.PI) + '°';
    $('rxI').textContent = f(ri);
    $('rxQ').textContent = f(rq);
    const st = sim.stats;
    el.stats.textContent = MODES[P.mode].pts
      ? `Symbole: ${st.symbols} · błędne bity: ${st.errors} z ${st.bits}` + (st.bits ? ` (${(100 * st.errors / st.bits).toFixed(1)}%)` : '')
      : 'Tryb bez bitów — obserwuj położenie punktu.';
  }

  function tick() {
    if (running) {
      acc += P.speed;
      while (acc >= 1) { sim.step(); acc--; }
    }
    const last = (sim.n - 1) & MASK, b = sim.buf;
    drawScope();
    drawTx(b.i[last], b.q[last]);
    drawRx(b.rI[last], b.rQ[last]);
    if (frame % 4 === 0) {
      updateSpectrum();
      updateReadouts(b.i[last], b.q[last], b.rI[last], b.rQ[last]);
    }
    drawSpectrum();
    frame++;
    requestAnimationFrame(tick);
  }

  // rozbieg, żeby ekran od razu był wypełniony
  for (let k = 0; k < NBUF; k++) sim.step();
  applyExperiment(2);
  requestAnimationFrame(tick);
}

if (typeof document !== 'undefined') initUI();
else module.exports = { createSim, DEFAULTS, MODES, SPS };
