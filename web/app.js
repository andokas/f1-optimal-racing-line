const DATA     = window.F1_DATA;
const C        = window.CIRCUITS;  // circuit metadata from circuits.js
let deckInstance = null;
let currentYear  = DATA.year;

// ── View switching ────────────────────────────────────────────────────────────
function showView(id) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

// ── SVG circuit minimap ───────────────────────────────────────────────────────
const SECTOR_COLORS = ['#e10600', '#00d2be', '#ffd700'];

function circuitSVG(c, w, h, strokeW) {
  const tel  = c.pole_tel;
  const xs   = (tel && tel.raw_x) ? tel.raw_x : c.ref.x;
  const ys   = (tel && tel.raw_y) ? tel.raw_y : c.ref.y;
  const N    = xs.length;

  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad    = strokeW * 4;
  const scaleX = (w - pad*2) / (maxX - minX);
  const scaleY = (h - pad*2) / (maxY - minY);
  const scale  = Math.min(scaleX, scaleY);
  const offX   = pad + ((w - pad*2) - (maxX - minX) * scale) / 2;
  const offY   = pad + ((h - pad*2) - (maxY - minY) * scale) / 2;

  const px = i => offX + (xs[i] - minX) * scale;
  const py = i => offY + (ys[i] - minY) * scale;

  // Sector boundaries
  let s1 = Math.floor(N / 3), s2 = Math.floor(2 * N / 3);
  if (tel && tel.sector_indices && tel.sector_indices.length === 2) {
    s1 = tel.sector_indices[0];
    s2 = tel.sector_indices[1];
  }
  const sectors = [
    { from: 0,  to: s1 },
    { from: s1, to: s2 },
    { from: s2, to: N - 1 },
  ];

  // Build polyline per sector
  let svgLines = '';
  sectors.forEach((s, si) => {
    const pts = [];
    for (let i = s.from; i <= Math.min(s.to, N-1); i++) pts.push(`${px(i)},${py(i)}`);
    if (pts.length > 1)
      svgLines += `<polyline points="${pts.join(' ')}" fill="none" stroke="${SECTOR_COLORS[si]}"
        stroke-width="${strokeW}" stroke-linejoin="round" stroke-linecap="round"/>`;
  });

  // Corner numbers
  let svgCorners = '';
  if (tel && tel.corners && tel.corners.length) {
    const fontSize = Math.max(6, strokeW * 2.2);
    const dotR     = Math.max(2, strokeW * 0.8);
    tel.corners.forEach(corner => {
      const cx2 = offX + (corner.x - minX) * scale;
      const cy2 = offY + (corner.y - minY) * scale;
      svgCorners += `
        <circle cx="${cx2}" cy="${cy2}" r="${dotR}" fill="#fff" opacity="0.7"/>
        <text x="${cx2 + dotR + 1}" y="${cy2 + fontSize*0.35}"
          font-size="${fontSize}" fill="#fff" opacity="0.75"
          font-family="JetBrains Mono,monospace" font-weight="500">${corner.n}</text>`;
    });
  }

  return `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">
    ${svgLines}${svgCorners}
  </svg>`;
}

// ── VIEW 1: Year grid ─────────────────────────────────────────────────────────
const ERA_LABELS = {
  2018:'2018–2021 Era', 2019:'2018–2021 Era', 2020:'2018–2021 Era', 2021:'2018–2021 Era',
  2022:'2022–2025 Era', 2023:'2022–2025 Era', 2024:'2022–2025 Era', 2025:'2022–2025 Era',
  2026:'2026 Era',
};
const years    = [...new Set(DATA.circuits.map(c => c.year))].sort();
const yearGrid = document.getElementById('year-grid');
years.forEach(y => {
  const card = document.createElement('div');
  card.className = 'year-card';
  card.innerHTML = `<div class="yr">${y}</div><div class="era">${ERA_LABELS[y] || ''}</div>`;
  card.onclick = () => {
    currentYear = y;
    buildCircuitSelection(y);
    document.getElementById('sel-season-label').textContent = 'Season ' + y;
    showView('view-circuit');
  };
  yearGrid.appendChild(card);
});

// ── VIEW 2: Circuit selection ─────────────────────────────────────────────────
function _stripLabel(slug) {
  const n = C[slug] || { country: slug, circuit: slug };
  const allSlugs = selCircuits.map(c => c.slug);
  const shared   = allSlugs.filter(s => (C[s] || {}).country === n.country);
  return shared.length > 1 ? n.circuit : n.country;
}

let selCircuit  = null;
let selIndex    = 0;
let selCircuits = [];

function buildCircuitSelection(year) {
  selCircuits = DATA.circuits.filter(c => c.year === year);
  selIndex    = 0;

  const strip = document.getElementById('flag-strip');
  strip.innerHTML = '';
  selCircuits.forEach((c, i) => {
    const code  = (C[c.slug] || {}).flag || 'un';
    const label = _stripLabel(c.slug);
    const item  = document.createElement('div');
    item.className   = 'flag-item' + (i === 0 ? ' active' : '');
    item.dataset.idx = i;
    item.innerHTML   = `
      <img src="https://flagcdn.com/w80/${code}.png" alt="${c.name}">
      <span class="flag-label">${label}</span>`;
    item.onclick = () => navigateTo(i, i > selIndex ? 'right' : 'left');
    strip.appendChild(item);
  });

  document.getElementById('nav-prev').onclick = () => navigateTo(selIndex - 1, 'left');
  document.getElementById('nav-next').onclick = () => navigateTo(selIndex + 1, 'right');

  selCircuit = selCircuits[0];
  renderPreview(selCircuits[0], null);
}

function navigateTo(idx, dir) {
  if (idx < 0 || idx >= selCircuits.length) return;
  const animClass = dir === 'right' ? 'slide-in-right' : 'slide-in-left';
  const prevIdx   = selIndex;
  selIndex        = idx;
  selCircuit      = selCircuits[idx];

  document.querySelectorAll('.flag-item').forEach((el, i) => {
    const c     = selCircuits[i];
    const label = el.querySelector('.flag-label');
    if (i === idx) {
      el.classList.add('active');
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    } else {
      el.classList.remove('active');
    }
  });

  renderPreview(selCircuit, animClass);
}

function renderPreview(c, animClass) {
  selCircuit = c;
  const opt     = c.opt;
  const preview = document.getElementById('circuit-preview');
  preview.className = 'circuit-preview';
  if (animClass) {
    preview.classList.add(animClass);
    preview.addEventListener('animationend', () => preview.classList.remove(animClass), { once: true });
  }
  const info = C[c.slug] || {};

  const lengthStr = info.length_km ? info.length_km.toFixed(3) + ' km' : (opt ? (opt.length_m / 1000).toFixed(3) + ' km' : '—');
  const firstStr  = info.first  || '—';
  const lapsStr   = info.laps   || '—';
  const recStr    = info.record || '—';
  const recWho    = info.record_driver ? `${info.record_driver} (${info.record_year})` : '';

  preview.innerHTML = `
    <div class="map-area">${circuitSVG(c, 520, 480, 4)}</div>
    <div class="info-panel">
      <div class="circuit-name">${(C[c.slug] || {}).country || c.name}</div>
      <div class="circuit-subtitle">${(C[c.slug] || {}).circuit || ''}</div>
      ${c.data_year !== c.year ? `<div class="circuit-year">Data ${c.data_year}</div>` : ''}
      <div class="info-grid">
        <div class="info-cell">
          <div class="ilbl">Circuit Length</div>
          <div class="ival">${lengthStr}</div>
        </div>
        <div class="info-cell">
          <div class="ilbl">Number of Laps</div>
          <div class="ival">${lapsStr}</div>
        </div>
        <div class="info-cell">
          <div class="ilbl">First Grand Prix</div>
          <div class="ival">${firstStr}</div>
        </div>
        <div class="info-cell">
          <div class="ilbl">Fastest Lap</div>
          <div class="ival">${recStr}</div>
          <div style="font-size:0.65rem;color:var(--muted);margin-top:3px;">${recWho}</div>
        </div>
      </div>
      <button class="select-btn" onclick="openDashboard()">Analyse Circuit →</button>
    </div>`;
}

function openDashboard() {
  if (!selCircuit) return;
  showView('view-dash');
  renderCircuit(selCircuit);
}

// ── Speed colorscale (global fixed scale) ─────────────────────────────────────
function speedColor(v) {
  const t = Math.max(0, Math.min(1, (v - 50) / (340 - 50)));
  const r = t < 0.5 ? 255 : Math.round(255 * (1 - (t-0.5)*2));
  const g = t < 0.5 ? Math.round(255 * t * 2) : 255;
  return [r, g, 40, 230];
}

// ── Deck.gl map ───────────────────────────────────────────────────────────────
function renderMap(c) {
  const container = document.getElementById('map-container');
  const opt = c.opt, N = opt.x.length;
  const cx = opt.x.reduce((a,b) => a+b, 0) / N;
  const cy = opt.y.reduce((a,b) => a+b, 0) / N;

  const refPath = [{ path: c.ref.x.map((x,i) => [x-cx, c.ref.y[i]-cy, 0]) }];
  const optSegs = [];
  for (let i = 0; i < N-1; i++) {
    optSegs.push({
      s: [opt.x[i]-cx,   opt.y[i]-cy,   0],
      e: [opt.x[i+1]-cx, opt.y[i+1]-cy, 0],
      color: speedColor(opt.v_kmh[i]),
    });
  }

  const maxRange = Math.max(
    Math.max(...c.ref.x) - Math.min(...c.ref.x),
    Math.max(...c.ref.y) - Math.min(...c.ref.y),
  );
  const zoom = Math.log2(500 / maxRange) + 1.2;

  const layers = [
    new deck.PathLayer({ id:'ref', data: refPath,
      getPath: d => d.path, getColor:[70,70,70,140], getWidth:1.5, widthUnits:'meters' }),
    new deck.LineLayer({ id:'opt', data: optSegs,
      getSourcePosition: d => d.s, getTargetPosition: d => d.e,
      getColor: d => d.color, getWidth:3, widthUnits:'meters' }),
  ];

  if (deckInstance) {
    deckInstance.setProps({
      layers,
      viewState: { target:[0,0,0], zoom, minZoom:zoom-3, maxZoom:zoom+5 },
    });
  } else {
    deckInstance = new deck.DeckGL({
      container,
      views: new deck.OrthographicView({ id:'ortho',
        controller:{ dragPan:true, scrollZoom:true, doubleClickZoom:true } }),
      initialViewState: { target:[0,0,0], zoom, minZoom:zoom-3, maxZoom:zoom+5 },
      layers,
      style: { background:'#0a0a0a' },
    });
  }
}

// ── Plotly ────────────────────────────────────────────────────────────────────
const LAYOUT = yTitle => ({
  paper_bgcolor: '#0a0a0a', plot_bgcolor: '#0a0a0a',
  font:  { color:'#888', size:10, family:'JetBrains Mono' },
  xaxis: { color:'#333', gridcolor:'#1a1a1a', title:{ text:'Distance (m)', font:{color:'#555'} } },
  yaxis: { color:'#333', gridcolor:'#1a1a1a', title:{ text:yTitle,         font:{color:'#555'} } },
  margin: { t:8, b:44, l:52, r:12 },
  legend: { bgcolor:'rgba(0,0,0,0)', font:{color:'#888'} },
  hovermode: 'x unified',
});

function rollingAvg(arr, w) {
  const h = Math.floor(w / 2);
  return arr.map((_, i) => {
    const lo = Math.max(0, i-h), hi = Math.min(arr.length-1, i+h);
    let s = 0; for (let j = lo; j <= hi; j++) s += arr[j];
    return s / (hi - lo + 1);
  });
}

function renderSpeedChart(c) {
  const traces = [{
    x: c.opt.dist, y: rollingAvg(c.opt.v_kmh, 5),
    type:'scatter', mode:'lines', name:'Optimal',
    line:{ color:'#e10600', width:1.8 },
  }];
  if (c.pole_tel) traces.push({
    x: c.pole_tel.dist, y: c.pole_tel.v_kmh,
    type:'scatter', mode:'lines',
    name: `Pole · ${c.pole_tel.driver} ${c.pole_tel.lap_time}s`,
    line:{ color:'#4a9eff', width:1.2, dash:'dot' },
  });
  Plotly.react('chart-speed', traces, LAYOUT('Speed (km/h)'), { responsive:true, displayModeBar:false });
}

function renderDespChart(c) {
  Plotly.react('chart-desp', [{
    x: c.opt.dist, y: c.opt.n_desp,
    type:'scatter', mode:'lines', name:'Displacement',
    line:{ color:'#f0a500', width:1.2 },
    fill:'tozeroy', fillcolor:'rgba(240,165,0,0.06)',
  },{
    x: [c.opt.dist[0], c.opt.dist.at(-1)], y: [0, 0],
    type:'scatter', mode:'lines',
    line:{ color:'#444', width:1, dash:'dot' }, showlegend:false,
  }], LAYOUT('Lateral offset (m)'), { responsive:true, displayModeBar:false });
}

function renderStats(c) {
  document.getElementById('dash-title').textContent = c.name + ' · ' + c.year;
  document.getElementById('stat-length').textContent = (c.opt.length_m / 1000).toFixed(3) + ' km';
  document.getElementById('stat-vmin').innerHTML  = c.opt.v_min  + ' <span>km/h</span>';
  document.getElementById('stat-vmean').innerHTML = c.opt.v_mean + ' <span>km/h</span>';
  document.getElementById('stat-vmax').innerHTML  = c.opt.v_max  + ' <span>km/h</span>';
  if (c.diff_vs_pole != null) {
    const s = c.diff_vs_pole <= 0 ? '' : '+';
    document.getElementById('stat-gain').innerHTML = `<span>${s}${c.diff_vs_pole.toFixed(3)}s</span>`;
  } else {
    document.getElementById('stat-gain').textContent = '—';
  }
}

function renderCircuit(c) {
  if (!c.opt) return;
  renderStats(c);
  renderMap(c);
  renderSpeedChart(c);
  renderDespChart(c);
}

// ── Resizer ───────────────────────────────────────────────────────────────────
let dragging = false, startY = 0, startMapH = 380, startChartH = 260;
document.getElementById('resizer').addEventListener('mousedown', e => {
  dragging    = true;
  startY      = e.clientY;
  startMapH   = document.getElementById('map-container').offsetHeight;
  startChartH = document.getElementById('chart-speed').offsetHeight;
  document.body.style.cursor     = 'ns-resize';
  document.body.style.userSelect = 'none';
});
document.addEventListener('mousemove', e => {
  if (!dragging) return;
  const dy = e.clientY - startY;
  const mh = Math.max(120, startMapH + dy);
  const ch = Math.max(120, startChartH - dy);
  document.documentElement.style.setProperty('--map-h',    mh + 'px');
  document.documentElement.style.setProperty('--charts-h', ch + 'px');
  Plotly.relayout('chart-speed', { height: ch });
  Plotly.relayout('chart-desp',  { height: ch });
});
document.addEventListener('mouseup', () => {
  if (!dragging) return;
  dragging = false;
  document.body.style.cursor     = '';
  document.body.style.userSelect = '';
  if (deckInstance) deckInstance.redraw();
});
