# =============================================================================
# PART 8 OF 9: PERIODIC-TABLE METADATA AND EMBEDDED WEB TEMPLATES
# =============================================================================
#
# Contains the periodic-table layout plus the HTML, CSS, and JavaScript for the
# selection screen, progress display, result navigation, and safe error page.
# Keeping presentation code here makes the scientific layers easier to read.
#
from __future__ import annotations

PERIOD_ROWS = {
    1: [("H", 1), ("He", 18)],
    2: [("Li", 1), ("Be", 2), ("B", 13), ("C", 14), ("N", 15), ("O", 16), ("F", 17), ("Ne", 18)],
    3: [("Na", 1), ("Mg", 2), ("Al", 13), ("Si", 14), ("P", 15), ("S", 16), ("Cl", 17), ("Ar", 18)],
    4: [(s, g) for s, g in zip(("K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr"), range(1,19))],
    5: [(s, g) for s, g in zip(("Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe"), range(1,19))],
    6: [("Cs",1),("Ba",2)] + [(s,g) for s,g in zip(("Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn"), range(4,19))],
    7: [("Fr",1),("Ra",2)] + [(s,g) for s,g in zip(("Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"), range(4,19))],
}
LANTHANIDES = ("La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu")
ACTINIDES = ("Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr")


def element_category(symbol: str, group: int, f_block: bool = False) -> str:
    if f_block:
        return "f"
    if symbol == "He" or group <= 2:
        return "s"
    if 3 <= group <= 12:
        return "d"
    return "p"


def periodic_table_tiles() -> list[dict[str, object]]:
    tiles: list[dict[str, object]] = []
    for period, entries in PERIOD_ROWS.items():
        for symbol, group in entries:
            tiles.append(
                {
                    "symbol": symbol,
                    "z": ATOMIC_NUMBERS[symbol],
                    "row": period,
                    "group": group,
                    "category": element_category(symbol, group),
                    "groundSpin": GROUND_STATE_2S[symbol],
                }
            )
    for index, symbol in enumerate(LANTHANIDES):
        tiles.append(
            {
                "symbol": symbol,
                "z": ATOMIC_NUMBERS[symbol],
                "row": 9,
                "group": index + 4,
                "category": "f",
                "groundSpin": GROUND_STATE_2S[symbol],
            }
        )
    for index, symbol in enumerate(ACTINIDES):
        tiles.append(
            {
                "symbol": symbol,
                "z": ATOMIC_NUMBERS[symbol],
                "row": 10,
                "group": index + 4,
                "category": "f",
                "groundSpin": GROUND_STATE_2S[symbol],
            }
        )
    return tiles


HOME_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DFT Atomic Orbital Explorer</title>
  <style>
    :root {
      --ink: #142033;
      --muted: #66758c;
      --bg: #f4f7fb;
      --panel: #ffffff;
      --s: #ff9b9b;
      --p: #e8ef72;
      --d: #86bce9;
      --f: #88df94;
      --accent: #315efb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    }
    .wrap { max-width: 1500px; margin: auto; padding: 28px; }
    .hero {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 24px;
      margin-bottom: 24px;
    }
    .eyebrow {
      color: var(--accent);
      font-size: .75rem;
      font-weight: 800;
      letter-spacing: .13em;
      text-transform: uppercase;
    }
    .hero h1 {
      margin: 8px 0 7px;
      font-size: clamp(2rem, 4vw, 3.6rem);
      letter-spacing: -.04em;
    }
    .hero p { max-width: 780px; margin: 0; color: var(--muted); line-height: 1.55; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px; }
    .legend span {
      padding: 7px 10px;
      border: 1px solid rgba(0, 0, 0, .12);
      border-radius: 999px;
      font-size: .75rem;
      font-weight: 700;
    }
    .table-shell {
      overflow: auto;
      padding: 18px;
      background: var(--panel);
      border: 1px solid #dce4ef;
      border-radius: 18px;
      box-shadow: 0 18px 50px rgba(25, 41, 66, .08);
    }
    .periodic {
      min-width: 1130px;
      display: grid;
      grid-template-columns: 48px repeat(18, minmax(52px, 1fr));
      grid-template-rows: 25px repeat(7, 68px) 24px repeat(2, 68px);
      gap: 5px;
    }
    .group, .period, .series-label { color: #78869a; font-size: .7rem; text-align: center; }
    .group { align-self: end; }
    .period, .series-label { align-self: center; }
    .series-label { line-height: 1.15; }
    .tile {
      position: relative;
      padding: 5px;
      border: 1px solid rgba(12, 28, 47, .35);
      border-radius: 7px;
      cursor: pointer;
      box-shadow: inset 0 1px rgba(255, 255, 255, .65);
      transition: transform .16s ease, box-shadow .16s ease;
    }
    .tile:hover, .tile.selected {
      z-index: 5;
      transform: scale(1.13);
      border-color: #101b2d;
      box-shadow: 0 10px 24px rgba(21, 42, 74, .23);
    }
    .tile .z { font-size: .65rem; }
    .tile .symbol { font-size: 1.2rem; font-weight: 800; line-height: 1; text-align: center; }
    .tile.s, .selected-card.s { background: var(--s); }
    .tile.p, .selected-card.p { background: var(--p); }
    .tile.d, .selected-card.d { background: var(--d); }
    .tile.f, .selected-card.f { background: var(--f); }
    .config {
      display: none;
      grid-template-columns: 190px minmax(0, 1fr);
      gap: 24px;
      margin-top: 18px;
      padding: 22px;
      color: white;
      background: #111d30;
      border-radius: 18px;
    }
    .config.open { display: grid; }
    .selected-card {
      min-height: 178px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 18px;
      color: #112033;
      border-radius: 14px;
    }
    .selected-card .big-z { align-self: flex-start; font-size: .9rem; }
    .selected-card .big-symbol { font-size: 4rem; font-weight: 850; line-height: 1; }
    .selected-card .big-name { margin-top: 5px; font-size: .82rem; opacity: .7; }
    .form-heading { grid-column: 1 / -1; margin-bottom: 2px; }
    .form-heading h2 { margin: 0 0 4px; font-size: 1.15rem; }
    .form-heading p { margin: 0; color: #9fb0c7; font-size: .82rem; }
    .fields { display: grid; grid-template-columns: repeat(3, minmax(150px, 1fr)); gap: 14px; }
    .field label { display: block; margin-bottom: 6px; color: #b5c2d5; font-size: .76rem; }
    .field input, .field select {
      width: 100%;
      min-height: 43px;
      padding: 10px;
      color: white;
      background: #19283e;
      border: 1px solid #40516b;
      border-radius: 8px;
      font: inherit;
    }
    .field input:focus, .field select:focus {
      outline: 2px solid #7291ff;
      outline-offset: 1px;
    }
    .note, .spin-help, .quality-control, .status, .calculate { grid-column: 1 / -1; }
    .spin-help { min-height: 1.25em; color: #b8c7dc; font-size: .76rem; }
    .note { color: #94a6bf; font-size: .76rem; line-height: 1.5; }
    .quality-control {
      margin-top: 2px;
      padding: 16px;
      background: #17263b;
      border: 1px solid #40516b;
      border-radius: 11px;
    }
    .quality-heading {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 14px;
      margin-bottom: 10px;
    }
    .quality-heading label { color: white; font-size: .88rem; font-weight: 800; }
    .quality-heading span { color: #9fb0c7; font-size: .76rem; text-align: right; }
    .quality-control input[type="range"] {
      width: 100%;
      min-height: 28px;
      margin: 0;
      padding: 0;
      accent-color: #7291ff;
      cursor: pointer;
    }
    .quality-slider-shell {
      position: relative;
      padding-top: 21px;
    }
    .quality-ticks {
      position: absolute;
      inset: 0 0 auto;
      height: 19px;
      color: #91a3bd;
      font-size: .68rem;
      line-height: 1;
      pointer-events: none;
    }
    .quality-ticks span {
      position: absolute;
      top: 0;
      white-space: nowrap;
      transform: translateX(-50%);
    }
    .quality-ticks span:nth-child(1) { left: 0; transform: none; }
    .quality-ticks span:nth-child(2) { left: 25%; }
    .quality-ticks span:nth-child(3) { left: 50%; }
    .quality-ticks span:nth-child(4) { left: 75%; }
    .quality-ticks span:nth-child(5) { right: 0; transform: none; }
    .quality-readout {
      margin-top: 11px;
      color: #dce6f4;
      font-size: .79rem;
      line-height: 1.5;
    }
    .time-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(105px, 1fr));
      gap: 7px;
      margin-top: 12px;
    }
    .time-card {
      width: 100%;
      padding: 9px;
      color: #9fb0c7;
      background: #111d30;
      border: 1px solid #34455e;
      border-radius: 8px;
      font: inherit;
      font-size: .67rem;
      line-height: 1.35;
      text-align: left;
      cursor: pointer;
    }
    .time-card strong { display: block; margin-top: 2px; color: #e6edf7; font-size: .75rem; }
    .time-card:hover { border-color: #7291ff; background: #182945; }
    .time-card:focus-visible { outline: 2px solid #9db1ff; outline-offset: 2px; }
    .time-card.selected { border-color: #7291ff; box-shadow: inset 0 0 0 1px #7291ff; }
    .time-note { margin: 9px 0 0; color: #8799b3; font-size: .68rem; line-height: 1.4; }
    .status {
      display: none;
      padding: 11px 13px;
      color: #d5e1f1;
      background: #1b2d47;
      border-radius: 8px;
    }
    .status.show { display: block; }
    .calculate {
      padding: 13px;
      color: white;
      background: var(--accent);
      border: 0;
      border-radius: 9px;
      cursor: pointer;
      font-weight: 800;
    }
    .calculate:disabled { cursor: wait; opacity: .55; }
    body.rendering { overflow: hidden; }
    .render-overlay {
      position: fixed;
      inset: 0;
      z-index: 1000;
      display: none;
      place-items: center;
      padding: 22px;
      background: rgba(8, 17, 31, .72);
      backdrop-filter: blur(5px);
    }
    .render-overlay.show { display: grid; }
    .render-progress-card {
      width: min(520px, 100%);
      padding: 27px;
      color: var(--ink);
      background: white;
      border: 1px solid #dce4ef;
      border-radius: 16px;
      box-shadow: 0 24px 70px rgba(5, 14, 29, .34);
    }
    .render-progress-card h2 { margin: 0 0 7px; font-size: 1.2rem; }
    .render-progress-card p { margin: 0; color: var(--muted); line-height: 1.5; }
    .render-progress-card progress {
      width: 100%;
      height: 16px;
      margin: 20px 0 12px;
      accent-color: var(--accent);
    }
    .render-progress-meta { display: flex; justify-content: space-between; gap: 16px; }
    .render-elapsed { color: #60718a; font-size: .78rem; }
    .render-percent { color: var(--accent); font-size: .82rem; font-weight: 800; }
    .render-note { margin-top: 5px; color: #8795a8; font-size: .72rem; }
    @media (max-width: 900px) {
      .hero { align-items: flex-start; flex-direction: column; }
      .config.open { grid-template-columns: 1fr; }
      .selected-card { min-height: 120px; }
      .fields { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
      .time-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    }
    @media (max-width: 560px) {
      .wrap { padding: 16px; }
      .fields { grid-template-columns: 1fr; }
      .time-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <div>
        <div class="eyebrow">Kohn–Sham DFT laboratory</div>
        <h1>Choose an atom. Explore its orbitals.</h1>
        <p>
          Select an element, isotope, charge, spin and the three spatial quantum
          numbers. Heavy elements automatically use an all-electron scalar-relativistic
          calculation. Previously calculated atomic fields and common orbitals are
          reused automatically. Use the five-position accuracy slider to trade
          execution time against DFT and spatial resolution.
        </p>
      </div>
      <div class="legend" aria-label="Periodic-table block colours">
        <span style="background: var(--s)">s block</span>
        <span style="background: var(--d)">d block</span>
        <span style="background: var(--p)">p block</span>
        <span style="background: var(--f)">f block</span>
      </div>
    </header>

    <section class="table-shell" aria-label="Periodic table">
      <div class="periodic" id="periodic"></div>
    </section>

    <form class="config" id="config" method="post" action="/calculate">
      <div class="selected-card" id="selectedCard">
        <div class="big-z" id="bigZ"></div>
        <div class="big-symbol" id="bigSymbol"></div>
        <div class="big-name">selected element</div>
      </div>

      <div class="fields">
        <div class="form-heading">
          <h2>Calculation settings</h2>
          <p>Spin is written as 2<i>S</i>; multiplicity is 2<i>S</i> + 1.</p>
        </div>
        <input type="hidden" name="symbol" id="symbolInput">

        <div class="field">
          <label for="isotope">Isotope mass number, <i>A</i></label>
          <input name="isotope" id="isotope" type="number" min="1" required>
        </div>
        <div class="field">
          <label for="charge">Ionic charge</label>
          <input name="charge" id="charge" type="number" value="0" required>
        </div>
        <div class="field">
          <label for="spin">Total spin, 2<i>S</i></label>
          <select name="spin" id="spin" required></select>
        </div>
        <div class="spin-help" id="spinHelp"></div>

        <div class="field">
          <label for="n">Principal quantum number, <i>n</i></label>
          <input name="n" id="n" type="number" min="1" max="7" value="1" required>
        </div>
        <div class="field">
          <label for="l">Orbital quantum number, <i>ℓ</i></label>
          <select name="l" id="l"></select>
        </div>
        <div class="field">
          <label for="m">Magnetic quantum number, <i>m</i></label>
          <select name="m" id="m"></select>
        </div>

        <div class="quality-control">
          <div class="quality-heading">
            <label for="quality">Accuracy and spatial resolution</label>
            <span id="qualityName">Level 3 · Balanced</span>
          </div>
          <div class="quality-slider-shell">
            <div class="quality-ticks" aria-hidden="true">
              <span>Preview</span><span>Fast</span><span>Balanced</span><span>Accurate</span><span>Maximum</span>
            </div>
            <input name="quality" id="quality" type="range" min="1" max="5" step="1" value="3">
          </div>
          <div class="quality-readout" id="qualityReadout"></div>
          <div class="time-grid" id="timeGrid" aria-live="polite"></div>
          <p class="time-note" id="timeNote"></p>
        </div>

        <div class="note">
          The isotope is recorded in the output. The electronic calculation uses
          clamped nuclei, so isotope mass does not materially change the orbital.
          The spin menu contains only values giving integer
          <i>N</i><sub>α</sub> and <i>N</i><sub>β</sub> populations for the
          selected ion. Each requested spin and <i>ℓ</i> family up to
          <i>n</i> ≤ 7 is cached after its first use. Runtime ranges account for
          uncertain SCF iteration counts; they are predictions rather than
          guarantees. F-block and superheavy elements use the deterministic
          preview at levels 1–3; levels 4–5 attempt fixed-ensemble SCF before
          falling back safely. Even the maximum setting remains an LDA visualization,
          not a benchmark-quality atomic-structure calculation.
        </div>
        <div class="status" id="status" role="status">
          Loading a cached result or calculating the self-consistent DFT density.
          Heavy-element attempts that hit an occupation boundary now continue to a
          safe fallback instead of returning an index error.
        </div>
        <button class="calculate" id="calculate" type="submit">Calculate · Balanced</button>
      </div>
    </form>
  </div>

  <div class="render-overlay" id="renderOverlay" aria-hidden="true">
    <div class="render-progress-card" role="status" aria-live="polite">
      <h2>Rendering orbital</h2>
      <p id="renderMessage">Preparing the atomic density, orbital grid and initial view.</p>
      <progress id="renderProgress" max="100" value="0" aria-label="Estimated orbital rendering progress">0%</progress>
      <div class="render-progress-meta">
        <div class="render-elapsed" id="renderElapsed">Elapsed: 0.0 s</div>
        <div class="render-percent" id="renderPercent">0.0%</div>
      </div>
      <div class="render-note">Estimated from this laptop's predicted runtime; 100% appears only after completion.</div>
    </div>
  </div>

  <script>
    const tiles = {{ tiles | safe }};
    const table = document.getElementById('periodic');
    const config = document.getElementById('config');
    const chargeInput = document.getElementById('charge');
    const spinSelect = document.getElementById('spin');
    const spinHelp = document.getElementById('spinHelp');
    const calculateButton = document.getElementById('calculate');
    const qualitySlider = document.getElementById('quality');
    const qualityName = document.getElementById('qualityName');
    const qualityReadout = document.getElementById('qualityReadout');
    const timeGrid = document.getElementById('timeGrid');
    const timeNote = document.getElementById('timeNote');
    const renderOverlay = document.getElementById('renderOverlay');
    const renderMessage = document.getElementById('renderMessage');
    const renderProgress = document.getElementById('renderProgress');
    const renderElapsed = document.getElementById('renderElapsed');
    const renderPercent = document.getElementById('renderPercent');
    const qualityProfiles = {{ quality_profiles | safe }};
    const runtimeCalibration = {{ runtime_calibration }};
    const calibrationSamples = {{ calibration_samples }};
    const orbitalLetters = ['s', 'p', 'd', 'f', 'g', 'h', 'i'];
    let selected = null;
    let renderingSubmitted = false;
    let renderTimer = null;

    for (let group = 1; group <= 18; group += 1) {
      const heading = document.createElement('div');
      heading.className = 'group';
      heading.style.gridColumn = group + 1;
      heading.style.gridRow = 1;
      heading.textContent = group;
      table.appendChild(heading);
    }
    for (let period = 1; period <= 7; period += 1) {
      const heading = document.createElement('div');
      heading.className = 'period';
      heading.style.gridColumn = 1;
      heading.style.gridRow = period + 1;
      heading.textContent = period;
      table.appendChild(heading);
    }
    [['La–Lu', 10], ['Ac–Lr', 11]].forEach(([label, row]) => {
      const heading = document.createElement('div');
      heading.className = 'series-label';
      heading.style.gridColumn = 1;
      heading.style.gridRow = row;
      heading.textContent = label;
      table.appendChild(heading);
    });

    tiles.forEach(tile => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `tile ${tile.category}`;
      button.style.gridColumn = tile.group + 1;
      button.style.gridRow = tile.row + 1;
      button.setAttribute('aria-label', `${tile.symbol}, atomic number ${tile.z}`);
      button.innerHTML = `<div class="z">${tile.z}</div><div class="symbol">${tile.symbol}</div>`;
      button.addEventListener('click', () => selectElement(tile, button));
      table.appendChild(button);
    });

    function physicallyAllowedSpins(electronCount) {
      const first = electronCount % 2;
      const values = [];
      for (let value = first; value <= electronCount; value += 2) values.push(value);
      return values;
    }

    function nearestAllowed(values, target) {
      return values.reduce((best, value) => {
        const difference = Math.abs(value - target);
        const bestDifference = Math.abs(best - target);
        return difference < bestDifference ||
          (difference === bestDifference && value > best) ? value : best;
      }, values[0]);
    }

    function formatDuration(seconds) {
      if (!Number.isFinite(seconds) || seconds <= 0) return 'unavailable';
      if (seconds < 90) return `${seconds.toFixed(1)} s`;
      if (seconds < 5400) {
        const minutes = seconds / 60;
        return `${minutes.toFixed(1)} min`;
      }
      const hours = seconds / 3600;
      return `${hours.toFixed(1)} h`;
    }

    function modelRuntimeSeconds(profile, qualityLevel, electronCount, spin, charge) {
      if (!selected || electronCount < 1) return Number.NaN;
      const electrons = Math.max(electronCount, 1);
      const surfaceRatio = Math.pow(Number(profile.surface_grid_points) / 81, 3);
      const radialRatio = Math.pow(Number(profile.radial_points) / 1600, 0.45);
      const initialRenderSeconds = 0.16 + 0.14 * surfaceRatio + 0.05 * radialRatio;
      const previewElement = selected.z >= 100
        || (selected.z >= 57 && selected.z <= 71)
        || (selected.z >= 89 && selected.z <= 103);
      const optimizedPreview = previewElement && qualityLevel <= 3;
      const ionFactor = 1.0 + 0.05 * Math.min(Math.abs(charge), 4);
      let densitySeconds;
      if (optimizedPreview) {
        const atomFactor = 0.82 + 0.0025 * electrons;
        const previewGridFactor = Math.pow(
          Number(profile.heavy_preview_grid_level) + 1,
          1.35,
        );
        const previewSeconds = 0.08
          + 0.12 * Number(profile.heavy_preview_fock_updates)
          * previewGridFactor * atomFactor * ionFactor;
        densitySeconds = Math.min(
          0.55 * Number(profile.heavy_preview_time_budget_seconds),
          previewSeconds,
        );
      } else {
        const electronicCost = 0.12 + 0.009 * Math.pow(electrons, 1.35);
        const gridFactor = Math.pow(
          Math.max(Number(profile.dft_grid_level), 1) / 3,
          1.45,
        );
        const cycleFactor = Math.pow(
          Math.max(Number(profile.scf_max_cycles), 80) / 180,
          0.30,
        );
        const openShellFactor = 1.0 + 0.025 * Math.min(Math.abs(spin), 8);
        const relativisticFactor = selected.z >= 37 ? 1.20 : 1.0;
        densitySeconds = electronicCost * gridFactor * cycleFactor
          * openShellFactor * ionFactor * relativisticFactor;
      }
      return Math.max(0.20, initialRenderSeconds + densitySeconds)
        * runtimeCalibration;
    }

    function estimatedProgressPercent(elapsedSeconds, expectedSeconds) {
      if (!Number.isFinite(expectedSeconds) || expectedSeconds <= 0) {
        return Math.min(99, 100 * (1 - Math.exp(-elapsedSeconds / 20)));
      }
      if (elapsedSeconds <= expectedSeconds) {
        return 90 * elapsedSeconds / expectedSeconds;
      }
      const overrunScale = Math.max(1, 0.5 * expectedSeconds);
      return Math.min(
        99,
        90 + 9 * (1 - Math.exp(-(elapsedSeconds - expectedSeconds) / overrunScale)),
      );
    }

    function setRenderProgress(percent) {
      const bounded = Math.min(100, Math.max(0, Number(percent) || 0));
      const label = bounded === 100 ? '100%' : `${bounded.toFixed(1)}%`;
      renderProgress.value = bounded;
      renderProgress.textContent = label;
      renderProgress.setAttribute('aria-valuenow', bounded.toFixed(1));
      renderProgress.setAttribute('aria-valuetext', label);
      renderPercent.textContent = label;
    }

    function updateQualityDisplay() {
      const level = Number.parseInt(qualitySlider.value, 10);
      const profile = qualityProfiles[String(level)];
      qualityName.textContent = `Level ${level} · ${profile.name}`;
      qualityReadout.innerHTML =
        `${profile.description}. DFT grid level ${profile.dft_grid_level}; ` +
        `SCF tolerance ${Number(profile.scf_tolerance).toExponential(0)}; ` +
        `${Number(profile.radial_points).toLocaleString()} radial points; ` +
        `${profile.angular_directions} angular directions; ` +
        `${profile.surface_grid_points}<sup>3</sup> surface grid.`;
      calculateButton.textContent = `Calculate · ${profile.name}`;
      updateRuntimeEstimates();
    }

    function updateRuntimeEstimates() {
      if (!selected) return;
      const charge = Number.parseInt(chargeInput.value, 10);
      const electronCount = selected.z - charge;
      const spin = Number.parseInt(spinSelect.value, 10);
      const selectedLevel = Number.parseInt(qualitySlider.value, 10);
      timeGrid.innerHTML = '';

      for (let level = 1; level <= 5; level += 1) {
        const profile = qualityProfiles[String(level)];
        const centre = modelRuntimeSeconds(
          profile,
          level,
          electronCount,
          spin,
          charge,
        );
        const card = document.createElement('button');
        card.type = 'button';
        card.className = `time-card${level === selectedLevel ? ' selected' : ''}`;
        card.setAttribute('aria-pressed', String(level === selectedLevel));
        card.setAttribute('aria-label', `Select level ${level}, ${profile.name}`);
        const optimizedPreview = level <= 3 && (selected.z >= 100
          || (selected.z >= 57 && selected.z <= 71)
          || (selected.z >= 89 && selected.z <= 103));
        const lowerMultiplier = optimizedPreview ? 0.75 : 0.65;
        const upperMultiplier = optimizedPreview ? 1.50 : 2.25;
        const range = Number.isFinite(centre)
          ? `${formatDuration(lowerMultiplier * centre)}–${formatDuration(upperMultiplier * centre)}`
          : 'select a physical ion';
        card.innerHTML = `${level}. ${profile.name}<strong>${range}</strong>`;
        card.addEventListener('click', () => {
          qualitySlider.value = String(level);
          qualitySlider.dispatchEvent(new Event('input', { bubbles: true }));
        });
        timeGrid.appendChild(card);
      }

      const calibrationText = calibrationSamples > 0
        ? `Calibrated from ${calibrationSamples} successful uncached run${calibrationSamples === 1 ? '' : 's'} on this laptop.`
        : 'Initial hardware estimate; the range will calibrate itself after successful uncached runs on this laptop.';
      timeNote.textContent =
        `Predicted optimized first-run wall time. ${calibrationText} Exact cached results normally open in under two seconds.`;
    }

    function updateSpinOptions(preferredValue = null) {
      if (!selected) return;
      const charge = Number.parseInt(chargeInput.value, 10);
      const electronCount = selected.z - charge;
      const previousValue = Number.parseInt(spinSelect.value, 10);
      spinSelect.innerHTML = '';

      if (!Number.isInteger(electronCount) || electronCount < 1) {
        spinSelect.add(new Option('No physical electron population', ''));
        spinSelect.disabled = true;
        calculateButton.disabled = true;
        spinHelp.textContent = 'Reduce the positive charge: the ion must retain at least one electron.';
        updateRuntimeEstimates();
        return;
      }

      const values = physicallyAllowedSpins(electronCount);
      const target = Number.isInteger(preferredValue)
        ? preferredValue
        : (Number.isInteger(previousValue) ? previousValue : selected.groundSpin);
      const recommended = nearestAllowed(values, target);
      values.forEach(value => {
        const suffix = value === recommended ? ' — recommended' : '';
        spinSelect.add(new Option(
          `2S = ${value} (multiplicity ${value + 1})${suffix}`,
          String(value),
          false,
          value === recommended,
        ));
      });
      spinSelect.disabled = false;
      calculateButton.disabled = false;
      spinHelp.textContent = `${electronCount} electrons: allowed 2S values have the same parity as the electron count.`;
      updateRuntimeEstimates();
    }

    function selectElement(tile, button) {
      document.querySelectorAll('.tile').forEach(item => item.classList.remove('selected'));
      button.classList.add('selected');
      selected = tile;
      config.classList.add('open');
      document.getElementById('bigZ').textContent = tile.z;
      document.getElementById('bigSymbol').textContent = tile.symbol;
      document.getElementById('selectedCard').className = `selected-card ${tile.category}`;
      document.getElementById('symbolInput').value = tile.symbol;

      const isotopeInput = document.getElementById('isotope');
      isotopeInput.min = tile.z;
      isotopeInput.value = Math.max(tile.z, Math.round(tile.z * 2.35));
      chargeInput.value = 0;
      chargeInput.max = tile.z - 1;
      updateSpinOptions(tile.groundSpin);

      const period = tile.row > 7 ? tile.row - 3 : tile.row;
      document.getElementById('n').value = Math.min(7, Math.max(1, period));
      updateL();
      updateQualityDisplay();
      config.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function updateL() {
      const n = Number(document.getElementById('n').value);
      const lSelect = document.getElementById('l');
      lSelect.innerHTML = '';
      for (let l = 0; l < n; l += 1) {
        lSelect.add(new Option(`${l} (${orbitalLetters[l]})`, String(l)));
      }
      lSelect.value = String(Math.min(n - 1, 1));
      updateM();
    }

    function updateM() {
      const l = Number(document.getElementById('l').value);
      const mSelect = document.getElementById('m');
      mSelect.innerHTML = '';
      for (let m = -l; m <= l; m += 1) mSelect.add(new Option(String(m), String(m)));
      mSelect.value = '0';
    }

    chargeInput.addEventListener('input', () => updateSpinOptions());
    spinSelect.addEventListener('change', () => {
      spinSelect.dataset.userSelection = spinSelect.value;
      updateRuntimeEstimates();
    });
    qualitySlider.addEventListener('input', updateQualityDisplay);
    document.getElementById('n').addEventListener('change', updateL);
    document.getElementById('l').addEventListener('change', updateM);
    config.addEventListener('submit', async event => {
      event.preventDefault();
      if (renderingSubmitted) return;
      renderingSubmitted = true;
      document.getElementById('status').classList.add('show');
      const profile = qualityProfiles[String(Number.parseInt(qualitySlider.value, 10))];
      document.getElementById('status').textContent =
        `Running the ${profile.name} profile. The measured time will be saved to improve future laptop estimates.`;
      renderMessage.textContent =
        `Running the ${profile.name} profile for ${selected.symbol}. ` +
        'Preparing the density, orbital grid and initial view.';
      renderOverlay.classList.add('show');
      renderOverlay.setAttribute('aria-hidden', 'false');
      document.body.classList.add('rendering');
      calculateButton.disabled = true;
      const renderStart = performance.now();
      const charge = Number.parseInt(chargeInput.value, 10);
      const electronCount = selected.z - charge;
      const spin = Number.parseInt(spinSelect.value, 10);
      const expectedSeconds = modelRuntimeSeconds(
        profile,
        Number.parseInt(qualitySlider.value, 10),
        electronCount,
        spin,
        charge,
      );
      setRenderProgress(0);
      renderTimer = window.setInterval(() => {
        const seconds = (performance.now() - renderStart) / 1000;
        renderElapsed.textContent = `Elapsed: ${seconds.toFixed(1)} s`;
        setRenderProgress(estimatedProgressPercent(seconds, expectedSeconds));
      }, 100);

      // Keep this document alive while Flask calculates so the elapsed timer
      // continues to repaint.  Request only the completed result URL; loading
      // the Plotly-heavy result through normal navigation avoids buffering and
      // rewriting the entire HTML document in JavaScript.
      await new Promise(resolve => {
        window.requestAnimationFrame(() => window.requestAnimationFrame(resolve));
      });
      try {
        const response = await window.fetch(config.action, {
          method: config.method || 'POST',
          body: new FormData(config),
          credentials: 'same-origin',
          headers: { 'X-Orbital-Progress': '1' },
        });
        if (!response.ok) {
          const errorHtml = await response.text();
          if (renderTimer !== null) window.clearInterval(renderTimer);
          renderTimer = null;
          document.open();
          document.write(errorHtml);
          document.close();
          return;
        }
        const resultInfo = await response.json();
        if (!resultInfo.result_url) throw new Error('The result URL was not returned.');
        if (renderTimer !== null) window.clearInterval(renderTimer);
        renderTimer = null;
        setRenderProgress(100);
        renderMessage.textContent = 'Rendering complete. Opening the result.';
        await new Promise(resolve => {
          window.requestAnimationFrame(() => window.requestAnimationFrame(resolve));
        });
        window.location.replace(resultInfo.result_url);
      } catch (error) {
        if (renderTimer !== null) window.clearInterval(renderTimer);
        renderTimer = null;
        renderingSubmitted = false;
        renderOverlay.classList.remove('show');
        renderOverlay.setAttribute('aria-hidden', 'true');
        document.body.classList.remove('rendering');
        setRenderProgress(0);
        calculateButton.disabled = spinSelect.disabled;
        document.getElementById('status').textContent =
          `The calculation request failed: ${error.message}`;
      }
    });
    window.addEventListener('pageshow', () => {
      if (renderTimer !== null) window.clearInterval(renderTimer);
      renderTimer = null;
      renderingSubmitted = false;
      renderOverlay.classList.remove('show');
      renderOverlay.setAttribute('aria-hidden', 'true');
      document.body.classList.remove('rendering');
      renderElapsed.textContent = 'Elapsed: 0.0 s';
      setRenderProgress(0);
      if (selected) calculateButton.disabled = spinSelect.disabled;
    });
    updateQualityDisplay();
  </script>
</body>
</html>'''


ERROR_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Calculation could not be completed</title>
  <style>
    * { box-sizing: border-box; }
    body {
      min-height: 100vh;
      display: grid;
      place-items: center;
      margin: 0;
      padding: 24px;
      color: #142033;
      background: #f4f7fb;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    }
    .card {
      width: min(720px, 100%);
      padding: 30px;
      background: white;
      border: 1px solid #dce4ef;
      border-radius: 16px;
      box-shadow: 0 18px 50px rgba(25, 41, 66, .10);
    }
    .label {
      color: #b33b3b;
      font-size: .75rem;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
    }
    h1 { margin: 8px 0 10px; font-size: 1.8rem; }
    p { color: #5f6f86; line-height: 1.6; }
    .details {
      margin: 18px 0;
      padding: 14px;
      overflow-wrap: anywhere;
      color: #25354c;
      background: #f2f5f9;
      border-radius: 9px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: .84rem;
      line-height: 1.5;
    }
    a {
      display: inline-block;
      padding: 10px 14px;
      color: white;
      background: #315efb;
      border-radius: 8px;
      font-weight: 750;
      text-decoration: none;
    }
  </style>
</head>
<body>
  <main class="card">
    <div class="label">{{ error_type }}</div>
    <h1>Calculation could not be completed</h1>
    <p>Review the selected ion, spin and quantum numbers, then try again.</p>
    <div class="details">{{ message }}</div>
    <a href="/">Return to the periodic table</a>
  </main>
</body>
</html>'''


