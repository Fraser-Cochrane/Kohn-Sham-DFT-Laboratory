# =============================================================================
# PART 7 OF 9: RESULT HTML, DATA EXPORT, AND END-TO-END CALCULATION PIPELINE
# =============================================================================
#
# Writes the interactive result page, exports radial data, coordinates all
# physical cache layers, prepares lazy tabs, and reports the completed runtime.
# This is the bridge between the numerical engine and the Flask routes.
#
from __future__ import annotations

def write_results_website(
    figures: dict[str, tuple[str, str, go.Figure | None]],
    dft_energy: float,
    orbital_energy: float,
    achieved_fraction: float,
    calculation_info: dict[str, object],
    output_path: Path,
    cache_summary: str,
    compute_seconds: float,
    visual_key: str,
) -> Path:
    """Write a result page whose inactive plots are loaded on first use."""
    plot_config = {
        "displaylogo": False,
        "scrollZoom": True,
        "responsive": True,
        "toImageButtonOptions": {"format": "png", "scale": 2},
    }
    fragments: dict[str, str] = {}
    for index, (panel_id, (_label, _description, figure)) in enumerate(figures.items()):
        if figure is not None:
            fragments[panel_id] = to_html(
                figure,
                full_html=False,
                include_plotlyjs=False,
                include_mathjax="cdn" if index == 0 else False,
                config=plot_config,
                default_width="100%",
                default_height="790px",
            )

    menu_html = "\n".join(
        f'<button class="menu-button{" active" if index == 0 else ""}" '
        f'data-panel="{panel_id}">{label}</button>'
        for index, (panel_id, (label, _description, _figure)) in enumerate(figures.items())
    )
    panels: list[str] = []
    for index, (panel_id, (label, description, figure)) in enumerate(figures.items()):
        loaded = figure is not None
        plot_markup = fragments.get(
            panel_id,
            (
                '<div class="lazy-plot" data-plot-host>'
                '<div class="loading-state">Open this tab to generate its cached view.</div>'
                '</div>'
            ),
        )
        panels.append(
            f'<section id="panel-{panel_id}" '
            f'class="view-panel{" active" if index == 0 else ""}" '
            f'data-loaded="{"true" if loaded else "false"}">'
            f'<div class="panel-heading"><h2>{label}</h2><p>{description}</p></div>'
            f'{plot_markup}</section>'
        )
    panels_html = "\n".join(panels)
    plotly_javascript = get_plotlyjs()
    plot_config_json = json.dumps(plot_config, separators=(",", ":"))
    element_symbol = parse_ion_name(ION)[0]
    isotope_html = (
        f"<sup>{ISOTOPE_MASS_NUMBER}</sup>{element_symbol}"
        if ISOTOPE_MASS_NUMBER is not None
        else element_symbol
    )
    energy_model = str(
        calculation_info.get("energy_model", "converged DFT total energy")
    )
    energy_label_latex = (
        r"E_{\mathrm{preview}}"
        if "preview" in energy_model.lower()
        else r"E_{\mathrm{DFT}}"
    )
    website = rf"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{ION} DFT orbital explorer</title>
<style>
:root {{ --ink:#122033; --muted:#65748b; --panel:#ffffff; --line:#dce4ee; --accent:#315efb; --bg:#f2f5f9; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--bg); }}
.app {{ min-height:100vh; display:grid; grid-template-columns:250px minmax(0,1fr); }}
aside {{ background:#101b2c; color:white; padding:26px 18px; position:sticky; top:0; height:100vh; overflow:auto; }}
.brand {{ font-size:1.28rem; font-weight:750; line-height:1.2; margin-bottom:7px; }}
.subtitle {{ color:#aebbd0; font-size:.88rem; line-height:1.45; margin-bottom:25px; }}
.menu-label {{ color:#8292aa; font-size:.72rem; font-weight:750; letter-spacing:.11em; text-transform:uppercase; margin:18px 10px 8px; }}
.menu-button {{ width:100%; border:0; border-radius:9px; background:transparent; color:#d8e0ec; text-align:left; padding:11px 12px; margin:3px 0; font-size:.92rem; cursor:pointer; transition:.16s ease; }}
.menu-button:hover {{ background:#1b2a40; color:white; }}
.menu-button.active {{ background:var(--accent); color:white; box-shadow:0 7px 18px rgba(49,94,251,.28); }}
.home-button {{ display:block; margin:20px 0 0; border:1px solid #40516a; border-radius:9px; padding:11px 12px; color:white; text-align:center; text-decoration:none; font-size:.9rem; font-weight:700; transition:.16s ease; }}
.home-button:hover {{ background:#1b2a40; border-color:#6d80a0; }}
.summary {{ margin-top:28px; border-top:1px solid #2b3a50; padding:18px 10px 0; color:#afbdd1; font-size:.78rem; line-height:1.65; }}
main {{ min-width:0; padding:28px; }}
.topbar {{ display:flex; justify-content:space-between; align-items:flex-start; gap:20px; margin:0 auto 20px; max-width:1320px; }}
.topbar h1 {{ margin:0 0 6px; font-size:1.55rem; }}
.topbar p {{ margin:0; color:var(--muted); }}
.metric-row {{ display:flex; gap:9px; flex-wrap:wrap; justify-content:flex-end; }}
.metric {{ background:white; border:1px solid var(--line); border-radius:9px; padding:8px 11px; white-space:nowrap; font-size:.78rem; color:var(--muted); }}
.metric strong {{ color:var(--ink); }}
.view-panel {{ display:none; max-width:1320px; margin:0 auto; background:var(--panel); border:1px solid var(--line); border-radius:14px; box-shadow:0 10px 30px rgba(24,39,64,.07); overflow:hidden; }}
.view-panel.active {{ display:block; }}
.panel-heading {{ padding:20px 24px 5px; }}
.panel-heading h2 {{ margin:0 0 5px; font-size:1.12rem; }}
.panel-heading p {{ margin:0; color:var(--muted); font-size:.88rem; }}
.lazy-plot {{ width:100%; min-height:650px; display:grid; place-items:center; }}
.loading-state {{ color:var(--muted); font-size:.9rem; padding:28px; text-align:center; }}
.loading-state.error {{ color:#a12b35; }}
.math {{ white-space:nowrap; }}
@media (max-width:850px) {{ .app {{ grid-template-columns:1fr; }} aside {{ position:relative; height:auto; }} main {{ padding:16px; }} .topbar {{ flex-direction:column; }} .metric-row {{ justify-content:flex-start; }} }}
</style>
<script>{plotly_javascript}</script>
</head>
<body>
<div class="app">
<aside>
  <div class="brand">Atomic orbital explorer</div>
  <div class="subtitle">
    {ION} · DFT/LDA ·
    <span class="math">\((n,\ell,m)=({N_QUANTUM},{L_QUANTUM},{M_QUANTUM})\)</span>
  </div>
  <div class="menu-label">Choose view</div>
  {menu_html}
  <a class="home-button" href="/">Return to main menu</a>
  <div class="summary">Drag 3D views to rotate.<br>Scroll to zoom.<br>Use each plot toolbar to export an image.</div>
</aside>
<main>
  <header class="topbar">
    <div>
      <h1>{ION} orbital results</h1>
      <p>
        Spin-resolved Kohn–Sham potential and spatial
        <span class="math">\(Z_{{\mathrm{{eff}}}}(r)\)</span>
      </p>
    </div>
    <div class="metric-row">
      <div class="metric">
        <span class="math">\({energy_label_latex}\)</span>
        <strong class="math">\({dft_energy:#.4g}\,E_{{\mathrm{{h}}}}\)</strong>
      </div>
      <div class="metric">
        <span class="math">\(E_{{n\ell}}\)</span>
        <strong class="math">\({orbital_energy:#.4g}\,E_{{\mathrm{{h}}}}\)</strong>
      </div>
      <div class="metric">Enclosed <strong>{100.0 * achieved_fraction:#.4g}%</strong></div>
      <div class="metric">Isotope <strong>{isotope_html}</strong></div>
      <div class="metric">Basis <strong>{calculation_info["basis"]}</strong></div>
      <div class="metric">Hamiltonian <strong>{calculation_info["hamiltonian"]}</strong></div>
      <div class="metric">Radial equation <strong>{calculation_info["radial_solver"]}</strong></div>
      <div class="metric">Energy model <strong>{energy_model}</strong></div>
      <div class="metric">Engine <strong>{calculation_info.get("engine_revision", "legacy")}</strong></div>
      <div class="metric">Occupations <strong>{calculation_info["occupations"]}</strong></div>
      <div class="metric">SCF status <strong>{calculation_info.get("convergence", "converged")}</strong></div>
      <div class="metric">Quality <strong>Level {calculation_info.get("quality_level", QUALITY_LEVEL)} · {calculation_info.get("quality_profile", QUALITY_PROFILE.lower())}</strong></div>
      <div class="metric">Compute time <strong>{compute_seconds:.1f} s</strong></div>
      <div class="metric">Cache <strong>{cache_summary}</strong></div>
    </div>
  </header>
  {panels_html}
</main>
</div>
<script>
const buttons = Array.from(document.querySelectorAll('.menu-button'));
const panels = Array.from(document.querySelectorAll('.view-panel'));
const plotConfig = {plot_config_json};
const lazyBase = '/api/result-tabs/{visual_key}/';
const panelLoads = new Map();
async function ensurePanelLoaded(panelId) {{
  const panel = document.getElementById(`panel-${{panelId}}`);
  if (!panel || panel.dataset.loaded === 'true') return;
  if (panelLoads.has(panelId)) return panelLoads.get(panelId);
  const task = (async () => {{
    const host = panel.querySelector('[data-plot-host]');
    if (!host) throw new Error('Plot container is unavailable.');
    host.innerHTML = '<div class="loading-state">Generating view from cached wavefunction data…</div>';
    try {{
      const response = await fetch(`${{lazyBase}}${{encodeURIComponent(panelId)}}`, {{cache:'force-cache'}});
      const figure = await response.json();
      if (!response.ok) throw new Error(figure.error || 'The view could not be generated.');
      host.innerHTML = '';
      await Plotly.newPlot(host, figure.data, figure.layout, plotConfig);
      panel.dataset.loaded = 'true';
      typesetMath(panel);
    }} catch (error) {{
      host.innerHTML = `<div class="loading-state error">${{error.message}}</div>`;
      throw error;
    }}
  }})();
  panelLoads.set(panelId, task);
  try {{ await task; }} finally {{ panelLoads.delete(panelId); }}
}}
function showPanel(panelId) {{
  buttons.forEach(button => button.classList.toggle('active', button.dataset.panel === panelId));
  panels.forEach(panel => panel.classList.toggle('active', panel.id === `panel-${{panelId}}`));
  ensurePanelLoaded(panelId).then(() => {{
    const graph = document.querySelector(`#panel-${{panelId}} .plotly-graph-div`);
    if (graph) requestAnimationFrame(() => Plotly.Plots.resize(graph));
  }}).catch(() => {{}});
  typesetMath(document.getElementById(`panel-${{panelId}}`));
  history.replaceState(null, '', `#${{panelId}}`);
}}
function typesetMath(root = document.body) {{
  if (!window.MathJax) return;
  if (typeof window.MathJax.typesetPromise === 'function') {{
    window.MathJax.typesetPromise([root]).catch(() => {{}});
  }} else if (window.MathJax.Hub) {{
    window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub, root]);
  }}
}}
buttons.forEach(button => button.addEventListener('click', () => showPanel(button.dataset.panel)));
const initial = location.hash.slice(1);
if (initial && document.getElementById(`panel-${{initial}}`)) {{ showPanel(initial); }}
window.addEventListener('resize', () => {{
  const graph = document.querySelector('.view-panel.active .plotly-graph-div');
  if (graph) Plotly.Plots.resize(graph);
}});
window.addEventListener('load', () => typesetMath(document.body));
</script>
</body>
</html>"""

    try:
        atomic_write_text(output_path, website)
    except OSError as exc:
        ion_slug = re.sub(r"[^A-Za-z0-9]+", "_", ION).strip("_")
        fallback = Path(
            f"{ion_slug}_DFT_orbital_explorer_"
            f"n{N_QUANTUM}_l{L_QUANTUM}_m{M_QUANTUM}.html"
        ).resolve()
        if fallback == output_path:
            raise
        print(f"Cache result directory was unavailable ({exc}); using {fallback}.")
        atomic_write_text(fallback, website)
        output_path = fallback
    prune_cache(exclude=output_path)
    return output_path


def save_radial_data(
    radial_grid: np.ndarray,
    density_alpha: np.ndarray,
    density_beta: np.ndarray,
    potentials: dict[str, np.ndarray],
    atomic_key: str,
) -> Path:
    """Save radial DFT density, potentials, and Z_eff(r) to CSV."""
    path = cache_file("radial-data", atomic_key, ".csv")
    if path is None:
        ion_slug = re.sub(r"[^A-Za-z0-9]+", "_", ION).strip("_")
        path = Path(f"{ion_slug}_DFT_spatial_Zeff.csv").resolve()
    if path.is_file() and path.stat().st_size > 200:
        return path
    table = np.column_stack(
        (
            radial_grid, density_alpha, density_beta,
            potentials["enclosed_electrons"], potentials["hartree"],
            potentials["vxc_alpha"], potentials["vxc_beta"],
            potentials["vks_alpha"], potentials["vks_beta"],
            potentials["zeff_alpha"], potentials["zeff_beta"],
        )
    )
    header = (
        "r_bohr,rho_alpha,rho_beta,N_enclosed,v_hartree,"
        "v_xc_alpha,v_xc_beta,v_ks_alpha,v_ks_beta,Zeff_alpha,Zeff_beta"
    )
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        np.savetxt(temporary, table, delimiter=",", header=header, comments="")
        temporary.replace(path)
    except OSError as exc:
        print(f"Radial CSV export skipped because it could not be written: {exc}")
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return path


def calculate_current_selection() -> Path:
    """Resolve one request through the dependency graph and return its HTML path.

    The order is intentionally coarse-to-fine: atomic fields -> radial family ->
    selected radial state -> angular state -> Cartesian grid -> surface mesh ->
    initial Plotly figure. Each stage can therefore stop at its own cache hit.
    """
    start = time.perf_counter()
    symbol, atomic_number, ionic_charge, electron_count = validate_settings()
    print(
        f"Ion: {ION}; Z={atomic_number}; charge={ionic_charge:+d}; "
        f"electrons={electron_count}; spin={SPIN}"
    )
    atomic_key = dft_cache_key(symbol, ionic_charge)
    visual_key = result_cache_key(atomic_key)
    html_path = cache_file("rendered-results", visual_key, ".html")
    if html_path is None:
        ion_slug = re.sub(r"[^A-Za-z0-9]+", "_", ION).strip("_")
        html_path = Path(
            f"{ion_slug}_DFT_orbital_explorer_"
            f"n{N_QUANTUM}_l{L_QUANTUM}_m{M_QUANTUM}.html"
        ).resolve()
    elif html_path.is_file():
        try:
            if (
                html_path.stat().st_size >= RESULT_CACHE_MIN_BYTES
                and representation_cache_ready(visual_key)
            ):
                print("Persistent cache hit: complete interactive result.")
                os.utime(html_path, None)
                touch_representation_dependencies(visual_key)
                return html_path
            html_path.unlink(missing_ok=True)
        except OSError:
            pass

    # Layer 1: expensive electronic structure shared by every orbital and m
    # representation having the same atom, charge, spin and quality physics.
    (
        radial_grid,
        density_alpha,
        density_beta,
        potentials,
        dft_energy,
        calculation_info,
        atomic_cache_hit,
    ) = load_or_calculate_atomic_fields(
        symbol,
        atomic_number,
        ionic_charge,
        atomic_key,
    )

    captured_electrons = float(potentials["enclosed_electrons"][-1])
    density_error = abs(captured_electrons - electron_count) / electron_count
    print(f"DFT total energy:          {dft_energy:#.4g} Eh")
    print(f"Radial density integral:   {captured_electrons:#.4g} electrons")
    print(f"Density truncation error:  {100.0 * density_error:#.4g}%")
    if density_error > 0.01:
        print(
            "WARNING: More than 1% of the DFT density is missing. Increase "
            "RADIAL_MAX_BOHR and/or ANGULAR_DIRECTIONS."
        )

    # Layer 2: solve and cache all supported n values for the selected l/spin
    # family, then select the requested state without another eigensolve.
    spin_key = ORBITAL_SPIN.lower()
    selected_zeff = potentials[f"zeff_{spin_key}"]
    (
        orbital_energy,
        orbital_r,
        radial_function,
        orbital_cache_hit,
        radial_family_key_value,
        radial_state_key_value,
    ) = load_or_build_common_orbital_bank(
        atomic_key,
        radial_grid,
        potentials,
        bool(calculation_info["relativistic"]),
    )
    if orbital_energy >= 0.0:
        print(
            "WARNING: The selected box-discretized radial state has non-negative "
            "energy. It may be unbound; increase RADIAL_MAX_BOHR or choose an "
            "occupied/lower state."
        )

    # Layers 3-4: angular data is atom-independent; the spatial key joins that
    # state with the radial dependency. The mesh then depends only on the grid.
    angular_key = angular_state_cache_key()
    spatial_key = spatial_grid_cache_key(radial_state_key_value, angular_key)
    (
        axis,
        wavefunction,
        density,
        surface_half_width,
        spatial_cache_hit,
    ) = load_or_build_spatial_grid(
        spatial_key,
        orbital_r,
        radial_function,
    )
    spacing = float(axis[1] - axis[0])
    mesh_key = isosurface_mesh_cache_key(spatial_key)
    mesh_data, threshold, achieved, mesh_cache_hit = load_or_build_isosurface_mesh(
        mesh_key,
        axis,
        wavefunction,
        density,
    )
    # The compact representation manifest is the hand-off to lazy tab routes.
    # Only the isosurface is built now; other figures remain None until selected.
    panel_keys = representation_panel_keys(
        atomic_key,
        radial_state_key_value,
        angular_key,
        spatial_key,
    )
    radial_csv = save_radial_data(
        radial_grid,
        density_alpha,
        density_beta,
        potentials,
        atomic_key,
    )
    cache_representation_manifest(
        visual_key,
        atomic_key,
        radial_family_key_value,
        radial_state_key_value,
        angular_key,
        spatial_key,
        panel_keys,
        orbital_energy,
        threshold,
        achieved,
        surface_half_width,
    )
    isosurface_label, isosurface_description = VIEW_DEFINITIONS["isosurface"]
    figures = {
        "isosurface": (
            isosurface_label,
            isosurface_description,
            build_figure_safely(
                "90% isosurface",
                lambda: build_isosurface_figure(
                    axis,
                    wavefunction,
                    density,
                    threshold,
                    achieved,
                    orbital_energy,
                    mesh_data,
                ),
            ),
        ),
        "dot-map": (
            *VIEW_DEFINITIONS["dot-map"],
            None,
        ),
        "radial": (
            *VIEW_DEFINITIONS["radial"],
            None,
        ),
        "angular": (
            *VIEW_DEFINITIONS["angular"],
            None,
        ),
        "contours": (
            *VIEW_DEFINITIONS["contours"],
            None,
        ),
    }
    dft_reused = atomic_cache_hit or calculation_info.get("density_cache") == "reused"
    cache_summary = (
        f"DFT {'reused' if dft_reused else 'cached'}; "
        f"radial {'reused' if orbital_cache_hit else 'cached'}; "
        f"grid {'reused' if spatial_cache_hit else 'cached'}; "
        f"mesh {'reused' if mesh_cache_hit else 'cached'}"
    )
    compute_seconds = time.perf_counter() - start
    html_path = write_results_website(
        figures,
        dft_energy,
        orbital_energy,
        achieved,
        calculation_info,
        html_path,
        cache_summary,
        compute_seconds,
        visual_key,
    )

    print(f"Radial KS orbital energy:  {orbital_energy:#.4g} Eh")
    surface_probability = float(density.sum(dtype=np.float64) * spacing**3)
    print(f"Surface-grid probability:  {surface_probability:#.4g}")
    print(f"Enclosed probability:      {achieved:#.4g}")
    print(f"Surface half-width:        {surface_half_width:#.4g} bohr")
    if radial_csv.is_file():
        print(f"Radial Z_eff data:         {radial_csv}")
    else:
        print("Radial Z_eff data:         export skipped")
    print(f"Interactive website:       {html_path}")
    print("Use the website menu to switch between all five visualizations.")
    print(f"Total runtime:             {time.perf_counter() - start:.1f} s")
    return html_path


