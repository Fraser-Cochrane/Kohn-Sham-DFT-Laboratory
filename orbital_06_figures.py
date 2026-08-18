# =============================================================================
# PART 6 OF 9: PLOTLY FIGURE BUILDERS AND LAZY REPRESENTATION DATA
# =============================================================================
#
# Builds the isosurface, density-dot, radial, angular, and contour views.  It
# also manages the cached wavefunction bundle and per-tab lazy Plotly JSON so
# changing representation does not repeat the expensive physical calculation.
#
from __future__ import annotations

def build_figure_safely(title: str, builder: object) -> go.Figure:
    """Keep one optional visualization failure from aborting the whole website."""
    try:
        return builder()
    except Exception as exc:
        print(
            f"{title} rendering was skipped safely "
            f"({type(exc).__name__}: {exc})."
        )
        figure = go.Figure()
        figure.add_annotation(
            text=(
                f"{title} could not be rendered at this resolution.<br>"
                "The other views and cached numerical data remain available."
            ),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="center",
            font={"size": 17, "color": "#5f6f86"},
        )
        figure.update_layout(
            title={"text": title, "x": 0.5},
            template="plotly_white",
            height=650,
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return figure


def build_isosurface_figure(
    axis: np.ndarray,
    wavefunction: np.ndarray,
    density: np.ndarray,
    threshold: float,
    achieved_fraction: float,
    energy: float,
    mesh_data: dict[str, np.ndarray] | None = None,
) -> go.Figure:
    """Build the draggable probability isosurface coloured by orbital phase."""
    spacing = float(axis[1] - axis[0])
    if mesh_data is None:
        mesh_density, _removed_voxels = filter_isosurface_density(
            density,
            threshold,
        )
        vertices, faces, _normals, _values = marching_cubes(
            writable_compiled_array(mesh_density, dtype=np.float32),
            level=threshold,
            spacing=(spacing, spacing, spacing),
            step_size=MARCHING_CUBES_STEP,
            allow_degenerate=False,
        )
        vertices += axis[0]
        surface_wavefunction = interpolate_surface_values(
            axis, wavefunction, vertices
        )
    else:
        vertices = mesh_data["vertices"]
        faces = mesh_data["faces"]
        surface_wavefunction = mesh_data["surface_wavefunction"]

    common_mesh = dict(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=SURFACE_ALPHA,
        flatshading=False,
        lighting={
            "ambient": 0.45, "diffuse": 0.75, "specular": 0.35,
            "roughness": 0.55, "fresnel": 0.15,
        },
        lightposition={"x": 100, "y": 200, "z": 150},
        hovertemplate=(
            "x=%{x:.4g} bohr<br>y=%{y:.4g} bohr<br>"
            "z=%{z:.4g} bohr<extra></extra>"
        ),
    )

    if np.iscomplexobj(wavefunction):
        phase_intensity = np.angle(surface_wavefunction) / np.pi
        phase_scale = "HSV"
        phase_bar = {
            "title": {"text": "phase / pi", "side": "top", "font": {"size": 14}},
            "tickformat": ".1f",
            "tickfont": {"size": 12},
            "x": 0.80,
            "xanchor": "left",
            "y": 0.50,
            "len": 0.72,
            "thickness": 24,
            "xpad": 12,
            "bgcolor": "rgba(255,255,255,0.94)",
            "outlinecolor": "#5f6f86",
            "outlinewidth": 1,
            "bordercolor": "rgba(95,111,134,0.35)",
            "borderwidth": 1,
        }
    else:
        phase_intensity = np.sign(surface_wavefunction.real)
        phase_scale = [
            [0.000, "royalblue"], [0.499, "royalblue"],
            [0.500, "crimson"], [1.000, "crimson"],
        ]
        phase_bar = {
            "title": {"text": "sign(ψ)", "side": "top", "font": {"size": 14}},
            "tickvals": [-1, 1],
            "ticktext": ["negative", "positive"],
            "tickfont": {"size": 12},
            "x": 0.80,
            "xanchor": "left",
            "y": 0.50,
            "len": 0.72,
            "thickness": 24,
            "xpad": 12,
            "bgcolor": "rgba(255,255,255,0.94)",
            "outlinecolor": "#5f6f86",
            "outlinewidth": 1,
            "bordercolor": "rgba(95,111,134,0.35)",
            "borderwidth": 1,
        }

    phase_mesh = go.Mesh3d(
        **common_mesh, name="Orbital phase", intensity=phase_intensity,
        intensitymode="vertex", colorscale=phase_scale, cmin=-1.0, cmax=1.0,
        colorbar=phase_bar, showscale=True, visible=True,
    )
    nucleus = go.Scatter3d(
        x=[0.0], y=[0.0], z=[0.0], mode="markers", name="Nucleus",
        marker={"size": 5, "color": "black"},
        hovertemplate="Nucleus<extra></extra>",
    )

    extent = max(float(np.max(np.abs(vertices))) * 1.08, spacing)
    axis_style = {
        "range": [-extent, extent], "showbackground": True,
        "backgroundcolor": "rgb(245,247,250)", "gridcolor": "white",
        "zerolinecolor": "rgb(170,170,170)", "tickformat": ".1f",
    }
    figure = go.Figure(data=[phase_mesh, nucleus])
    figure.update_layout(
        title={
            "text": (
                f"{ION}: {100.0 * achieved_fraction:#.4g}% probability surface"
                "<br>"
                rf"$(n,\ell,m)=({N_QUANTUM},{L_QUANTUM},{M_QUANTUM}),\quad "
                rf"E_{{n\ell}}={energy:#.4g}\,E_\mathrm{{h}}$"
            ),
            "x": 0.5,
        },
        scene={
            "domain": {"x": [0.0, 0.72], "y": [0.0, 1.0]},
            "xaxis": {
                **axis_style,
                "title": {"text": "<i>x</i>/<i>a</i><sub>0</sub>"},
            },
            "yaxis": {
                **axis_style,
                "title": {"text": "<i>y</i>/<i>a</i><sub>0</sub>"},
            },
            "zaxis": {
                **axis_style,
                "title": {"text": "<i>z</i>/<i>a</i><sub>0</sub>"},
            },
            "aspectmode": "cube", "dragmode": "orbit",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.15}},
        },
        legend={"x": 0.01, "y": 0.99},
        margin={"l": 10, "r": 20, "b": 10, "t": 110},
        autosize=True, height=840, template="plotly_white",
    )

    return figure


def sample_density_dot_data(
    axis: np.ndarray,
    density: np.ndarray,
) -> dict[str, np.ndarray]:
    """Sample deterministic probability-weighted coordinates from |psi|^2."""
    spacing = float(axis[1] - axis[0])
    probabilities = density.ravel().astype(float)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(DOT_MAP_SEED)
    chosen = rng.choice(
        probabilities.size,
        size=DOT_MAP_POINTS,
        replace=True,
        p=probabilities,
    )
    x_index, y_index, z_index = np.unravel_index(chosen, density.shape)
    jitter = rng.uniform(-0.5 * spacing, 0.5 * spacing, size=(DOT_MAP_POINTS, 3))
    x_coord = axis[x_index] + jitter[:, 0]
    y_coord = axis[y_index] + jitter[:, 1]
    z_coord = axis[z_index] + jitter[:, 2]
    sampled_log_density = np.log10(
        np.maximum(density[x_index, y_index, z_index], 1.0e-30)
    )
    return {
        "x": np.asarray(x_coord, dtype=np.float32),
        "y": np.asarray(y_coord, dtype=np.float32),
        "z": np.asarray(z_coord, dtype=np.float32),
        "log_density": np.asarray(sampled_log_density, dtype=np.float32),
    }


def load_or_build_density_dot_data(
    dot_key: str,
    axis: np.ndarray,
    density: np.ndarray,
) -> tuple[dict[str, np.ndarray], bool]:
    """Reuse deterministic dot samples without reopening the full density grid."""
    path = cache_bundle("density-dot-data", dot_key)
    required = ("x", "y", "z", "log_density")
    loaded = load_array_bundle(path, required)
    if loaded is not None:
        arrays, _metadata = loaded
        if all(arrays[name].shape == (DOT_MAP_POINTS,) for name in required):
            print("Persistent cache hit: mmap density-dot samples.")
            return arrays, True
    samples = sample_density_dot_data(axis, density)
    if atomic_save_array_bundle(path, samples):
        remapped = load_array_bundle(path, required)
        if remapped is not None:
            return remapped[0], False
    return samples, False


def build_density_dot_map(
    axis: np.ndarray,
    density: np.ndarray,
    sampled_data: dict[str, np.ndarray] | None = None,
) -> go.Figure:
    """Draw probability-weighted dots whose spatial frequency follows |psi|^2."""
    samples = (
        sample_density_dot_data(axis, density)
        if sampled_data is None
        else sampled_data
    )
    x_coord = samples["x"]
    y_coord = samples["y"]
    z_coord = samples["z"]
    sampled_log_density = samples["log_density"]

    figure = go.Figure(
        go.Scatter3d(
            x=x_coord,
            y=y_coord,
            z=z_coord,
            mode="markers",
            name="Probability samples",
            marker={
                "size": 2.0,
                "opacity": 0.38,
                "color": sampled_log_density,
                "colorscale": "Turbo",
                "showscale": True,
                "colorbar": {
                    "title": {
                        "text": "log₁₀ |ψ|²",
                        "side": "top",
                        "font": {"size": 14},
                    },
                    "tickformat": ".1f",
                    "tickfont": {"size": 12},
                    "x": 0.80,
                    "xanchor": "left",
                    "y": 0.50,
                    "len": 0.72,
                    "thickness": 24,
                    "xpad": 12,
                    "bgcolor": "rgba(255,255,255,0.94)",
                    "outlinecolor": "#5f6f86",
                    "outlinewidth": 1,
                    "bordercolor": "rgba(95,111,134,0.35)",
                    "borderwidth": 1,
                },
            },
            hovertemplate=(
                "x=%{x:.4g} bohr<br>y=%{y:.4g} bohr<br>"
                "z=%{z:.4g} bohr<extra></extra>"
            ),
        )
    )
    extent = max(abs(axis[0]), abs(axis[-1]))
    scene_axis = {
        "range": [-extent, extent],
        "showbackground": True,
        "backgroundcolor": "rgb(246,248,252)",
        "gridcolor": "white",
        "tickformat": ".1f",
    }
    figure.update_layout(
        title={
            "text": (
                f"Electron-density dot map: {ION}<br>"
                rf"$(n,\ell,m)=({N_QUANTUM},{L_QUANTUM},{M_QUANTUM})$"
            ),
            "x": 0.5,
        },
        scene={
            "domain": {"x": [0.0, 0.72], "y": [0.0, 1.0]},
            "xaxis": {
                **scene_axis,
                "title": {"text": "<i>x</i>/<i>a</i><sub>0</sub>"},
            },
            "yaxis": {
                **scene_axis,
                "title": {"text": "<i>y</i>/<i>a</i><sub>0</sub>"},
            },
            "zaxis": {
                **scene_axis,
                "title": {"text": "<i>z</i>/<i>a</i><sub>0</sub>"},
            },
            "aspectmode": "cube",
            "dragmode": "orbit",
        },
        margin={"l": 10, "r": 20, "b": 10, "t": 70},
        autosize=True,
        height=790,
        template="plotly_white",
    )
    return figure


def build_radial_figure(
    orbital_r: np.ndarray,
    radial_function: np.ndarray,
) -> go.Figure:
    """Plot R_nl(r) and radial probability r^2 |R_nl(r)|^2."""
    radial_probability = orbital_r**2 * np.abs(radial_function) ** 2
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=orbital_r,
            y=radial_function,
            mode="lines",
            name=r"$R_{n\ell}(r)$",
            line={"color": "royalblue", "width": 3},
            hovertemplate="r=%{x:.4g}<br>R=%{y:.4g}<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=orbital_r,
            y=radial_probability,
            mode="lines",
            name=r"$r^2|R_{n\ell}(r)|^2$",
            line={"color": "darkorange", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(255,140,0,0.15)",
            hovertemplate="r=%{x:.4g}<br>P(r)=%{y:.4g}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_xaxes(
        title_text="<i>r</i>/<i>a</i><sub>0</sub>",
        tickformat=".1f",
    )
    figure.update_yaxes(
        title_text=(
            "<i>R</i><sub>nℓ</sub>(<i>r</i>) / "
            "<i>a</i><sub>0</sub><sup>−3/2</sup>"
        ),
        tickformat=".1f",
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text=(
            "<i>r</i><sup>2</sup>|<i>R</i><sub>nℓ</sub>(<i>r</i>)|"
            "<sup>2</sup> / <i>a</i><sub>0</sub><sup>−1</sup>"
        ),
        tickformat=".1f",
        secondary_y=True,
    )
    figure.update_layout(
        title={
            "text": (
                "Numerical radial Kohn–Sham wavefunction<br>"
                rf"$n={N_QUANTUM},\quad \ell={L_QUANTUM}$"
            ),
            "x": 0.5,
        },
        hovermode="x unified",
        height=720,
        template="plotly_white",
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.04},
        margin={"l": 80, "r": 80, "b": 70, "t": 90},
    )
    return figure


def calculate_angular_plot_data() -> dict[str, np.ndarray]:
    """Calculate only the arrays required by the angular heatmaps."""
    theta = np.linspace(0.0, np.pi, ANGULAR_PLOT_THETA_POINTS)
    phi = np.linspace(-np.pi, np.pi, ANGULAR_PLOT_PHI_POINTS)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    angular = angular_wavefunction(theta_grid, np.mod(phi_grid, 2.0 * np.pi))
    left_values = (
        np.angle(angular) / np.pi if np.iscomplexobj(angular) else angular
    )
    return {
        "theta": np.asarray(theta, dtype=np.float32),
        "phi": np.asarray(phi, dtype=np.float32),
        "left_values": np.asarray(left_values, dtype=np.float32),
        "probability": np.asarray(np.abs(angular) ** 2, dtype=np.float32),
    }


def load_or_build_angular_plot_data(
    angular_plot_key_value: str,
) -> tuple[dict[str, np.ndarray], bool]:
    """Reuse angular grids across atoms and radial states with the same l/m/form."""
    path = cache_bundle("angular-plot-data", angular_plot_key_value)
    required = ("theta", "phi", "left_values", "probability")
    loaded = load_array_bundle(path, required)
    expected_shape = (ANGULAR_PLOT_THETA_POINTS, ANGULAR_PLOT_PHI_POINTS)
    if loaded is not None:
        arrays, _metadata = loaded
        if (
            arrays["theta"].shape == (ANGULAR_PLOT_THETA_POINTS,)
            and arrays["phi"].shape == (ANGULAR_PLOT_PHI_POINTS,)
            and arrays["left_values"].shape == expected_shape
            and arrays["probability"].shape == expected_shape
        ):
            print("Persistent cache hit: mmap angular plot data.")
            return arrays, True
    arrays = calculate_angular_plot_data()
    if atomic_save_array_bundle(path, arrays):
        remapped = load_array_bundle(path, required)
        if remapped is not None:
            return remapped[0], False
    return arrays, False


def build_angular_figure(
    angular_data: dict[str, np.ndarray] | None = None,
) -> go.Figure:
    """Plot angular amplitude/phase and angular probability over the sphere."""
    data = calculate_angular_plot_data() if angular_data is None else angular_data
    theta = data["theta"]
    phi = data["phi"]
    left_values = data["left_values"]
    probability = data["probability"]

    if ORBITAL_FORM.upper() == "COMPLEX":
        left_title = r"$\arg(Y_{\ell}^{m})/\pi$"
        left_bar_title = "arg(<i>Y</i><sub>ℓ</sub><sup>m</sup>)/π"
        left_scale = "HSV"
        left_min, left_max = -1.0, 1.0
    else:
        left_title = r"$\operatorname{Re}Y_{\ell}^{m}$"
        left_bar_title = "<i>Y</i><sub>ℓ</sub><sup>m</sup>"
        limit = float(np.max(np.abs(left_values)))
        left_scale = "RdBu_r"
        left_min, left_max = -limit, limit

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(left_title, r"$|Y_{\ell}^{m}|^2$"),
        horizontal_spacing=0.13,
    )
    figure.add_trace(
        go.Heatmap(
            x=np.degrees(phi),
            y=np.degrees(theta),
            z=left_values,
            colorscale=left_scale,
            zmin=left_min,
            zmax=left_max,
            colorbar={
                "title": {"text": left_bar_title, "side": "top"},
                "x": 0.44,
                "y": 0.50,
                "yanchor": "middle",
                "len": 0.82,
                "lenmode": "fraction",
                "ypad": 0,
                "tickformat": ".1f",
            },
            hovertemplate="phi=%{x:.4g} deg<br>theta=%{y:.4g} deg<br>value=%{z:.4g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            x=np.degrees(phi),
            y=np.degrees(theta),
            z=probability,
            colorscale="Viridis",
            colorbar={
                "title": {
                    "text": "|<i>Y</i><sub>ℓ</sub><sup>m</sup>|<sup>2</sup>",
                    "side": "top",
                },
                "x": 1.02,
                "y": 0.50,
                "yanchor": "middle",
                "len": 0.82,
                "lenmode": "fraction",
                "ypad": 0,
                "tickformat": ".1f",
            },
            hovertemplate="phi=%{x:.4g} deg<br>theta=%{y:.4g} deg<br>|Y|^2=%{z:.4g}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(
        title_text="<i>φ</i> / degrees",
        tickformat=".1f",
    )
    figure.update_yaxes(
        title_text="<i>θ</i> / degrees",
        tickformat=".1f",
        autorange="reversed",
    )
    figure.update_layout(
        title={
            "text": (
                "Angular wavefunction<br>"
                rf"$\ell={L_QUANTUM},\quad m={M_QUANTUM}$ "
                f"({ORBITAL_FORM.lower()} form)"
            ),
            "x": 0.5,
        },
        height=690,
        template="plotly_white",
        margin={"l": 70, "r": 90, "b": 70, "t": 95},
    )
    return figure


def calculate_contour_data(
    axis: np.ndarray,
    density: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate the three central log-density planes once."""
    centre = int(np.argmin(np.abs(axis)))
    positive = density[density > 0.0]
    floor = max(float(np.max(density)) * 1.0e-10, float(np.min(positive)))
    log_density = np.log10(np.maximum(density, floor))
    return {
        "axis": np.asarray(axis, dtype=np.float32),
        "log_planes": np.asarray(
            (
                log_density[:, :, centre].T,
                log_density[:, centre, :].T,
                log_density[centre, :, :].T,
            ),
            dtype=np.float32,
        ),
    }


def load_or_build_contour_data(
    contour_key: str,
    axis: np.ndarray,
    density: np.ndarray,
) -> tuple[dict[str, np.ndarray], bool]:
    """Reuse central contour planes without remapping the full 3D density later."""
    path = cache_bundle("contour-data", contour_key)
    required = ("axis", "log_planes")
    loaded = load_array_bundle(path, required)
    if loaded is not None:
        arrays, _metadata = loaded
        if (
            arrays["axis"].ndim == 1
            and arrays["log_planes"].shape
            == (3, arrays["axis"].size, arrays["axis"].size)
        ):
            print("Persistent cache hit: mmap contour planes.")
            return arrays, True
    arrays = calculate_contour_data(axis, density)
    if atomic_save_array_bundle(path, arrays):
        remapped = load_array_bundle(path, required)
        if remapped is not None:
            return remapped[0], False
    return arrays, False


def build_contour_figure(
    axis: np.ndarray,
    density: np.ndarray | None = None,
    contour_data: dict[str, np.ndarray] | None = None,
) -> go.Figure:
    """Plot logarithmic probability contours through x-y, x-z, and y-z planes."""
    data = (
        calculate_contour_data(axis, density)
        if contour_data is None and density is not None
        else contour_data
    )
    if data is None:
        raise ValueError("Contour data or a density grid is required.")
    axis = data["axis"]
    log_planes = data["log_planes"]
    planes = (
        (log_planes[0], "x", "y", r"$x\text{–}y\ \mathrm{plane}\ (z=0)$"),
        (log_planes[1], "x", "z", r"$x\text{–}z\ \mathrm{plane}\ (y=0)$"),
        (log_planes[2], "y", "z", r"$y\text{–}z\ \mathrm{plane}\ (x=0)$"),
    )
    figure = make_subplots(rows=1, cols=3, subplot_titles=[item[3] for item in planes])
    z_min = float(np.min(log_planes))
    z_max = float(np.max(log_planes))
    for column, (plane, x_title, y_title, _title) in enumerate(planes, 1):
        figure.add_trace(
            go.Contour(
                x=axis,
                y=axis,
                z=plane,
                colorscale="Magma",
                zmin=z_min,
                zmax=z_max,
                contours={"coloring": "heatmap", "showlines": True},
                line={"width": 0.6, "color": "rgba(255,255,255,0.42)"},
                showscale=column == 3,
                colorbar={
                    "title": {"text": "log₁₀ |ψ|²"},
                    "tickformat": ".1f",
                },
                hovertemplate=(
                    f"{x_title}=%{{x:.4g}} bohr<br>"
                    f"{y_title}=%{{y:.4g}} bohr<br>"
                    "log density=%{z:.4g}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(
            title_text=f"<i>{x_title}</i>/<i>a</i><sub>0</sub>",
            tickformat=".1f",
            scaleanchor=f"y{column if column > 1 else ''}",
            row=1,
            col=column,
        )
        figure.update_yaxes(
            title_text=f"<i>{y_title}</i>/<i>a</i><sub>0</sub>",
            tickformat=".1f",
            row=1,
            col=column,
        )
    figure.update_layout(
        title={
            "text": "Orbital probability-density contours: log₁₀ |ψ|²",
            "x": 0.5,
        },
        height=650,
        template="plotly_white",
        margin={"l": 60, "r": 90, "b": 70, "t": 95},
    )
    return figure


VIEW_DEFINITIONS: dict[str, tuple[str, str]] = {
    "isosurface": (
        "90% isosurface",
        r"Rotate the probability isosurface coloured by orbital phase.",
    ),
    "dot-map": (
        "Electron-density dot map",
        r"Each dot is sampled from \(|ψ|^2\), so denser clouds indicate greater orbital probability.",
    ),
    "radial": (
        "Radial wavefunction",
        r"Compare \(R_{n\ell}(r)\) with the normalized radial probability \(r^2|R_{n\ell}(r)|^2\).",
    ),
    "angular": (
        "Angular wavefunction",
        r"Inspect the amplitude or phase of \(Y_{\ell}^{m}\) alongside \(|Y_{\ell}^{m}|^2\).",
    ),
    "contours": (
        "Orbital contours",
        r"View \(\log_{10}|ψ|^2\) contours through the three central coordinate planes.",
    ),
}

def wavefunction_cache_path(visual_key: str) -> Path:
    """Return the small manifest linking one page to physical cache layers.

    The manifest stores keys, not duplicate numerical arrays. It lets the result
    page locate independently cached density, radial, spatial and mesh data.
    """
    path = cache_file("representation-manifests", visual_key, ".json")
    if path is not None:
        return path
    return Path.cwd() / f"orbital-representation-{visual_key}.json"


def read_representation_manifest(visual_key: str) -> dict[str, object] | None:
    path = wavefunction_cache_path(visual_key)
    if not path.is_file():
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(metadata, dict)
            or metadata.get("format_version") != CACHE_FORMAT_VERSION
            or not isinstance(metadata.get("panel_keys"), dict)
        ):
            raise ValueError("representation manifest is stale or incomplete")
        os.utime(path, None)
        return metadata
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Ignoring invalid representation manifest {path.name}: {exc}")
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def representation_cache_ready(visual_key: str) -> bool:
    """Require every non-lazy physical parent needed by the result page.

    Cached HTML alone is insufficient: eviction may have removed a numerical
    parent needed by a later lazy tab, so all mandatory dependencies are checked.
    """
    metadata = read_representation_manifest(visual_key)
    if metadata is None:
        return False
    try:
        dependencies = (
            cache_bundle("dft-fields", str(metadata["atomic_key"])),
            cache_bundle("radial-families", str(metadata["radial_family_key"])),
            cache_bundle("spatial-grids", str(metadata["spatial_grid_key"])),
            cache_bundle("isosurface-meshes", str(metadata["mesh_key"])),
        )
        return all(load_array_bundle(path, ()) is not None for path in dependencies)
    except (KeyError, TypeError, ValueError):
        return False


def touch_representation_dependencies(visual_key: str) -> None:
    """Keep the manifest and its physical parents together in the LRU cache."""
    metadata = read_representation_manifest(visual_key)
    if metadata is None:
        return
    mappings = (
        ("dft-fields", "atomic_key"),
        ("radial-families", "radial_family_key"),
        ("spatial-grids", "spatial_grid_key"),
        ("isosurface-meshes", "mesh_key"),
    )
    now = time.time()
    for category, field in mappings:
        try:
            path = cache_bundle(category, str(metadata[field]))
            if path is not None and path.is_dir():
                os.utime(path, (now, now))
                manifest = path / "manifest.json"
                if manifest.is_file():
                    os.utime(manifest, (now, now))
        except (KeyError, OSError, TypeError):
            continue


def lazy_figure_cache_path(visual_key: str, panel_id: str) -> Path:
    """Return the cached Plotly JSON path for one lazily generated tab."""
    manifest = read_representation_manifest(visual_key)
    panel_dependency = visual_key
    if manifest is not None:
        panel_keys = manifest.get("panel_keys", {})
        if isinstance(panel_keys, dict):
            panel_dependency = str(panel_keys.get(panel_id, visual_key))
    key = stable_cache_key(
        "lazy-tab-figure",
        {
            "panel_dependency": panel_dependency,
            "panel_id": panel_id,
            "render_version": RESULT_RENDER_VERSION,
        },
    )
    path = cache_file("lazy-tab-figures", key, ".json")
    if path is not None:
        return path
    return Path.cwd() / f"orbital-tab-{key}.json"


def cache_representation_manifest(
    visual_key: str,
    atomic_key: str,
    radial_family_key_value: str,
    radial_state_key_value: str,
    angular_key: str,
    spatial_key: str,
    panel_keys: dict[str, str],
    orbital_energy: float,
    threshold: float,
    achieved: float,
    surface_half_width: float,
) -> Path:
    """Persist lightweight links instead of duplicating numerical arrays."""
    path = wavefunction_cache_path(visual_key)
    if read_representation_manifest(visual_key) is not None:
        return path
    metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "visual_key": visual_key,
        "atomic_key": atomic_key,
        "radial_family_key": radial_family_key_value,
        "radial_state_key": radial_state_key_value,
        "radial_state_index": N_QUANTUM - L_QUANTUM - 1,
        "angular_key": angular_key,
        "angular_plot_key": angular_plot_cache_key(angular_key),
        "spatial_grid_key": spatial_key,
        "mesh_key": isosurface_mesh_cache_key(spatial_key),
        "dot_data_key": dot_data_cache_key(spatial_key),
        "contour_data_key": contour_data_cache_key(spatial_key),
        "panel_keys": panel_keys,
        "ion": ION,
        "spin_2s": SPIN,
        "orbital_spin": ORBITAL_SPIN,
        "n": N_QUANTUM,
        "l": L_QUANTUM,
        "m": M_QUANTUM,
        "form": ORBITAL_FORM,
        "quality_level": QUALITY_LEVEL,
        "isotope": ISOTOPE_MASS_NUMBER,
        "orbital_energy": float(orbital_energy),
        "threshold": float(threshold),
        "achieved": float(achieved),
        "surface_half_width": float(surface_half_width),
    }
    atomic_write_text(
        path,
        json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    )
    prune_cache(exclude=path)
    return path


def load_wavefunction_data(
    visual_key: str,
    panel_id: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Map only the physical cache branches needed by one lazy panel."""
    if panel_id not in VIEW_DEFINITIONS:
        raise ValueError(f"Unknown result panel {panel_id!r}.")
    metadata = read_representation_manifest(visual_key)
    if metadata is None:
        raise FileNotFoundError("Cached representation dependencies are unavailable.")
    data: dict[str, np.ndarray] = {}

    if panel_id == "radial":
        loaded = load_array_bundle(
            cache_bundle("radial-families", str(metadata["radial_family_key"])),
            ("interior_r", "radial_functions"),
        )
        if loaded is None:
            raise FileNotFoundError("Cached radial family is unavailable.")
        arrays, _bundle_metadata = loaded
        index = int(metadata["radial_state_index"])
        data = {
            "orbital_r": arrays["interior_r"],
            "radial_function": arrays["radial_functions"][index],
        }
    elif panel_id == "angular":
        loaded = load_array_bundle(
            cache_bundle("angular-plot-data", str(metadata["angular_plot_key"])),
            ("theta", "phi", "left_values", "probability"),
        )
        if loaded is not None:
            data = loaded[0]
    elif panel_id == "dot-map":
        dots = load_array_bundle(
            cache_bundle("density-dot-data", str(metadata["dot_data_key"])),
            ("x", "y", "z", "log_density"),
        )
        spatial = load_array_bundle(
            cache_bundle("spatial-grids", str(metadata["spatial_grid_key"])),
            ("axis",) if dots is not None else ("axis", "density"),
        )
        if spatial is None:
            raise FileNotFoundError("Cached Cartesian density is unavailable.")
        data = spatial[0]
        if dots is not None:
            data.update(dots[0])
    elif panel_id == "contours":
        contours = load_array_bundle(
            cache_bundle("contour-data", str(metadata["contour_data_key"])),
            ("axis", "log_planes"),
        )
        if contours is not None:
            data = contours[0]
        else:
            spatial = load_array_bundle(
                cache_bundle("spatial-grids", str(metadata["spatial_grid_key"])),
                ("axis", "density"),
            )
            if spatial is None:
                raise FileNotFoundError("Cached Cartesian density is unavailable.")
            data = spatial[0]
    else:
        spatial = load_array_bundle(
            cache_bundle("spatial-grids", str(metadata["spatial_grid_key"])),
            ("axis", "wavefunction", "density"),
        )
        mesh = load_array_bundle(
            cache_bundle("isosurface-meshes", str(metadata["mesh_key"])),
            ("vertices", "faces", "surface_wavefunction"),
        )
        if spatial is None:
            raise FileNotFoundError("Cached isosurface dependencies are unavailable.")
        data = spatial[0]
        if mesh is not None:
            data.update(mesh[0])
    return data, metadata


def build_lazy_figure_from_data(
    panel_id: str,
    data: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> go.Figure:
    """Build one requested visualization from cached numerical arrays."""
    if panel_id == "dot-map":
        if all(name in data for name in ("x", "y", "z", "log_density")):
            samples = {name: data[name] for name in ("x", "y", "z", "log_density")}
        else:
            samples, _hit = load_or_build_density_dot_data(
                str(metadata["dot_data_key"]), data["axis"], data["density"]
            )
        density = data.get("density", np.empty(0, dtype=np.float32))
        return build_density_dot_map(data["axis"], density, samples)
    if panel_id == "radial":
        return build_radial_figure(data["orbital_r"], data["radial_function"])
    if panel_id == "angular":
        angular_data = data
        if not all(
            name in angular_data
            for name in ("theta", "phi", "left_values", "probability")
        ):
            angular_data, _hit = load_or_build_angular_plot_data(
                str(metadata["angular_plot_key"])
            )
        return build_angular_figure(angular_data)
    if panel_id == "contours":
        if "log_planes" in data:
            contour_data = {"axis": data["axis"], "log_planes": data["log_planes"]}
        else:
            contour_data, _hit = load_or_build_contour_data(
                str(metadata["contour_data_key"]), data["axis"], data["density"]
            )
        return build_contour_figure(data["axis"], contour_data=contour_data)
    if panel_id == "isosurface":
        mesh_data = None
        if all(
            name in data
            for name in ("vertices", "faces", "surface_wavefunction")
        ):
            mesh_data = {
                name: data[name]
                for name in ("vertices", "faces", "surface_wavefunction")
            }
        return build_isosurface_figure(
            data["axis"],
            data["wavefunction"],
            data["density"],
            float(metadata["threshold"]),
            float(metadata["achieved"]),
            float(metadata["orbital_energy"]),
            mesh_data,
        )
    raise ValueError(f"Unknown result panel {panel_id!r}.")


def read_cached_figure_json(path: Path) -> str | None:
    """Return validated Plotly JSON, deleting a corrupt lazy cache entry."""
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        document = json.loads(text)
        if not isinstance(document, dict) or not isinstance(document.get("data"), list):
            raise ValueError("figure JSON has no data array")
        if not isinstance(document.get("layout", {}), dict):
            raise ValueError("figure JSON has no layout object")
        os.utime(path, None)
        return text
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Ignoring invalid lazy figure cache {path.name}: {exc}")
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


