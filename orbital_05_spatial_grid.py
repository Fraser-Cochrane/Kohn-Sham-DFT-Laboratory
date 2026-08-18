# =============================================================================
# PART 5 OF 9: ANGULAR WAVEFUNCTION, CARTESIAN GRID, AND SURFACE EXTRACTION
# =============================================================================
#
# Combines the selected numerical radial state with its spherical harmonic,
# samples the orbital in three dimensions, finds the requested probability
# threshold, filters small components, and caches the marching-cubes mesh.
#
from __future__ import annotations

def angular_wavefunction(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Return complex or conventional real/tesseral Y_l^m."""
    def harmonic(m_value: int) -> np.ndarray:
        if sph_harm_y is not None:
            return sph_harm_y(L_QUANTUM, m_value, theta, phi)
        if sph_harm is not None:
            return sph_harm(m_value, L_QUANTUM, phi, theta)
        raise ImportError("SciPy spherical harmonics are unavailable.")

    if ORBITAL_FORM.upper() == "COMPLEX":
        return harmonic(M_QUANTUM)
    if M_QUANTUM == 0:
        return harmonic(0).real
    value = harmonic(abs(M_QUANTUM))
    if M_QUANTUM > 0:
        return np.sqrt(2.0) * (-1.0) ** M_QUANTUM * value.real
    return np.sqrt(2.0) * (-1.0) ** M_QUANTUM * value.imag


def make_orbital_grid(
    orbital_r: np.ndarray,
    radial_function: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Combine numerical R_nl(r) with Y_l^m on a Cartesian grid."""
    orbital_r = np.asarray(orbital_r, dtype=float)
    radial_function = np.asarray(radial_function)
    if (
        orbital_r.ndim != 1
        or radial_function.ndim != 1
        or orbital_r.size != radial_function.size
        or orbital_r.size < 4
    ):
        raise ValueError(
            "The radial solver returned incompatible arrays; clear the cache and "
            "retry the calculation."
        )
    if not np.all(np.isfinite(orbital_r)) or not np.all(np.isfinite(radial_function)):
        raise ValueError("The radial orbital contains non-finite values.")

    radial_probability = np.abs(radial_function * orbital_r) ** 2
    cumulative = cumulative_trapezoid(
        radial_probability, orbital_r, initial=0.0
    )
    radial_norm = float(cumulative[-1])
    if not np.isfinite(radial_norm) or radial_norm <= 0.0:
        raise ValueError("The radial orbital has zero or invalid probability.")
    cumulative /= radial_norm
    radius_999 = orbital_r[
        min(np.searchsorted(cumulative, 0.999), orbital_r.size - 1)
    ]
    half_width = (
        float(SURFACE_BOX_HALF_WIDTH_BOHR)
        if SURFACE_BOX_HALF_WIDTH_BOHR is not None
        else min(RADIAL_MAX_BOHR, max(3.0, 1.10 * radius_999))
    )

    axis = np.linspace(-half_width, half_width, SURFACE_GRID_POINTS)
    wavefunction_dtype = (
        np.complex64 if ORBITAL_FORM.upper() == "COMPLEX" else np.float32
    )
    wavefunction = np.empty(
        (SURFACE_GRID_POINTS,) * 3,
        dtype=wavefunction_dtype,
    )
    y_grid = axis[None, :, None]
    z_grid = axis[None, None, :]
    origin_value = radial_function[0] if L_QUANTUM == 0 else 0.0
    for start in range(0, SURFACE_GRID_POINTS, SURFACE_GRID_BLOCK_SIZE):
        stop = min(start + SURFACE_GRID_BLOCK_SIZE, SURFACE_GRID_POINTS)
        x_grid = axis[start:stop, None, None]
        radius = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)
        cosine_theta = np.divide(
            z_grid,
            radius,
            out=np.zeros_like(radius),
            where=radius > 0.0,
        )
        theta = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
        phi = np.mod(np.arctan2(y_grid, x_grid), 2.0 * np.pi)
        radial_values = np.interp(
            radius,
            orbital_r,
            radial_function,
            left=origin_value,
            right=0.0,
        )
        block = radial_values * angular_wavefunction(theta, phi)
        wavefunction[start:stop] = np.asarray(block, dtype=wavefunction_dtype)

    density = np.asarray(np.abs(wavefunction) ** 2, dtype=np.float32)
    spacing = float(axis[1] - axis[0])
    integral = float(density.sum(dtype=np.float64) * spacing**3)
    if integral <= 0.0:
        raise RuntimeError("The Cartesian orbital grid has zero probability.")
    wavefunction /= np.sqrt(integral)
    density /= integral
    return axis, wavefunction, density, half_width


def load_or_build_spatial_grid(
    spatial_key: str,
    orbital_r: np.ndarray,
    radial_function: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Reuse the mmap Cartesian grid for a fixed radial and angular state."""
    path = cache_bundle("spatial-grids", spatial_key)
    loaded = load_array_bundle(path, ("axis", "wavefunction", "density"))
    if loaded is not None:
        arrays, metadata = loaded
        axis = arrays["axis"]
        wavefunction = arrays["wavefunction"]
        density = arrays["density"]
        expected = (axis.size, axis.size, axis.size)
        try:
            half_width = float(metadata["half_width"])
            if (
                axis.ndim == 1
                and axis.size >= 3
                and wavefunction.shape == expected
                and density.shape == expected
                and np.isfinite(half_width)
            ):
                print("Persistent cache hit: mmap Cartesian orbital grid.")
                return axis, wavefunction, density, half_width, True
        except (KeyError, TypeError, ValueError):
            pass
        print("Cartesian-grid cache dimensions were invalid; rebuilding it.")

    axis, wavefunction, density, half_width = make_orbital_grid(
        orbital_r, radial_function
    )
    saved = atomic_save_array_bundle(
        path,
        {
            "axis": np.asarray(axis, dtype=np.float32),
            "wavefunction": np.asarray(
                wavefunction,
                dtype=np.complex64 if np.iscomplexobj(wavefunction) else np.float32,
            ),
            "density": np.asarray(density, dtype=np.float32),
        },
        {"half_width": float(half_width)},
    )
    if saved:
        remapped = load_array_bundle(path, ("axis", "wavefunction", "density"))
        if remapped is not None:
            arrays, metadata = remapped
            return (
                arrays["axis"],
                arrays["wavefunction"],
                arrays["density"],
                float(metadata["half_width"]),
                False,
            )
    return axis, wavefunction, density, half_width, False


def find_probability_threshold(
    density: np.ndarray,
    spacing: float,
) -> tuple[float, float]:
    """Find the isodensity threshold enclosing ENCLOSED_FRACTION."""
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered, dtype=np.float64) * spacing**3
    target = ENCLOSED_FRACTION * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, target)), ordered.size - 1)
    threshold = float(ordered[index])
    enclosed = float(density[density >= threshold].sum() * spacing**3)
    return threshold, enclosed / float(cumulative[-1])


def filter_isosurface_density(
    density: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, int]:
    """Remove only sub-resolution disconnected components above the isovalue."""
    density_array = np.asarray(density, dtype=np.float32)
    occupied = density_array >= threshold
    occupied_count = int(np.count_nonzero(occupied))
    if occupied_count == 0:
        return density_array, 0
    labels, component_count = connected_components(
        occupied,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    if component_count <= 1:
        return density_array, 0
    sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
    minimum_size = max(
        SURFACE_COMPONENT_MIN_VOXELS,
        int(np.ceil(SURFACE_COMPONENT_MIN_FRACTION * occupied_count)),
    )
    keep = sizes >= minimum_size
    keep[0] = False
    if not np.any(keep[1:]):
        keep[1 + int(np.argmax(sizes[1:]))] = True
    rejected = occupied & ~keep[labels]
    removed = int(np.count_nonzero(rejected))
    if removed == 0:
        return density_array, 0
    filtered = density_array.copy()
    filtered[rejected] = 0.0
    return filtered, removed


def interpolate_surface_values(
    axis: np.ndarray,
    field: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    axis_buffer = writable_compiled_array(axis, dtype=np.float64)
    field_buffer = writable_compiled_array(field)
    vertex_buffer = writable_compiled_array(vertices, dtype=np.float64)
    interpolator = RegularGridInterpolator(
        (axis_buffer, axis_buffer, axis_buffer),
        field_buffer,
        bounds_error=False,
        fill_value=0.0,
    )
    return interpolator(vertex_buffer)


def load_or_build_isosurface_mesh(
    mesh_key: str,
    axis: np.ndarray,
    wavefunction: np.ndarray,
    density: np.ndarray,
) -> tuple[dict[str, np.ndarray], float, float, bool]:
    """Reuse marching-cubes geometry and phase values."""
    path = cache_bundle("isosurface-meshes", mesh_key)
    required = ("vertices", "faces", "surface_wavefunction")
    loaded = load_array_bundle(path, required)
    if loaded is not None:
        arrays, metadata = loaded
        try:
            threshold = float(metadata["threshold"])
            achieved = float(metadata["achieved"])
            if (
                arrays["vertices"].ndim == 2
                and arrays["vertices"].shape[1] == 3
                and arrays["faces"].ndim == 2
                and arrays["faces"].shape[1] == 3
                and arrays["surface_wavefunction"].shape
                == (arrays["vertices"].shape[0],)
                and np.isfinite(threshold)
                and np.isfinite(achieved)
            ):
                print("Persistent cache hit: mmap isosurface mesh.")
                return arrays, threshold, achieved, True
        except (KeyError, TypeError, ValueError):
            pass
        print("Isosurface-mesh cache dimensions were invalid; rebuilding it.")

    spacing = float(axis[1] - axis[0])
    threshold, _unfiltered_achieved = find_probability_threshold(density, spacing)
    mesh_density, removed_voxels = filter_isosurface_density(density, threshold)
    total_probability = float(density.sum(dtype=np.float64) * spacing**3)
    achieved = float(
        mesh_density[mesh_density >= threshold].sum(dtype=np.float64)
        * spacing**3
        / total_probability
    )
    if removed_voxels:
        print(
            f"Removed {removed_voxels} sub-resolution isosurface voxels "
            "before meshing."
        )
    vertices, faces, _normals, _values = marching_cubes(
        writable_compiled_array(mesh_density, dtype=np.float32),
        level=threshold,
        spacing=(spacing, spacing, spacing),
        step_size=MARCHING_CUBES_STEP,
        allow_degenerate=False,
    )
    vertices += float(axis[0])
    surface_wavefunction = interpolate_surface_values(axis, wavefunction, vertices)
    arrays = {
        "vertices": np.asarray(vertices, dtype=np.float32),
        "faces": np.asarray(faces, dtype=np.int32),
        "surface_wavefunction": np.asarray(
            surface_wavefunction,
            dtype=(
                np.complex64
                if np.iscomplexobj(surface_wavefunction)
                else np.float32
            ),
        ),
    }
    if atomic_save_array_bundle(
        path,
        arrays,
        {"threshold": threshold, "achieved": achieved},
    ):
        remapped = load_array_bundle(path, required)
        if remapped is not None:
            return remapped[0], threshold, achieved, False
    return arrays, threshold, achieved, False


