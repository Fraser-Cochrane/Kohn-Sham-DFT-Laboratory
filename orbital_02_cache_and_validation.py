# =============================================================================
# PART 2 OF 9: CACHE, RUNTIME ESTIMATION, INPUT VALIDATION, AND BASIS SELECTION
# =============================================================================
#
# This layer owns physical-dependency cache keys, safe atomic writes, runtime
# calibration, ion parsing, input checks, and relativistic basis selection.
# It relies on imports and settings established by orbital_01_settings.py.
#
from __future__ import annotations

CACHE_LOCK = threading.RLock()
_CACHE_WARNING_SHOWN = False


def cache_root() -> Path | None:
    """Return the writable persistent-cache directory, or disable caching safely."""
    global _CACHE_WARNING_SHOWN
    if not ENABLE_PERSISTENT_CACHE:
        return None
    root = CACHE_DIRECTORY.expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if not _CACHE_WARNING_SHOWN:
            print(f"Cache disabled because {root} is not writable: {exc}")
            _CACHE_WARNING_SHOWN = True
        return None
    return root


def uncalibrated_runtime_seconds(
    atomic_number: int,
    electron_count: int,
    spin_2s: int,
    ionic_charge: int,
    quality_level: int,
) -> float:
    """Model an optimized cold run before applying laptop calibration.

    Rendering cost follows the actual radial and Cartesian sample counts. The
    electronic term is deliberately piecewise: low-quality difficult elements
    use bounded preview work, whereas levels 4-5 and ordinary elements include
    grid, SCF-cycle, spin, charge and relativistic cost. Measured uncached runs
    later multiply this hardware-independent estimate by a robust local factor.
    """
    profile = QUALITY_PROFILES[quality_level]
    electrons = max(electron_count, 1)
    surface_ratio = (float(profile["surface_grid_points"]) / 81.0) ** 3
    radial_ratio = (float(profile["radial_points"]) / 1_600.0) ** 0.45
    initial_render_seconds = 0.16 + 0.14 * surface_ratio + 0.05 * radial_ratio

    preview_element = (
        atomic_number >= SUPERHEAVY_FAST_PATH_Z
        or any(
            lower <= atomic_number <= upper
            for lower, upper in F_BLOCK_PREVIEW_RANGES
        )
    )
    optimized_preview = preview_element and quality_level <= 3
    ion_factor = 1.0 + 0.05 * min(abs(ionic_charge), 4)
    if optimized_preview:
        atom_factor = 0.82 + 0.0025 * electrons
        preview_grid_factor = (
            float(profile["heavy_preview_grid_level"]) + 1.0
        ) ** 1.35
        preview_seconds = (
            0.08
            + 0.12
            * float(profile["heavy_preview_fock_updates"])
            * preview_grid_factor
            * atom_factor
            * ion_factor
        )
        density_seconds = min(
            0.55 * float(profile["heavy_preview_time_budget_seconds"]),
            preview_seconds,
        )
    else:
        electronic_cost = 0.12 + 0.009 * electrons**1.35
        grid_factor = (
            max(float(profile["dft_grid_level"]), 1.0) / 3.0
        ) ** 1.45
        cycle_factor = (
            max(float(profile["scf_max_cycles"]), 80.0) / 180.0
        ) ** 0.30
        open_shell_factor = 1.0 + 0.025 * min(abs(spin_2s), 8)
        relativistic_factor = (
            1.20 if atomic_number >= RELATIVISTIC_Z_THRESHOLD else 1.0
        )
        density_seconds = (
            electronic_cost
            * grid_factor
            * cycle_factor
            * open_shell_factor
            * ion_factor
            * relativistic_factor
        )
    return float(max(0.20, initial_render_seconds + density_seconds))


def runtime_history_path() -> Path | None:
    root = cache_root()
    return None if root is None else root / "runtime-history.json"


def read_runtime_history() -> list[dict[str, object]]:
    """Read bounded local timing samples without ever blocking a calculation."""
    path = runtime_history_path()
    if path is None or not path.is_file():
        return []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        records = document.get("records", []) if isinstance(document, dict) else []
        return [record for record in records if isinstance(record, dict)][-80:]
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []


def laptop_runtime_calibration() -> tuple[float, int]:
    """Estimate this laptop's speed from successful uncached calculations."""
    ratios: list[float] = []
    for record in read_runtime_history():
        try:
            if not bool(record.get("cold_run", False)):
                continue
            if int(record.get("runtime_model_version", 0)) != RUNTIME_MODEL_VERSION:
                continue
            measured = float(record["seconds"])
            modelled = float(record["model_seconds"])
            if measured > 0.0 and modelled > 0.0:
                ratios.append(measured / modelled)
        except (KeyError, TypeError, ValueError):
            continue
    if not ratios:
        return 1.0, 0
    # A median is resistant to one unusually difficult SCF convergence path.
    factor = float(np.clip(np.median(ratios[-30:]), 0.25, 4.0))
    return factor, len(ratios[-30:])


def record_runtime_sample(
    *,
    atomic_number: int,
    electron_count: int,
    spin_2s: int,
    ionic_charge: int,
    quality_level: int,
    seconds: float,
    cold_run: bool,
) -> None:
    """Persist one laptop timing sample; failure never affects the result."""
    path = runtime_history_path()
    if path is None or not np.isfinite(seconds) or seconds <= 0.0:
        return
    model_seconds = uncalibrated_runtime_seconds(
        atomic_number,
        electron_count,
        spin_2s,
        ionic_charge,
        quality_level,
    )
    record = {
        "timestamp": time.time(),
        "atomic_number": atomic_number,
        "electron_count": electron_count,
        "spin_2s": spin_2s,
        "ionic_charge": ionic_charge,
        "quality_level": quality_level,
        "runtime_model_version": RUNTIME_MODEL_VERSION,
        "seconds": float(seconds),
        "model_seconds": model_seconds,
        "cold_run": bool(cold_run),
    }
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with CACHE_LOCK:
            records = read_runtime_history()
            records.append(record)
            temporary.write_text(
                json.dumps({"records": records[-80:]}, separators=(",", ":")),
                encoding="utf-8",
            )
            temporary.replace(path)
    except OSError as exc:
        print(f"Runtime calibration sample was not saved: {exc}")
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def stable_cache_key(category: str, parameters: dict[str, object]) -> str:
    """Hash all numerical assumptions into a deterministic cache identifier.

    Stable JSON ordering makes the same physical request reproduce the same
    digest across processes and restarts. CACHE_FORMAT_VERSION invalidates only
    layouts whose stored representation has changed.
    """
    document = {
        "category": category,
        "format_version": CACHE_FORMAT_VERSION,
        "parameters": parameters,
    }
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_preview_policy(atomic_number: int) -> str:
    """Identify atoms that always retain a deterministic fallback density.

    This function identifies availability, not whether SCF runs: the quality
    branch inside run_atomic_dft() uses the fallback directly at levels 1-3 and
    attempts fixed-ensemble SCF first at levels 4-5.
    """
    if atomic_number >= SUPERHEAVY_FAST_PATH_Z:
        return "superheavy-sap-fock-v3"
    if any(lower <= atomic_number <= upper for lower, upper in F_BLOCK_PREVIEW_RANGES):
        return "f-block-sap-fock-v3"
    return "scf"


def dft_cache_key(symbol: str, ionic_charge: int) -> str:
    """Key atomic radial fields only by inputs that can change the physics.

    Plot choices such as m, colours and active tab are intentionally absent.
    Consequently, changing a representation can reuse the expensive atomic
    density, while a change to basis, spin, SCF policy or sampling invalidates it.
    """
    atomic_number = ATOMIC_NUMBERS[symbol]
    _basis_specification, resolved_basis, resolved_relativistic = (
        select_orbital_basis(symbol, ATOMIC_NUMBERS[symbol])
    )
    heavy_atom = atomic_number >= RELATIVISTIC_Z_THRESHOLD
    high_accuracy_heavy = heavy_atom and QUALITY_LEVEL >= 4
    effective_grid_level = (
        DFT_GRID_LEVEL
        if not heavy_atom or high_accuracy_heavy
        else min(DFT_GRID_LEVEL, 3)
    )
    effective_tolerance = (
        SCF_TOLERANCE
        if not heavy_atom or high_accuracy_heavy
        else max(SCF_TOLERANCE, 5.0e-8)
    )
    preview_policy = deterministic_preview_policy(atomic_number)
    high_accuracy_ensemble_scf = preview_policy != "scf" and QUALITY_LEVEL >= 4
    return stable_cache_key(
        "atomic-dft-fields",
        {
            "pyscf": getattr(pyscf, "__version__", "unknown"),
            "symbol": symbol,
            "atomic_number": atomic_number,
            "ionic_charge": ionic_charge,
            "spin_2s": SPIN,
            "basis": BASIS,
            "resolved_basis": resolved_basis,
            "resolved_relativistic": resolved_relativistic,
            "relativistic_threshold": RELATIVISTIC_Z_THRESHOLD,
            "relativistic_bases": RELATIVISTIC_BASIS_CANDIDATES,
            "speed_of_light_au": SPEED_OF_LIGHT_AU,
            "functional": DFT_FUNCTIONAL,
            "dft_grid_level": effective_grid_level,
            "scf_tolerance": effective_tolerance,
            "scf_max_cycles": SCF_MAX_CYCLES,
            "scf_newton_max_cycles": SCF_NEWTON_MAX_CYCLES,
            "scf_cycle_policy_version": SCF_CYCLE_POLICY_VERSION,
            "occupation_algorithm": 3,
            **(
                {
                    "deterministic_preview_policy": preview_policy,
                    "preview_fock_updates": HEAVY_PREVIEW_MAX_FOCK_UPDATES,
                    "preview_grid_level": HEAVY_PREVIEW_GRID_LEVEL,
                    "preview_time_budget_seconds": HEAVY_PREVIEW_TIME_BUDGET_SECONDS,
                    "preview_new_density_weight": HEAVY_PREVIEW_NEW_DENSITY_WEIGHT,
                    "high_accuracy_ensemble_scf": high_accuracy_ensemble_scf,
                    "ensemble_energy_tolerance": HEAVY_ENSEMBLE_ENERGY_TOLERANCE,
                }
                if preview_policy != "scf"
                else {}
            ),
            "radial_max_bohr": RADIAL_MAX_BOHR,
            "radial_points": RADIAL_POINTS,
            "angular_directions": ANGULAR_DIRECTIONS,
        },
    )


def radial_family_cache_key(
    atomic_key: str,
    spin_dependency: str,
    angular_quantum_number: int,
    relativistic: bool,
) -> str:
    """Key one spin/l radial eigensystem independently of n, m, and rendering."""
    return stable_cache_key(
        "radial-orbital-family",
        {
            "atomic_key": atomic_key,
            "spin": spin_dependency.upper(),
            "l": angular_quantum_number,
            "maximum_n": COMMON_ORBITAL_MAX_N,
            "relativistic": bool(relativistic),
        },
    )


def radial_state_cache_key(family_key: str, principal: int, angular: int) -> str:
    """Key one n/l state as a lightweight dependency on its cached family."""
    return stable_cache_key(
        "radial-orbital-state",
        {"family_key": family_key, "n": principal, "l": angular},
    )


def selected_radial_state_dependency_key(atomic_key: str) -> str:
    """Return a prospective state key before spin sharing is known."""
    logical_family = radial_family_cache_key(
        atomic_key,
        ORBITAL_SPIN.upper(),
        L_QUANTUM,
        ATOMIC_NUMBERS[parse_ion_name(ION)[0]] >= RELATIVISTIC_Z_THRESHOLD,
    )
    return radial_state_cache_key(logical_family, N_QUANTUM, L_QUANTUM)


def angular_state_cache_key() -> str:
    """Key Y_l^m independently of atom, radial state, and plot resolution."""
    return stable_cache_key(
        "angular-orbital-state",
        {"l": L_QUANTUM, "m": M_QUANTUM, "form": ORBITAL_FORM.upper()},
    )


def angular_plot_cache_key(angular_key: str) -> str:
    return stable_cache_key(
        "angular-plot-data",
        {
            "angular_key": angular_key,
            "theta_points": ANGULAR_PLOT_THETA_POINTS,
            "phi_points": ANGULAR_PLOT_PHI_POINTS,
        },
    )


def spatial_grid_cache_key(radial_state_key: str, angular_key: str) -> str:
    """Key the Cartesian wavefunction grid without mesh or view settings."""
    return stable_cache_key(
        "cartesian-orbital-grid",
        {
            "radial_state_key": radial_state_key,
            "angular_key": angular_key,
            "surface_grid_points": SURFACE_GRID_POINTS,
            "surface_box_half_width": SURFACE_BOX_HALF_WIDTH_BOHR,
        },
    )


def isosurface_mesh_cache_key(spatial_key: str) -> str:
    return stable_cache_key(
        "isosurface-mesh",
        {
            "spatial_grid_key": spatial_key,
            "enclosed_fraction": ENCLOSED_FRACTION,
            "marching_cubes_step": MARCHING_CUBES_STEP,
            "surface_filter_version": SURFACE_FILTER_VERSION,
            "component_min_voxels": SURFACE_COMPONENT_MIN_VOXELS,
            "component_min_fraction": SURFACE_COMPONENT_MIN_FRACTION,
        },
    )


def dot_data_cache_key(spatial_key: str) -> str:
    return stable_cache_key(
        "density-dot-data",
        {
            "spatial_grid_key": spatial_key,
            "dot_points": DOT_MAP_POINTS,
            "dot_seed": DOT_MAP_SEED,
        },
    )


def contour_data_cache_key(spatial_key: str) -> str:
    return stable_cache_key(
        "orbital-contour-data",
        {"spatial_grid_key": spatial_key},
    )


def representation_panel_keys(
    atomic_key: str,
    radial_state_key_value: str,
    angular_key: str,
    spatial_key: str,
) -> dict[str, str]:
    """Return independent figure keys for the five physical view branches."""
    mesh_key = isosurface_mesh_cache_key(spatial_key)
    dot_key = dot_data_cache_key(spatial_key)
    contour_key = contour_data_cache_key(spatial_key)
    angular_plot_key_value = angular_plot_cache_key(angular_key)
    title_context = {
        "ion": ION,
        "n": N_QUANTUM,
        "l": L_QUANTUM,
        "m": M_QUANTUM,
        "form": ORBITAL_FORM.upper(),
    }
    return {
        "isosurface": stable_cache_key(
            "isosurface-figure",
            {
                "mesh_key": mesh_key,
                "surface_alpha": SURFACE_ALPHA,
                "title": title_context,
                "render_version": RESULT_RENDER_VERSION,
            },
        ),
        "dot-map": stable_cache_key(
            "density-dot-figure",
            {
                "dot_data_key": dot_key,
                "title": title_context,
                "render_version": RESULT_RENDER_VERSION,
            },
        ),
        "radial": stable_cache_key(
            "radial-figure",
            {
                "radial_state_key": radial_state_key_value,
                "render_version": RESULT_RENDER_VERSION,
            },
        ),
        "angular": stable_cache_key(
            "angular-figure",
            {
                "angular_plot_key": angular_plot_key_value,
                "render_version": RESULT_RENDER_VERSION,
            },
        ),
        "contours": stable_cache_key(
            "contour-figure",
            {
                "contour_data_key": contour_key,
                "render_version": RESULT_RENDER_VERSION,
            },
        ),
    }


def result_cache_key(atomic_key: str) -> str:
    """Key the page from independent prospective physical view dependencies."""
    radial_key = selected_radial_state_dependency_key(atomic_key)
    angular_key = angular_state_cache_key()
    spatial_key = spatial_grid_cache_key(radial_key, angular_key)
    return stable_cache_key(
        "rendered-orbital-result",
        {
            "panel_keys": representation_panel_keys(
                atomic_key, radial_key, angular_key, spatial_key
            ),
            "isotope": ISOTOPE_MASS_NUMBER,
            "render_version": RESULT_RENDER_VERSION,
        },
    )


def cache_file(category: str, key: str, suffix: str) -> Path | None:
    root = cache_root()
    if root is None:
        return None
    directory = root / category
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return directory / f"{key}{suffix}"


def cache_bundle(category: str, key: str) -> Path | None:
    """Return a directory whose members are independently mmap-able NPY arrays."""
    root = cache_root()
    if root is None:
        return None
    directory = root / category
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return directory / key


def _cache_entry_size_and_time(path: Path) -> tuple[int, float]:
    if path.is_file():
        stat = path.stat()
        return stat.st_size, stat.st_mtime
    files = [item for item in path.rglob("*") if item.is_file()]
    if not files:
        return 0, path.stat().st_mtime
    stats = [item.stat() for item in files]
    return sum(stat.st_size for stat in stats), max(stat.st_mtime for stat in stats)


def _cache_entry_is_excluded(entry: Path, exclude: Path | None) -> bool:
    if exclude is None:
        return False
    try:
        return entry == exclude or exclude.is_relative_to(entry)
    except (AttributeError, ValueError):
        try:
            exclude.relative_to(entry)
            return True
        except ValueError:
            return entry == exclude


def prune_cache(exclude: Path | None = None) -> None:
    """Bound disk use while treating every NPY bundle as one cache entry."""
    root = cache_root()
    if root is None:
        return
    try:
        entries: list[Path] = []
        for category in root.iterdir():
            if category.name.startswith("."):
                continue
            if category.is_file():
                entries.append(category)
                continue
            entries.extend(
                child
                for child in category.iterdir()
                if not child.name.startswith(".")
            )
        records = []
        for entry in entries:
            size, modified = _cache_entry_size_and_time(entry)
            records.append((modified, size, entry))
        total = sum(size for _modified, size, _entry in records)
        for _modified, size, entry in sorted(records):
            if total <= CACHE_MAX_BYTES:
                break
            if _cache_entry_is_excluded(entry, exclude):
                continue
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                total -= size
            except OSError:
                continue
    except OSError:
        return


def atomic_save_array_bundle(
    path: Path | None,
    arrays: dict[str, object],
    metadata: dict[str, object] | None = None,
) -> bool:
    """Atomically write independent uncompressed NPY files plus JSON metadata."""
    if path is None:
        return False
    if any(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None for name in arrays):
        raise ValueError("Cache array names must be simple identifiers.")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    backup = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.old"
    )
    with CACHE_LOCK:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)
            temporary.mkdir()
            manifest_arrays: dict[str, dict[str, object]] = {}
            for name, value in arrays.items():
                array = np.ascontiguousarray(np.asarray(value))
                with (temporary / f"{name}.npy").open("wb") as handle:
                    np.save(handle, array, allow_pickle=False)
                manifest_arrays[name] = {
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                }
            manifest = {
                "format_version": CACHE_FORMAT_VERSION,
                "arrays": manifest_arrays,
                "metadata": metadata or {},
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            if path.exists():
                path.replace(backup)
            temporary.replace(path)
            shutil.rmtree(backup, ignore_errors=True)
            prune_cache(exclude=path)
            return True
        except (OSError, TypeError, ValueError) as exc:
            print(f"Cache bundle write skipped for {path.name}: {exc}")
            if backup.exists() and not path.exists():
                try:
                    backup.replace(path)
                except OSError:
                    pass
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)
            return False


def load_array_bundle(
    path: Path | None,
    required_arrays: tuple[str, ...],
    *,
    mmap: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, object]] | None:
    """Load only requested NPY members, using read-only memory maps by default."""
    if path is None or not path.is_dir():
        return None
    try:
        with CACHE_LOCK:
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format_version") != CACHE_FORMAT_VERSION:
                raise ValueError("cache bundle version is stale")
            array_specs = manifest.get("arrays")
            metadata = manifest.get("metadata", {})
            if not isinstance(array_specs, dict) or not isinstance(metadata, dict):
                raise ValueError("cache bundle manifest is invalid")
            if any(name not in array_specs for name in required_arrays):
                raise ValueError("cache bundle is incomplete")
            result: dict[str, np.ndarray] = {}
            for name in required_arrays:
                spec = array_specs[name]
                array = np.load(
                    path / f"{name}.npy",
                    mmap_mode="r" if mmap else None,
                    allow_pickle=False,
                )
                if list(array.shape) != spec.get("shape") or array.dtype.str != spec.get("dtype"):
                    raise ValueError(f"cached array {name!r} disagrees with its manifest")
                result[name] = array
            now = time.time()
            os.utime(manifest_path, (now, now))
            os.utime(path, (now, now))
            return result, metadata
    except (OSError, TypeError, ValueError, EOFError, json.JSONDecodeError) as exc:
        print(f"Ignoring invalid cache bundle {path.name}: {exc}")
        try:
            shutil.rmtree(path)
        except OSError:
            pass
        return None


def writable_compiled_array(
    value: object,
    dtype: object | None = None,
) -> np.ndarray:
    """Copy a read-only/misaligned cache view only at a compiled-code boundary."""
    array = np.asarray(value, dtype=dtype)
    if (
        array.flags.writeable
        and array.flags.c_contiguous
        and array.flags.aligned
    ):
        return array
    return np.array(array, dtype=dtype, order="C", copy=True)


def atomic_write_text(path: Path, text: str) -> None:
    """Write HTML atomically so an interrupted render cannot poison the cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def parse_ion_name(ion_name: str) -> tuple[str, int, int]:
    """Return element symbol, nuclear charge Z, and signed ionic charge."""
    compact = re.sub(r"\s+", "", ion_name)
    patterns = (
        r"^([A-Za-z]{1,2})(\d+)([+-])$",
        r"^([A-Za-z]{1,2})([+-])(\d+)$",
        r"^([A-Za-z]{1,2})([+-]+)$",
        r"^([A-Za-z]{1,2})$",
    )
    match = None
    pattern_index = -1
    for pattern_index, pattern in enumerate(patterns):
        match = re.fullmatch(pattern, compact)
        if match:
            break
    if match is None:
        raise ValueError(
            f"Could not parse ION={ion_name!r}; try 'Ne', 'Cl-', or 'Fe2+'."
        )

    symbol = match.group(1).capitalize()
    if symbol not in ATOMIC_NUMBERS:
        raise ValueError(f"Unknown element symbol in ION={ion_name!r}.")

    if pattern_index == 0:
        magnitude, sign = int(match.group(2)), match.group(3)
    elif pattern_index == 1:
        sign, magnitude = match.group(2), int(match.group(3))
    elif pattern_index == 2:
        signs = match.group(2)
        if "+" in signs and "-" in signs:
            raise ValueError("ION cannot mix positive and negative signs.")
        sign, magnitude = signs[0], len(signs)
    else:
        sign, magnitude = "+", 0

    charge = magnitude if sign == "+" else -magnitude
    return symbol, ATOMIC_NUMBERS[symbol], charge


def physically_allowed_spin_values(electron_count: int) -> tuple[int, ...]:
    """Return 2S values giving non-negative integer alpha/beta populations."""
    if electron_count < 1:
        return ()
    return tuple(range(electron_count % 2, electron_count + 1, 2))


def validate_settings() -> tuple[str, int, int, int]:
    """Validate physics/numerics and return ion metadata."""
    symbol, atomic_number, ionic_charge = parse_ion_name(ION)
    electron_count = atomic_number - ionic_charge
    if electron_count < 1:
        raise ValueError("The named ion must contain at least one electron.")
    if SPIN not in physically_allowed_spin_values(electron_count):
        raise ValueError(
            f"ION={ION} has {electron_count} electrons, so 2S must lie between "
            f"{electron_count % 2} and {electron_count} in steps of two. "
            f"The selected value 2S={SPIN} would not give physical integer "
            "alpha/beta populations."
        )
    if ORBITAL_SPIN.upper() not in {"ALPHA", "BETA"}:
        raise ValueError("ORBITAL_SPIN must be 'ALPHA' or 'BETA'.")
    if not isinstance(N_QUANTUM, int) or N_QUANTUM < 1:
        raise ValueError("N_QUANTUM must be a positive integer.")
    if N_QUANTUM > COMMON_ORBITAL_MAX_N:
        raise ValueError(
            f"The interactive fast profile supports n <= {COMMON_ORBITAL_MAX_N}."
        )
    if not isinstance(L_QUANTUM, int) or not 0 <= L_QUANTUM < N_QUANTUM:
        raise ValueError("L_QUANTUM must satisfy 0 <= l < n.")
    if not isinstance(M_QUANTUM, int) or abs(M_QUANTUM) > L_QUANTUM:
        raise ValueError("M_QUANTUM must satisfy -l <= m <= l.")
    if ORBITAL_FORM.upper() not in {"REAL", "COMPLEX"}:
        raise ValueError("ORBITAL_FORM must be 'REAL' or 'COMPLEX'.")
    if RADIAL_MAX_BOHR <= 0.0 or RADIAL_POINTS < 300:
        raise ValueError("Use RADIAL_MAX_BOHR > 0 and RADIAL_POINTS >= 300.")
    if ANGULAR_DIRECTIONS < 12:
        raise ValueError("ANGULAR_DIRECTIONS must be at least 12.")
    if ANGULAR_PLOT_THETA_POINTS < 31 or ANGULAR_PLOT_PHI_POINTS < 61:
        raise ValueError("The angular-plot grids are too small for visualization.")
    if SURFACE_GRID_POINTS < 31:
        raise ValueError("SURFACE_GRID_POINTS must be at least 31.")
    if SURFACE_GRID_POINTS % 2 == 0 or SURFACE_GRID_BLOCK_SIZE < 1:
        raise ValueError(
            "Use an odd SURFACE_GRID_POINTS value and SURFACE_GRID_BLOCK_SIZE >= 1."
        )
    if QUALITY_LEVEL not in QUALITY_PROFILES:
        raise ValueError("QUALITY_LEVEL must be one of the five slider positions.")
    expected_profile_name = str(QUALITY_PROFILES[QUALITY_LEVEL]["name"])
    if QUALITY_PROFILE != expected_profile_name:
        raise ValueError(
            "The numerical quality settings are inconsistent; reselect the "
            "accuracy slider position."
        )
    if not 0.0 < ENCLOSED_FRACTION < 1.0:
        raise ValueError("ENCLOSED_FRACTION must lie between zero and one.")

    xc_type = dft.libxc.xc_type(DFT_FUNCTIONAL).upper()
    if xc_type != "LDA":
        raise ValueError(
            f"DFT_FUNCTIONAL={DFT_FUNCTIONAL!r} is {xc_type}, not LDA. "
            "This implementation requires an explicitly local LDA potential."
        )
    return symbol, atomic_number, ionic_charge, electron_count


@lru_cache(maxsize=None)
def select_orbital_basis(
    symbol: str,
    atomic_number: int,
) -> tuple[object, str, bool]:
    """Choose a compatible all-electron basis and relativistic treatment."""
    if atomic_number < RELATIVISTIC_Z_THRESHOLD:
        return BASIS, BASIS, False

    failures: list[str] = []
    for basis_name in RELATIVISTIC_BASIS_CANDIDATES:
        try:
            basis_data = gto.basis.load(basis_name, symbol)
        except Exception as exc:  # Basis availability depends on PySCF version.
            failures.append(f"{basis_name}: {type(exc).__name__}")
            continue
        return {symbol: basis_data}, basis_name, True

    details = "; ".join(failures)
    raise RuntimeError(
        f"No all-electron relativistic basis was found for {symbol}. "
        "Upgrade PySCF so that dyall-v2z is available, or install "
        f"basis-set-exchange. Basis checks: {details}."
    )