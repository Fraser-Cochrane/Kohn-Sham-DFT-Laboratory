# =============================================================================
# PART 4 OF 9: SPHERICAL DENSITY, KOHN-SHAM POTENTIAL, AND RADIAL SOLVER
# =============================================================================
#
# Converts the atomic spin density into radial fields, constructs local
# Kohn-Sham potentials, solves radial orbital families, and caches reusable
# radial states independently of the later three-dimensional representation.
#
from __future__ import annotations

@lru_cache(maxsize=8)
def fibonacci_sphere(number: int) -> np.ndarray:
    """Generate nearly equal-area unit directions."""
    indices = np.arange(number, dtype=float)
    z_coord = 1.0 - 2.0 * (indices + 0.5) / number
    radius = np.sqrt(np.maximum(0.0, 1.0 - z_coord**2))
    azimuth = np.pi * (3.0 - np.sqrt(5.0)) * indices
    directions = np.column_stack(
        (radius * np.cos(azimuth), radius * np.sin(azimuth), z_coord)
    )
    directions.setflags(write=False)
    return directions


def spherical_average_spin_density(
    mol: gto.Mole,
    dm_alpha: np.ndarray,
    dm_beta: np.ndarray,
    radial_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical-average alpha/beta DFT densities on a radial grid."""
    directions = fibonacci_sphere(ANGULAR_DIRECTIONS)
    density_alpha = np.empty_like(radial_grid)
    density_beta = np.empty_like(radial_grid)
    numerical_integrator = dft.numint.NumInt()
    dm_alpha_buffer = writable_compiled_array(dm_alpha)
    dm_beta_buffer = writable_compiled_array(dm_beta)

    for start in range(0, radial_grid.size, RADIAL_BLOCK_SIZE):
        stop = min(start + RADIAL_BLOCK_SIZE, radial_grid.size)
        radii = radial_grid[start:stop]
        coordinates = (radii[:, None, None] * directions[None, :, :]).reshape(-1, 3)
        ao_values = numerical_integrator.eval_ao(mol, coordinates, deriv=0)
        rho_alpha = numerical_integrator.eval_rho(
            mol, ao_values, dm_alpha_buffer, xctype="LDA", hermi=1
        )
        rho_beta = numerical_integrator.eval_rho(
            mol, ao_values, dm_beta_buffer, xctype="LDA", hermi=1
        )
        density_alpha[start:stop] = rho_alpha.reshape(-1, ANGULAR_DIRECTIONS).mean(axis=1)
        density_beta[start:stop] = rho_beta.reshape(-1, ANGULAR_DIRECTIONS).mean(axis=1)
        print(f"Spherical DFT density: {stop:5d}/{radial_grid.size}", end="\r")

    print()
    return np.maximum(density_alpha, 0.0), np.maximum(density_beta, 0.0)


def hartree_potential_from_radial_density(
    radial_grid: np.ndarray,
    total_density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return v_H(r) and enclosed electron number for spherical rho(r)."""
    enclosed = cumulative_trapezoid(
        4.0 * np.pi * total_density * radial_grid**2,
        radial_grid,
        initial=0.0,
    )
    outer_cumulative = cumulative_trapezoid(
        4.0 * np.pi * total_density * radial_grid,
        radial_grid,
        initial=0.0,
    )
    outer_remaining = outer_cumulative[-1] - outer_cumulative
    potential = outer_remaining.copy()
    nonzero = radial_grid > 0.0
    potential[nonzero] += enclosed[nonzero] / radial_grid[nonzero]
    return potential, enclosed


def extract_lda_xc_potential(
    density_alpha: np.ndarray,
    density_beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate spin-resolved local XC potentials with PySCF/LibXC."""
    numerical_integrator = dft.numint.NumInt()
    density_floor = 1.0e-20
    number = density_alpha.size

    if SPIN == 0:
        total_density = np.maximum(density_alpha + density_beta, density_floor)
        _exc, vxc = numerical_integrator.eval_xc_eff(
            DFT_FUNCTIONAL,
            total_density,
            deriv=1,
            xctype="LDA",
            spin=0,
        )[:2]
        potential = np.asarray(vxc, dtype=float).squeeze()
        if potential.shape != (number,):
            potential = potential.reshape(-1, number)[0]
        return potential, potential.copy()

    spin_density = np.vstack(
        (np.maximum(density_alpha, density_floor),
         np.maximum(density_beta, density_floor))
    )
    _exc, vxc = numerical_integrator.eval_xc_eff(
        DFT_FUNCTIONAL,
        spin_density,
        deriv=1,
        xctype="LDA",
        spin=1,
    )[:2]
    potential = np.asarray(vxc, dtype=float).squeeze()
    if potential.shape == (number, 2):
        potential = potential.T
    elif potential.shape != (2, number):
        potential = potential.reshape(2, -1, number)[:, 0, :]
    return potential[0], potential[1]


def construct_ks_potentials(
    atomic_number: int,
    radial_grid: np.ndarray,
    density_alpha: np.ndarray,
    density_beta: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct v_H, v_xc, v_KS, and Z_eff(r) for both spins."""
    total_density = density_alpha + density_beta
    hartree, enclosed = hartree_potential_from_radial_density(
        radial_grid, total_density
    )
    vxc_alpha, vxc_beta = extract_lda_xc_potential(density_alpha, density_beta)

    nuclear = np.empty_like(radial_grid)
    nuclear[0] = -np.inf
    nuclear[1:] = -atomic_number / radial_grid[1:]
    vks_alpha = nuclear + hartree + vxc_alpha
    vks_beta = nuclear + hartree + vxc_beta

    # Avoid 0 * infinity at the origin by applying the analytic limit Z_eff(0)=Z.
    zeff_alpha = np.empty_like(radial_grid)
    zeff_beta = np.empty_like(radial_grid)
    zeff_alpha[0] = float(atomic_number)
    zeff_beta[0] = float(atomic_number)
    zeff_alpha[1:] = -radial_grid[1:] * vks_alpha[1:]
    zeff_beta[1:] = -radial_grid[1:] * vks_beta[1:]

    return {
        "total_density": total_density,
        "enclosed_electrons": enclosed,
        "hartree": hartree,
        "vxc_alpha": vxc_alpha,
        "vxc_beta": vxc_beta,
        "vks_alpha": vks_alpha,
        "vks_beta": vks_beta,
        "zeff_alpha": zeff_alpha,
        "zeff_beta": zeff_beta,
    }


ATOMIC_CACHE_ARRAYS = (
    "radial_grid",
    "density_alpha",
    "density_beta",
    "total_density",
    "enclosed_electrons",
    "hartree",
    "vxc_alpha",
    "vxc_beta",
    "vks_alpha",
    "vks_beta",
    "zeff_alpha",
    "zeff_beta",
)


def load_or_calculate_atomic_fields(
    symbol: str,
    atomic_number: int,
    ionic_charge: int,
    atomic_key: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    float,
    dict[str, object],
    bool,
]:
    """Reuse the DFT radial fields, or calculate and persist them once.

    This is the boundary between electronic structure and orbital rendering.
    Everything downstream consumes spherical radial arrays, so changing m or a
    Plotly representation does not rerun PySCF.
    """
    path = cache_bundle("dft-fields", atomic_key)
    loaded = load_array_bundle(path, ATOMIC_CACHE_ARRAYS)
    if loaded is not None:
        cached, metadata = loaded
        try:
            calculation_info = metadata["calculation_info"]
            dft_energy = float(metadata["dft_energy"])
            if not isinstance(calculation_info, dict) or not np.isfinite(dft_energy):
                raise ValueError("atomic-field metadata is invalid")
            potentials = {
                name: cached[name]
                for name in ATOMIC_CACHE_ARRAYS
                if name not in {
                    "radial_grid",
                    "density_alpha",
                    "density_beta",
                }
            }
            print("Persistent cache hit: atomic DFT density and radial fields.")
            return (
                cached["radial_grid"],
                cached["density_alpha"],
                cached["density_beta"],
                potentials,
                dft_energy,
                calculation_info,
                True,
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"Atomic cache metadata was invalid; recalculating: {exc}")

    (
        mol,
        _mean_field,
        dm_alpha,
        dm_beta,
        dft_energy,
        calculation_info,
    ) = run_atomic_dft(symbol, ionic_charge)
    radial_grid = np.linspace(0.0, RADIAL_MAX_BOHR, RADIAL_POINTS + 1)
    density_alpha, density_beta = spherical_average_spin_density(
        mol, dm_alpha, dm_beta, radial_grid
    )
    potentials = construct_ks_potentials(
        atomic_number, radial_grid, density_alpha, density_beta
    )
    atomic_save_array_bundle(
        path,
        {
            "radial_grid": np.asarray(radial_grid, dtype=np.float64),
            "density_alpha": np.asarray(density_alpha, dtype=np.float64),
            "density_beta": np.asarray(density_beta, dtype=np.float64),
            **{
                name: np.asarray(value, dtype=np.float64)
                for name, value in potentials.items()
            },
        },
        {
            "dft_energy": float(dft_energy),
            "calculation_info": calculation_info,
        },
    )
    return (
        radial_grid,
        density_alpha,
        density_beta,
        potentials,
        dft_energy,
        calculation_info,
        False,
    )


def radial_ks_tridiagonal(
    radial_grid: np.ndarray,
    ks_potential: np.ndarray,
    angular_quantum_number: int,
    relativistic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the Hermitian radial KS tridiagonal for one value of l.

    The heavy-element path uses the scalar zeroth-order regular approximation
    (ZORA), whose position-dependent kinetic factor is

        K(r) = [1 - V_KS(r)/(2 c^2)]^-1.

    For K=1 the discretisation reduces exactly to the ordinary radial
    Schrodinger operator used for lighter elements.
    """
    interior_r = radial_grid[1:-1]
    spacing = float(radial_grid[1] - radial_grid[0])
    angular_factor = angular_quantum_number * (angular_quantum_number + 1)

    if relativistic:
        denominator = 1.0 - ks_potential / (2.0 * SPEED_OF_LIGHT_AU**2)
        kinetic_factor = np.zeros_like(denominator)
        finite_positive = np.isfinite(denominator) & (denominator > 0.0)
        kinetic_factor[finite_positive] = 1.0 / denominator[finite_positive]
        if np.any(kinetic_factor[1:] <= 0.0):
            raise RuntimeError(
                "The scalar-ZORA kinetic factor became non-positive. "
                "Increase the radial resolution or inspect the KS potential."
            )

        half_factor = 0.5 * (kinetic_factor[:-1] + kinetic_factor[1:])
        left_factor = half_factor[:-1]
        right_factor = half_factor[1:]
        factor_derivative = np.gradient(kinetic_factor, spacing)[1:-1]
        centrifugal = (
            kinetic_factor[1:-1] * angular_factor / (2.0 * interior_r**2)
        )
        diagonal = (
            (left_factor + right_factor) / (2.0 * spacing**2)
            + factor_derivative / (2.0 * interior_r)
            + centrifugal
            + ks_potential[1:-1]
        )
        off_diagonal = -half_factor[1:-1] / (2.0 * spacing**2)
    else:
        centrifugal = angular_factor / (2.0 * interior_r**2)
        diagonal = 1.0 / spacing**2 + centrifugal + ks_potential[1:-1]
        off_diagonal = np.full(interior_r.size - 1, -0.5 / spacing**2)

    return interior_r, diagonal, off_diagonal


def solve_radial_ks_family(
    radial_grid: np.ndarray,
    ks_potential: np.ndarray,
    angular_quantum_number: int,
    maximum_principal_quantum_number: int,
    relativistic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve every radial state n=l+1,...,n_max with one eigensolver call."""
    if angular_quantum_number < 0:
        raise ValueError("The angular quantum number l cannot be negative.")
    if maximum_principal_quantum_number <= angular_quantum_number:
        raise ValueError("The maximum principal quantum number must exceed l.")

    interior_r, diagonal, off_diagonal = radial_ks_tridiagonal(
        radial_grid,
        ks_potential,
        angular_quantum_number,
        relativistic=relativistic,
    )
    number_of_states = maximum_principal_quantum_number - angular_quantum_number
    if number_of_states > diagonal.size:
        raise ValueError(
            f"Requested {number_of_states} radial states, but the fast radial "
            f"grid supports at most {diagonal.size}."
        )
    energies, eigenvectors = eigh_tridiagonal(
        writable_compiled_array(diagonal, dtype=np.float64),
        writable_compiled_array(off_diagonal, dtype=np.float64),
        select="i",
        select_range=(0, number_of_states - 1),
        check_finite=True,
    )
    radial_functions = np.empty((number_of_states, interior_r.size), dtype=float)
    for state_index in range(number_of_states):
        reduced_radial = eigenvectors[:, state_index]
        reduced_radial /= np.sqrt(
            trapezoid(np.abs(reduced_radial) ** 2, interior_r)
        )
        largest = int(np.argmax(np.abs(reduced_radial)))
        if reduced_radial[largest] < 0.0:
            reduced_radial *= -1.0
        radial_functions[state_index] = reduced_radial / interior_r
    return np.asarray(energies, dtype=float), interior_r, radial_functions


def solve_radial_ks_orbital(
    radial_grid: np.ndarray,
    ks_potential: np.ndarray,
    relativistic: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compatibility wrapper returning the currently selected (n,l) state."""
    energies, interior_r, radial_functions = solve_radial_ks_family(
        radial_grid,
        ks_potential,
        L_QUANTUM,
        N_QUANTUM,
        relativistic=relativistic,
    )
    state_index = N_QUANTUM - L_QUANTUM - 1
    return (
        float(energies[state_index]),
        interior_r,
        radial_functions[state_index],
    )


def load_or_build_common_orbital_bank(
    atomic_key: str,
    radial_grid: np.ndarray,
    potentials: dict[str, np.ndarray],
    relativistic: bool,
) -> tuple[float, np.ndarray, np.ndarray, bool, str, str]:
    """Load one mmap radial family and return its selected n/l state."""
    family_indices_valid = (
        1 <= N_QUANTUM <= COMMON_ORBITAL_MAX_N
        and 0 <= L_QUANTUM < N_QUANTUM
        and ORBITAL_SPIN.upper() in {"ALPHA", "BETA"}
    )
    requested_spin = ORBITAL_SPIN.upper()
    selected_vks = potentials[f"vks_{requested_spin.lower()}"]
    shared_spin_potential = np.array_equal(
        potentials["vks_alpha"],
        potentials["vks_beta"],
    )
    cached_spin = "SHARED" if shared_spin_potential else requested_spin
    family_key = radial_family_cache_key(
        atomic_key,
        cached_spin,
        L_QUANTUM,
        relativistic,
    )
    state_key = radial_state_cache_key(family_key, N_QUANTUM, L_QUANTUM)

    if not family_indices_valid:
        energy, interior_r, radial_function = solve_radial_ks_orbital(
            radial_grid,
            selected_vks,
            relativistic=relativistic,
        )
        return energy, interior_r, radial_function, False, family_key, state_key

    required = ("interior_r", "energies", "radial_functions")
    family_path = cache_bundle("radial-families", family_key)
    loaded = load_array_bundle(family_path, required)
    state_count = COMMON_ORBITAL_MAX_N - L_QUANTUM
    expected_function_shape = (state_count, radial_grid.size - 2)
    selected_index = N_QUANTUM - L_QUANTUM - 1

    if loaded is not None:
        family_cached, _metadata = loaded
        family_energies = family_cached["energies"]
        family_functions = family_cached["radial_functions"]
        interior_r = family_cached["interior_r"]
        if (
            family_energies.shape == (state_count,)
            and family_functions.shape == expected_function_shape
            and interior_r.shape == (radial_grid.size - 2,)
            and np.isfinite(family_energies[selected_index])
            and np.any(family_functions[selected_index])
        ):
            print(
                "Persistent cache hit: requested radial-orbital family "
                f"({cached_spin.lower()}, l={L_QUANTUM})."
            )
            return (
                float(family_energies[selected_index]),
                interior_r,
                family_functions[selected_index],
                True,
                family_key,
                state_key,
            )
        print("Radial-orbital family cache was invalid; rebuilding it.")

    print(
        "Building only the requested radial-orbital family "
        f"({cached_spin.lower()}, l={L_QUANTUM}, n<={COMMON_ORBITAL_MAX_N})."
    )
    try:
        family_energies, interior_r, family_functions = solve_radial_ks_family(
            radial_grid,
            selected_vks,
            L_QUANTUM,
            COMMON_ORBITAL_MAX_N,
            relativistic=relativistic,
        )
        atomic_save_array_bundle(
            family_path,
            {
                "interior_r": np.asarray(interior_r, dtype=np.float64),
                "energies": np.asarray(family_energies, dtype=np.float64),
                "radial_functions": np.asarray(
                    family_functions, dtype=np.float64
                ),
            },
            {
                "atomic_key": atomic_key,
                "spin_dependency": cached_spin,
                "l": L_QUANTUM,
                "maximum_n": COMMON_ORBITAL_MAX_N,
                "relativistic": bool(relativistic),
            },
        )
        return (
            float(family_energies[selected_index]),
            interior_r,
            family_functions[selected_index],
            False,
            family_key,
            state_key,
        )
    except Exception as exc:
        print(
            "Radial-orbital family caching was unavailable; solving only the "
            f"requested state ({type(exc).__name__}: {exc})."
        )
        energy, interior_r, radial_function = solve_radial_ks_orbital(
            radial_grid,
            selected_vks,
            relativistic=relativistic,
        )
        return energy, interior_r, radial_function, False, family_key, state_key


