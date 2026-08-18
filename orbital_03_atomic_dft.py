# =============================================================================
# PART 3 OF 9: ATOMIC DFT, OCCUPATIONS, HEAVY-ELEMENT POLICY, AND FALLBACKS
# =============================================================================
#
# This is the electronic-structure engine.  It contains the SCF retry ladder,
# fixed average-of-configuration occupations, deterministic heavy-element
# preview density, and the safe density/energy fallbacks used by the UI.
#
from __future__ import annotations

def _normalise_density_electron_counts(
    density: np.ndarray,
    overlap: np.ndarray,
    target_counts: tuple[int, ...],
) -> np.ndarray:
    """Hermitise and scale density channels to their exact electron counts."""
    array = np.asarray(density)
    overlap_array = np.asarray(overlap)
    if overlap_array.ndim != 2 or overlap_array.shape[0] != overlap_array.shape[1]:
        raise ValueError("the AO overlap matrix must be square")
    if array.shape[-2:] != overlap_array.shape:
        raise ValueError("the density and AO overlap dimensions do not match")
    if len(target_counts) == 1 and array.ndim == 2:
        channels = (array,)
    elif len(target_counts) == 2 and array.ndim == 3 and array.shape[0] == 2:
        channels = (array[0], array[1])
    else:
        raise ValueError("density channels do not match the requested populations")

    normalised: list[np.ndarray] = []
    for channel, target in zip(channels, target_counts):
        hermitian = 0.5 * (channel + channel.conj().T)
        population_value = np.trace(hermitian @ overlap_array)
        population_real = float(np.real(population_value))
        population_imag = float(np.imag(population_value))
        if (
            not np.isfinite(population_real)
            or not np.isfinite(population_imag)
            or abs(population_imag) > 1.0e-8 * max(1.0, abs(population_real))
            or population_real <= 1.0e-12
        ):
            raise ValueError("density has an invalid AO-overlap population")
        scaled = hermitian * (float(target) / population_real)
        checked_population = float(np.real(np.trace(scaled @ overlap_array)))
        if not np.isfinite(checked_population) or not np.isclose(
            checked_population,
            float(target),
            rtol=1.0e-9,
            atol=1.0e-8,
        ):
            raise ValueError("density population normalisation failed")
        normalised.append(np.real_if_close(scaled))

    return normalised[0] if len(normalised) == 1 else np.stack(normalised)


def _canonical_occupied_density(
    hamiltonian: np.ndarray,
    overlap: np.ndarray,
    electron_count: int,
) -> np.ndarray:
    """Build an occupied AO density after removing overlap null directions."""
    hcore = np.asarray(hamiltonian)
    overlap_array = np.asarray(overlap)
    if hcore.ndim != 2 or hcore.shape[0] != hcore.shape[1]:
        raise ValueError("the one-electron Hamiltonian must be square")
    if overlap_array.shape != hcore.shape:
        raise ValueError("the Hamiltonian and AO overlap dimensions do not match")
    if electron_count <= 0:
        return np.zeros_like(hcore)

    overlap_hermitian = 0.5 * (overlap_array + overlap_array.conj().T)
    overlap_values, overlap_vectors = np.linalg.eigh(overlap_hermitian)
    largest_overlap = float(np.max(overlap_values))
    if not np.isfinite(largest_overlap) or largest_overlap <= 0.0:
        raise ValueError("the AO overlap matrix has no positive subspace")
    keep = overlap_values > max(1.0e-12, largest_overlap * 1.0e-10)
    if int(np.count_nonzero(keep)) < electron_count:
        positive_order = np.argsort(overlap_values)[::-1]
        selected = positive_order[:electron_count]
        if np.any(overlap_values[selected] <= 1.0e-14):
            raise ValueError("insufficient independent AO functions")
        keep = np.zeros_like(overlap_values, dtype=bool)
        keep[selected] = True

    orthogonaliser = overlap_vectors[:, keep] / np.sqrt(overlap_values[keep])
    hcore_hermitian = 0.5 * (hcore + hcore.conj().T)
    orthogonal_hamiltonian = (
        orthogonaliser.conj().T @ hcore_hermitian @ orthogonaliser
    )
    _energies, orthogonal_coefficients = np.linalg.eigh(
        0.5 * (orthogonal_hamiltonian + orthogonal_hamiltonian.conj().T)
    )
    if orthogonal_coefficients.shape[1] < electron_count:
        raise ValueError("insufficient canonical orbitals")
    occupied_coefficients = (
        orthogonaliser @ orthogonal_coefficients[:, :electron_count]
    )
    density = occupied_coefficients @ occupied_coefficients.conj().T
    population = float(np.real(np.trace(density @ overlap_hermitian)))
    if not np.isfinite(population) or not np.isclose(
        population,
        float(electron_count),
        rtol=1.0e-8,
        atol=1.0e-7,
    ):
        raise ValueError("canonical density has the wrong electron population")
    return np.real_if_close(0.5 * (density + density.conj().T))


def _overlap_metric_preview_density(
    overlap: np.ndarray,
    target_counts: tuple[int, ...],
) -> np.ndarray:
    """Construct a finite AO density using only the positive overlap metric."""
    overlap_array = np.asarray(overlap)
    if overlap_array.ndim != 2 or overlap_array.shape[0] != overlap_array.shape[1]:
        raise ValueError("the AO overlap matrix must be square")
    overlap_hermitian = 0.5 * (overlap_array + overlap_array.conj().T)
    values, vectors = np.linalg.eigh(overlap_hermitian)
    largest = float(np.max(values))
    if not np.isfinite(largest) or largest <= 0.0:
        raise ValueError("the AO overlap matrix has no positive subspace")
    keep = values > max(1.0e-14, largest * 1.0e-13)
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        raise ValueError("the AO overlap matrix has zero numerical rank")
    inverse_metric = (
        vectors[:, keep] / values[keep]
    ) @ vectors[:, keep].conj().T
    inverse_metric = 0.5 * (inverse_metric + inverse_metric.conj().T)
    metric_population = float(
        np.real(np.trace(inverse_metric @ overlap_hermitian))
    )
    if not np.isfinite(metric_population) or metric_population <= 1.0e-12:
        raise ValueError("the inverse overlap metric has zero population")

    channels: list[np.ndarray] = []
    for target in target_counts:
        density = inverse_metric * (float(target) / metric_population)
        population = float(np.real(np.trace(density @ overlap_hermitian)))
        if not np.isfinite(population) or not np.isclose(
            population,
            float(target),
            rtol=5.0e-7,
            atol=1.0e-6,
        ):
            raise ValueError("overlap-metric density has the wrong population")
        channels.append(np.real_if_close(density))
    return channels[0] if len(channels) == 1 else np.stack(channels)


def _thomas_fermi_preview_energy(
    atomic_number: int,
    electron_count: int,
) -> float:
    """Return a finite scale estimate used only as heavy-preview metadata."""
    if atomic_number <= 0 or electron_count <= 0:
        raise ValueError("atomic and electron counts must be positive")
    neutral_scale = -0.768745 * float(atomic_number) ** (7.0 / 3.0)
    return neutral_scale * (float(electron_count) / float(atomic_number))


def run_atomic_dft(
    symbol: str,
    ionic_charge: int,
) -> tuple[gto.Mole, object, np.ndarray, np.ndarray, float, dict[str, object]]:
    """Run robust atomic KS-DFT and return spin density matrices.

    Elements from Rb onward use an all-electron relativistic basis and a
    spin-free one-electron X2C Hamiltonian.  Heavy-atom SCF starts from a SAP
    density.  If ordinary convergence strategies fail, Fermi smearing is used
    only to generate a warm-start density.  The reported calculation is at
    zero temperature; exactly degenerate open shells may use ensemble
    occupations while preserving the selected alpha/beta populations. At
    Accurate and Maximum quality, f-block and superheavy atoms first attempt a
    fixed average-of-configuration ensemble SCF from the refined preview seed.
    """
    atomic_number = ATOMIC_NUMBERS[symbol]
    heavy_atom = atomic_number >= RELATIVISTIC_Z_THRESHOLD
    preview_policy = deterministic_preview_policy(atomic_number)
    superheavy_fast_path = preview_policy.startswith("superheavy-")
    f_block_preview_path = preview_policy.startswith("f-block-")

    # Three-way heavy-element policy:
    #   * normal atoms: staged integer/degenerate-shell SCF;
    #   * difficult atoms at levels 1-3: deterministic preview, no SCF kernel;
    #   * difficult atoms at levels 4-5: fixed-ensemble SCF, preview on failure.
    # Keeping availability separate from selection makes the last branch safe:
    # the same refined seed that starts SCF remains available if convergence fails.
    preview_fallback_available = preview_policy != "scf"
    high_accuracy_ensemble_scf = (
        preview_fallback_available and QUALITY_LEVEL >= 4
    )
    deterministic_preview_path = (
        preview_fallback_available and not high_accuracy_ensemble_scf
    )
    print(f"Atomic orbital engine revision: {ENGINE_REVISION}")
    basis_specification, basis_name, relativistic = select_orbital_basis(
        symbol, atomic_number
    )
    mol = gto.M(
        atom=f"{symbol} 0 0 0",
        unit="Bohr",
        charge=ionic_charge,
        spin=SPIN,
        basis=basis_specification,
        symmetry=False,
        max_memory=PYSCF_MAX_MEMORY_MB,
        verbose=PYSCF_VERBOSE,
    )
    alpha_electrons, beta_electrons = map(int, mol.nelec)
    number_orbitals = int(mol.nao_nr())
    if max(alpha_electrons, beta_electrons) > number_orbitals:
        raise ValueError(
            f"The selected 2S={SPIN} requires {max(alpha_electrons, beta_electrons)} "
            f"same-spin orbitals, but the {basis_name} basis supplies only "
            f"{number_orbitals}. Choose a lower spin value."
        )

    x2c_active = relativistic
    # Preview/Balanced retain a bounded heavy-atom SCF cost.  Accurate and
    # Maximum deliberately use their full quadrature and convergence settings.
    high_accuracy_heavy = heavy_atom and QUALITY_LEVEL >= 4
    scf_grid_level = (
        DFT_GRID_LEVEL
        if not heavy_atom or high_accuracy_heavy
        else min(DFT_GRID_LEVEL, 3)
    )
    scf_tolerance = (
        SCF_TOLERANCE
        if not heavy_atom or high_accuracy_heavy
        else max(SCF_TOLERANCE, 5.0e-8)
    )

    def stage_cycle_limit(stage: str) -> int:
        """Use short, escalating heavy-atom attempts instead of full retries."""
        if not heavy_atom:
            return SCF_MAX_CYCLES
        quality_bonus = 12 * (QUALITY_LEVEL - 1)
        budgets = {
            "standard": 40 + quality_bonus,
            "stabilised": 55 + quality_bonus,
            "smearing": 24 + quality_bonus // 2,
            "warm": 50 + quality_bonus,
            "roks": 50 + quality_bonus,
            "ensemble": 65 + quality_bonus,
        }
        return min(SCF_MAX_CYCLES, budgets.get(stage, 55 + quality_bonus))

    newton_cycle_limit = (
        min(SCF_NEWTON_MAX_CYCLES, 24 + 12 * QUALITY_LEVEL)
        if heavy_atom
        else SCF_NEWTON_MAX_CYCLES
    )
    common_density_parameters = {
        "pyscf": getattr(pyscf, "__version__", "unknown"),
        "symbol": symbol,
        "ionic_charge": ionic_charge,
        "spin_2s": SPIN,
        "resolved_basis": basis_name,
        "relativistic": relativistic,
        "heavy_atom": heavy_atom,
        "superheavy_fast_path": superheavy_fast_path,
        "f_block_preview_path": f_block_preview_path,
        "deterministic_preview_policy": preview_policy,
        "high_accuracy_ensemble_scf": high_accuracy_ensemble_scf,
        "ensemble_energy_tolerance": HEAVY_ENSEMBLE_ENERGY_TOLERANCE,
        "preview_fock_updates": HEAVY_PREVIEW_MAX_FOCK_UPDATES,
        "preview_grid_level": HEAVY_PREVIEW_GRID_LEVEL,
        "preview_time_budget_seconds": HEAVY_PREVIEW_TIME_BUDGET_SECONDS,
        "preview_new_density_weight": HEAVY_PREVIEW_NEW_DENSITY_WEIGHT,
        "engine_revision": ENGINE_REVISION,
        "finite_density_preview_algorithm": 5,
        "speed_of_light_au": SPEED_OF_LIGHT_AU,
        "functional": DFT_FUNCTIONAL,
        "occupation_algorithm": 3,
        "scf_max_cycles": SCF_MAX_CYCLES,
        "scf_newton_max_cycles": SCF_NEWTON_MAX_CYCLES,
        "scf_cycle_policy_version": SCF_CYCLE_POLICY_VERSION,
    }
    exact_density_key = stable_cache_key(
        "atomic-scf-density",
        {
            **common_density_parameters,
            "dft_grid_level": scf_grid_level,
            "scf_tolerance": scf_tolerance,
        },
    )
    warm_start_key = stable_cache_key(
        "atomic-scf-warm-start",
        common_density_parameters,
    )
    exact_density_path = cache_bundle("scf-densities", exact_density_key)
    warm_start_path = cache_bundle("scf-warm-starts", warm_start_key)

    def make_solver(kind: str, stage: str = "standard") -> object:
        nonlocal x2c_active
        if kind == "RKS":
            solver = dft.RKS(mol)
        elif kind == "UKS":
            solver = dft.UKS(mol)
        elif kind == "ROKS":
            solver = dft.ROKS(mol)
        else:  # Defensive guard for future edits.
            raise ValueError(f"Unknown Kohn-Sham solver kind {kind!r}.")
        if x2c_active:
            try:
                solver = solver.sfx2c1e()
            except Exception as exc:
                x2c_active = False
                print(
                    "X2C solver decoration failed safely; continuing with the "
                    "all-electron Hamiltonian and scalar-ZORA radial correction "
                    f"({type(exc).__name__}: {exc})."
                )
        solver.xc = DFT_FUNCTIONAL
        solver.grids.level = scf_grid_level
        solver.conv_tol = scf_tolerance
        solver.max_cycle = stage_cycle_limit(stage)
        # The built-in atomic/MINAO guess is not defined for every superheavy
        # element.  Heavy atoms receive an explicit SAP or one-electron density
        # below, and "1e" is the safe fallback if neither can be constructed.
        solver.init_guess = "1e" if heavy_atom else "minao"
        if heavy_atom:
            solver.sap_basis = "sapgrasplarge"
        solver.diis_space = 12
        solver.diis_start_cycle = 1
        solver.direct_scf = True
        return solver

    attempt_errors: list[str] = []
    last_valid_density: np.ndarray | None = None
    last_successful_solver: object | None = None
    last_finite_energy = float("nan")
    retained_seed_density: np.ndarray | None = None
    retained_seed_solver: object | None = None
    retained_seed_label = ""

    def validate_density(
        density: object,
        fallback: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Return a finite 2D/3D density matrix without propagating index errors."""
        try:
            array = np.asarray(density)
            if array.ndim not in {2, 3}:
                raise ValueError(f"unexpected density rank {array.ndim}")
            if array.shape[-1] != array.shape[-2] or array.shape[-1] == 0:
                raise ValueError(f"unexpected density shape {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError("density contains non-finite values")
            return array.copy()
        except (IndexError, TypeError, ValueError) as exc:
            print(f"Density update was unusable ({exc}); retaining the previous density.")
            return None if fallback is None else np.asarray(fallback).copy()

    def density_from_solver(
        solver: object,
        fallback: np.ndarray | None = None,
    ) -> np.ndarray | None:
        try:
            return validate_density(solver.make_rdm1(), fallback)
        except Exception as exc:
            print(
                "Density construction failed "
                f"({type(exc).__name__}: {exc}); retaining the previous density."
            )
            return None if fallback is None else np.asarray(fallback).copy()

    def solver_density_kind(solver: object) -> str:
        """Return the density-matrix convention expected by a solver."""
        try:
            return "UKS" if bool(solver.istype("UHF")) else "RESTRICTED"
        except Exception:
            return "UKS" if primary_kind == "UKS" else "RESTRICTED"

    def compatible_density(
        density: np.ndarray | None,
        solver: object,
    ) -> np.ndarray | None:
        """Convert between spin-resolved UKS and total RKS/ROKS densities."""
        checked = validate_density(density) if density is not None else None
        if checked is None:
            return None
        kind = solver_density_kind(solver)
        if kind == "UKS":
            if checked.ndim == 3:
                return checked if checked.shape[0] == 2 else None
            electron_total = alpha_electrons + beta_electrons
            return np.stack(
                (
                    (alpha_electrons / electron_total) * checked,
                    (beta_electrons / electron_total) * checked,
                )
            )
        if checked.ndim == 3:
            return checked.sum(axis=0) if checked.shape[0] == 2 else None
        return checked

    def prepare_preview_density(
        density: np.ndarray | None,
        solver: object,
    ) -> np.ndarray | None:
        """Validate a retained seed and restore its exact AO electron count."""
        prepared = compatible_density(density, solver)
        if prepared is None or prepared.shape[-2:] != (
            number_orbitals,
            number_orbitals,
        ):
            return None
        targets = (
            (alpha_electrons, beta_electrons)
            if solver_density_kind(solver) == "UKS"
            else (mol.nelectron,)
        )
        try:
            normalised = _normalise_density_electron_counts(
                prepared,
                np.asarray(solver.get_ovlp()),
                targets,
            )
            return validate_density(normalised)
        except Exception as exc:
            print(
                "Retained preview density was unusable "
                f"({type(exc).__name__}: {exc})."
            )
            return None

    def remember_seed_density(
        label: str,
        solver: object,
        density: np.ndarray | None,
    ) -> None:
        """Keep the first physical seed independently of later SCF failures."""
        nonlocal retained_seed_density, retained_seed_solver, retained_seed_label
        if retained_seed_density is not None:
            return
        prepared = prepare_preview_density(density, solver)
        if prepared is not None:
            retained_seed_density = prepared
            retained_seed_solver = solver
            retained_seed_label = label

    def evaluate_preview_density(
        solver: object,
        density: np.ndarray | None,
        allow_dft_energy: bool = True,
    ) -> tuple[np.ndarray | None, float, str]:
        """Evaluate one finite density directly, without occupation callbacks."""
        prepared = prepare_preview_density(density, solver)
        if prepared is None:
            return None, float("nan"), ""
        if allow_dft_energy:
            try:
                energy_value = float(np.real(solver.energy_tot(dm=prepared)))
                if np.isfinite(energy_value):
                    return prepared, energy_value, "single-density DFT evaluation"
            except Exception as exc:
                attempt_errors.append(
                    f"single-density energy: {type(exc).__name__}: {exc}"
                )
                print(
                    "Direct single-density DFT evaluation was unavailable "
                    f"({type(exc).__name__}: {exc})."
                )

        # The total energy is display metadata and is not used to construct the
        # radial potential.  A finite one-electron expectation therefore keeps
        # the heavy-element rendering path usable when numerical DFT quadrature is
        # also unavailable, while the page remains labelled as a preview.
        try:
            hcore = np.asarray(solver.get_hcore())
            if hcore.ndim == 2:
                total_density = prepared.sum(axis=0) if prepared.ndim == 3 else prepared
                energy_value = np.einsum(
                    "ij,ji->",
                    total_density,
                    hcore,
                    optimize=True,
                )
            elif hcore.ndim == 3 and hcore.shape[0] == 2 and prepared.ndim == 3:
                energy_value = sum(
                    np.einsum(
                        "ij,ji->",
                        prepared[channel],
                        hcore[channel],
                        optimize=True,
                    )
                    for channel in range(2)
                )
            else:
                raise ValueError("incompatible preview Hamiltonian and density")
            energy_value = float(np.real(energy_value)) + float(mol.energy_nuc())
            if not np.isfinite(energy_value):
                raise ValueError("non-finite one-electron preview energy")
            return prepared, energy_value, "one-electron preview expectation"
        except Exception as exc:
            attempt_errors.append(
                f"one-electron preview energy: {type(exc).__name__}: {exc}"
            )
            print(
                "One-electron preview energy was unavailable "
                f"({type(exc).__name__}: {exc})."
            )
            return prepared, float("nan"), ""

    def density_from_one_electron_hamiltonian(
        solver: object,
        hamiltonian: object,
    ) -> np.ndarray | None:
        """Occupy a generalized one-electron eigenproblem in the AO basis."""
        try:
            overlap = np.asarray(solver.get_ovlp())
            one_electron = np.asarray(hamiltonian)

            if one_electron.ndim == 2:
                dm_alpha = _canonical_occupied_density(
                    one_electron,
                    overlap,
                    alpha_electrons,
                )
                dm_beta = _canonical_occupied_density(
                    one_electron,
                    overlap,
                    beta_electrons,
                )
            elif one_electron.ndim == 3 and one_electron.shape[0] == 2:
                dm_alpha = _canonical_occupied_density(
                    one_electron[0],
                    overlap,
                    alpha_electrons,
                )
                dm_beta = _canonical_occupied_density(
                    one_electron[1],
                    overlap,
                    beta_electrons,
                )
            else:
                raise ValueError(
                    f"unexpected one-electron Hamiltonian shape {one_electron.shape}"
                )

            density = (
                np.stack((dm_alpha, dm_beta))
                if solver_density_kind(solver) == "UKS"
                else dm_alpha + dm_beta
            )
            return validate_density(np.real_if_close(density))
        except Exception as exc:
            print(
                "One-electron density construction was unavailable "
                f"({type(exc).__name__}: {exc})."
            )
            return None

    def core_hamiltonian_density(solver: object) -> np.ndarray | None:
        """Build a table-independent one-electron guess for heavy previews."""
        try:
            return density_from_one_electron_hamiltonian(
                solver,
                solver.get_hcore(),
            )
        except Exception as exc:
            print(
                "One-electron initial density was unavailable "
                f"({type(exc).__name__}: {exc})."
            )
            return None

    def extended_slater_density(solver: object) -> np.ndarray | None:
        """Build a screened AO density from shell-dependent Slater charges."""
        # Madelung order is adequate for a starting potential; the subsequent
        # SCF and ensemble occupations determine the final electronic state.
        aufbau_order = (
            (1, 0, 2),
            (2, 0, 2), (2, 1, 6),
            (3, 0, 2), (3, 1, 6),
            (4, 0, 2), (3, 2, 10), (4, 1, 6),
            (5, 0, 2), (4, 2, 10), (5, 1, 6),
            (6, 0, 2), (4, 3, 14), (5, 2, 10), (6, 1, 6),
            (7, 0, 2), (5, 3, 14), (6, 2, 10), (7, 1, 6),
        )
        try:
            remaining = int(mol.nelectron)
            occupied_shells: list[tuple[int, int, int, int]] = []
            for order_index, (principal, angular_momentum, capacity) in enumerate(
                aufbau_order
            ):
                occupation = min(remaining, capacity)
                if occupation > 0:
                    occupied_shells.append(
                        (principal, angular_momentum, occupation, order_index)
                    )
                    remaining -= occupation
                if remaining == 0:
                    break
            if remaining != 0:
                raise ValueError("Aufbau table does not cover the electron count")

            shell_models: list[tuple[int, int, float, float]] = []
            for principal, angular_momentum, occupation, order_index in occupied_shells:
                if angular_momentum <= 1:
                    same_group = sum(
                        count
                        for n_value, l_value, count, _index in occupied_shells
                        if n_value == principal and l_value <= 1
                    ) - 1
                    same_weight = 0.30 if principal == 1 else 0.35
                    screening = same_weight * max(same_group, 0)
                    screening += 0.85 * sum(
                        count
                        for n_value, _l_value, count, _index in occupied_shells
                        if n_value == principal - 1
                    )
                    screening += sum(
                        count
                        for n_value, _l_value, count, _index in occupied_shells
                        if n_value <= principal - 2
                    )
                else:
                    screening = 0.35 * max(occupation - 1, 0)
                    screening += sum(
                        count
                        for _n_value, _l_value, count, source_index in occupied_shells
                        if source_index < order_index
                    )
                effective_charge = max(atomic_number - screening, 0.5)
                shell_models.append(
                    (principal, angular_momentum, float(occupation), effective_charge)
                )

            guess_grid = dft.gen_grid.Grids(mol)
            guess_grid.level = 0
            guess_grid.build(with_non0tab=False)
            coordinates = np.asarray(guess_grid.coords)
            weights = np.asarray(guess_grid.weights)
            radii = np.linalg.norm(coordinates, axis=1)
            screening_potential = np.zeros_like(radii)
            self_interaction_scale = (
                (mol.nelectron - 1) / mol.nelectron if mol.nelectron > 1 else 0.0
            )
            for principal, _angular_momentum, occupation, effective_charge in shell_models:
                shell_radius = float(
                    np.clip(principal**2 / effective_charge, 0.06, RADIAL_MAX_BOHR)
                )
                regularized_coulomb = np.divide(
                    np.tanh(radii / shell_radius),
                    radii,
                    out=np.full_like(radii, 1.0 / shell_radius),
                    where=radii > 1.0e-12,
                )
                screening_potential += (
                    self_interaction_scale * occupation * regularized_coulomb
                )

            ao_values = np.asarray(dft.numint.eval_ao(mol, coordinates))
            weighted_potential = weights * screening_potential
            screening_matrix = np.einsum(
                "pi,p,pj->ij",
                ao_values.conj(),
                weighted_potential,
                ao_values,
                optimize=True,
            )
            screening_matrix = np.real_if_close(
                0.5 * (screening_matrix + screening_matrix.conj().T)
            )
            hcore = np.asarray(solver.get_hcore())
            screened_hamiltonian = hcore + screening_matrix
            return density_from_one_electron_hamiltonian(
                solver,
                screened_hamiltonian,
            )
        except Exception as exc:
            print(
                "Extended-Slater initial density was unavailable "
                f"({type(exc).__name__}: {exc})."
            )
            return None

    def initial_density_residual(
        solver: object,
        density: np.ndarray,
    ) -> float:
        """Return the normalized FDS-SDF residual for one candidate density."""
        prepared = compatible_density(density, solver)
        if prepared is None:
            return float("inf")
        overlap = np.asarray(solver.get_ovlp())
        fock = np.asarray(solver.get_fock(dm=prepared))

        def channel_residual(
            fock_channel: np.ndarray,
            density_channel: np.ndarray,
        ) -> float:
            left = fock_channel @ density_channel @ overlap
            right = overlap @ density_channel @ fock_channel
            denominator = max(
                float(np.linalg.norm(left) + np.linalg.norm(right)),
                1.0e-14,
            )
            return float(np.linalg.norm(left - right) / denominator)

        if prepared.ndim == 2 and fock.ndim == 2:
            residual = channel_residual(fock, prepared)
        elif prepared.ndim == 3 and fock.ndim == 3 and prepared.shape[0] == 2:
            residual = float(
                np.hypot(
                    channel_residual(fock[0], prepared[0]),
                    channel_residual(fock[1], prepared[1]),
                )
            )
        else:
            return float("inf")
        return residual if np.isfinite(residual) else float("inf")

    def refine_heavy_preview_density(
        solver: object,
        density: np.ndarray | None,
    ) -> tuple[np.ndarray | None, int]:
        """Apply bounded, occupation-safe Fock updates to a physical seed."""
        prepared = prepare_preview_density(density, solver)
        if prepared is None or HEAVY_PREVIEW_MAX_FOCK_UPDATES < 1:
            return prepared, 0

        original_grids = solver.grids
        started = time.perf_counter()
        completed = 0
        try:
            preview_grids = dft.gen_grid.Grids(mol)
            preview_grids.level = HEAVY_PREVIEW_GRID_LEVEL
            solver.grids = preview_grids
            for update_index in range(HEAVY_PREVIEW_MAX_FOCK_UPDATES):
                if completed > 0:
                    elapsed = time.perf_counter() - started
                    mean_update_time = elapsed / completed
                    if (
                        elapsed >= HEAVY_PREVIEW_TIME_BUDGET_SECONDS
                        or elapsed + mean_update_time
                        > HEAVY_PREVIEW_TIME_BUDGET_SECONDS
                    ):
                        break
                fock = np.asarray(solver.get_fock(dm=prepared))
                candidate = density_from_one_electron_hamiltonian(solver, fock)
                candidate = prepare_preview_density(candidate, solver)
                if candidate is None:
                    break
                mixed = (
                    HEAVY_PREVIEW_NEW_DENSITY_WEIGHT * candidate
                    + (1.0 - HEAVY_PREVIEW_NEW_DENSITY_WEIGHT) * prepared
                )
                updated = prepare_preview_density(mixed, solver)
                if updated is None:
                    break
                prepared = updated
                completed = update_index + 1
                print(
                    "Heavy-preview fixed-occupation Fock update "
                    f"{completed}/{HEAVY_PREVIEW_MAX_FOCK_UPDATES} completed."
                )
        except Exception as exc:
            print(
                "Heavy-preview Fock refinement stopped safely "
                f"({type(exc).__name__}: {exc})."
            )
        finally:
            solver.grids = original_grids
        return prepared, completed

    def select_heavy_initial_density(
        solver: object,
    ) -> tuple[np.ndarray | None, str]:
        """Choose SAP or extended Slater by a cheap coarse-grid SCF residual."""
        candidates: list[tuple[str, np.ndarray]] = []
        core_density: np.ndarray | None = None
        try:
            print("Preparing heavy-atom SAP initial density.")
            sap_density = compatible_density(
                validate_density(solver.get_init_guess(key="sap")),
                solver,
            )
            if sap_density is not None:
                candidates.append(("SAP", sap_density))
        except Exception as exc:
            print(
                "SAP initial density was unavailable "
                f"({type(exc).__name__}: {exc})."
            )

        # For deterministic heavy previews this density is only a clearly
        # labelled fallback, so avoid the expensive residual comparison and
        # all occupation logic.
        # SAP remains more physical than an unscreened core-Hamiltonian seed.
        if candidates and (atomic_number <= 96 or deterministic_preview_path):
            return candidates[0][1], candidates[0][0]

        if deterministic_preview_path:
            core_density = core_hamiltonian_density(solver)
            if core_density is not None:
                return core_density, "canonical core Hamiltonian"
            targets = (
                (alpha_electrons, beta_electrons)
                if solver_density_kind(solver) == "UKS"
                else (mol.nelectron,)
            )
            try:
                overlap_density = _overlap_metric_preview_density(
                    np.asarray(solver.get_ovlp()),
                    targets,
                )
                checked_density = validate_density(overlap_density)
                if checked_density is not None:
                    return checked_density, "overlap-metric emergency fallback"
            except Exception as exc:
                print(
                    "Overlap-metric emergency density was unavailable "
                    f"({type(exc).__name__}: {exc}); trying screened Slater."
                )

        print("Preparing extended-Slater screened initial density.")
        slater_density = extended_slater_density(solver)
        if slater_density is not None:
            candidates.append(("extended Slater", slater_density))

        if len(candidates) == 1:
            return candidates[0][1], candidates[0][0]
        if len(candidates) >= 2:
            original_grids = solver.grids
            try:
                scoring_grids = dft.gen_grid.Grids(mol)
                scoring_grids.level = 0
                solver.grids = scoring_grids
                scored: list[tuple[float, str, np.ndarray]] = []
                for label, candidate in candidates:
                    residual = initial_density_residual(solver, candidate)
                    print(f"Initial-density residual ({label}): {residual:.3e}")
                    scored.append((residual, label, candidate))
                finite_scored = [item for item in scored if np.isfinite(item[0])]
                if finite_scored:
                    residual, label, candidate = min(
                        finite_scored,
                        key=lambda item: item[0],
                    )
                    print(
                        f"Selected {label} initial density "
                        f"(coarse residual {residual:.3e})."
                    )
                    return candidate, label
            except Exception as exc:
                print(
                    "Initial-density residual comparison was unavailable "
                    f"({type(exc).__name__}: {exc}); preferring SAP."
                )
            finally:
                solver.grids = original_grids
            return candidates[0][1], candidates[0][0]

        if core_density is None:
            core_density = core_hamiltonian_density(solver)
        return core_density, "core Hamiltonian"

    def install_safe_degenerate_occupations(
        solver: object,
        tolerance: float = 2.0e-3,
    ) -> None:
        """Install finite, electron-conserving occupations without empty-list indexing."""
        original_get_occ = solver.get_occ

        def fractional_channel(
            orbital_energies: object,
            electron_count: int,
            maximum_occupation: float,
        ) -> np.ndarray | None:
            energies = np.asarray(orbital_energies, dtype=float)
            if energies.ndim != 1 or not np.all(np.isfinite(energies)):
                return None
            occupied_orbitals = int(np.ceil(electron_count / maximum_occupation))
            if occupied_orbitals <= 0 or occupied_orbitals >= energies.size:
                return None
            ordered = np.sort(energies)
            homo = ordered[occupied_orbitals - 1]
            lumo = ordered[occupied_orbitals]
            if abs(lumo - homo) >= tolerance:
                return None
            shell = np.abs(energies - homo) < tolerance
            below_shell = energies < homo - tolerance
            remaining = electron_count - maximum_occupation * int(below_shell.sum())
            shell_size = int(shell.sum())
            if shell_size == 0 or not 0.0 <= remaining <= maximum_occupation * shell_size:
                return None
            occupations = np.zeros_like(energies)
            occupations[below_shell] = maximum_occupation
            occupations[shell] = remaining / shell_size
            return occupations

        def safe_get_occ(mo_energy: object, mo_coeff: object = None) -> np.ndarray:
            energies = np.asarray(mo_energy, dtype=float)

            def integer_channel(
                channel_energies: np.ndarray,
                electron_count: int,
                maximum_occupation: float,
            ) -> np.ndarray:
                occupations = np.zeros_like(channel_energies)
                remaining = float(electron_count)
                for index in np.argsort(channel_energies, kind="stable"):
                    if remaining <= 0.0:
                        break
                    amount = min(maximum_occupation, remaining)
                    occupations[index] = amount
                    remaining -= amount
                if remaining > 1.0e-8:
                    raise ValueError("finite orbital set cannot hold the electrons")
                return occupations

            try:
                baseline = np.asarray(
                    original_get_occ(mo_energy, mo_coeff),
                    dtype=float,
                ).copy()
            except Exception as exc:
                print(
                    "Default occupation assignment failed; using safe Aufbau "
                    f"occupations ({type(exc).__name__}: {exc})."
                )
                if solver_density_kind(solver) == "UKS":
                    if energies.ndim != 2 or energies.shape[0] != 2:
                        raise ValueError("invalid UKS orbital-energy array") from exc
                    baseline = np.stack(
                        tuple(
                            integer_channel(energies[channel], count, 1.0)
                            for channel, count in enumerate(
                                (alpha_electrons, beta_electrons)
                            )
                        )
                    )
                else:
                    if energies.ndim != 1:
                        raise ValueError("invalid restricted orbital-energy array") from exc
                    baseline = integer_channel(energies, mol.nelectron, 2.0)

            if solver_density_kind(solver) == "UKS":
                if energies.ndim != 2 or energies.shape[0] != 2 or baseline.shape != energies.shape:
                    return baseline
                for channel, electron_count in enumerate(
                    (alpha_electrons, beta_electrons)
                ):
                    fractional = fractional_channel(
                        energies[channel], electron_count, 1.0
                    )
                    if fractional is not None:
                        baseline[channel] = fractional
                return baseline

            fractional = fractional_channel(energies, mol.nelectron, 2.0)
            return baseline if fractional is None else fractional

        solver.get_occ = safe_get_occ

    def install_fixed_average_configuration_occupations(
        solver: object,
        tolerance: float = HEAVY_ENSEMBLE_ENERGY_TOLERANCE,
    ) -> None:
        """Freeze a finite frontier-shell ensemble without PySCF occupation indexing.

        On the first call for each spin channel, orbitals within ``tolerance``
        of the frontier are grouped into one shell. Electrons left after the
        lower orbitals are filled are shared equally across that shell. Only
        the *counts* are frozen: on later iterations the pattern follows the
        energy ordering, so harmless orbital reordering cannot change the total
        alpha/beta populations or trigger empty-list indexing.
        """
        patterns: dict[int, tuple[int, int, float]] = {}

        def channel_occupations(
            orbital_energies: object,
            electron_count: int,
            maximum_occupation: float,
            channel: int,
        ) -> np.ndarray:
            energies = np.asarray(orbital_energies, dtype=float)
            if energies.ndim != 1 or energies.size == 0:
                raise ValueError("invalid ensemble orbital-energy array")
            if not np.all(np.isfinite(energies)):
                raise ValueError("ensemble orbital energies are not finite")
            if not 0 <= electron_count <= maximum_occupation * energies.size:
                raise ValueError("ensemble electron count exceeds orbital capacity")

            order = np.argsort(energies, kind="stable")
            ordered = energies[order]
            pattern = patterns.get(channel)
            if pattern is None:
                # Learn one average-of-configuration pattern from the first
                # finite spectrum, then reuse it throughout this SCF attempt.
                occupied_orbitals = int(
                    np.ceil(float(electron_count) / maximum_occupation)
                )
                if occupied_orbitals == 0:
                    pattern = (0, 0, 0.0)
                else:
                    frontier = min(occupied_orbitals - 1, energies.size - 1)
                    frontier_energy = ordered[frontier]
                    lower = frontier
                    while (
                        lower > 0
                        and abs(ordered[lower - 1] - frontier_energy) < tolerance
                    ):
                        lower -= 1
                    upper = frontier
                    while (
                        upper + 1 < energies.size
                        and abs(ordered[upper + 1] - frontier_energy) < tolerance
                    ):
                        upper += 1
                    shell_size = upper - lower + 1
                    remaining = (
                        float(electron_count) - maximum_occupation * lower
                    )
                    if not 0.0 <= remaining <= maximum_occupation * shell_size:
                        raise ValueError("invalid frontier-shell ensemble capacity")
                    pattern = (lower, shell_size, remaining / shell_size)
                patterns[channel] = pattern

            below_count, shell_size, shell_occupation = pattern
            if below_count + shell_size > energies.size:
                raise ValueError("stored ensemble pattern exceeds orbital count")
            occupations = np.zeros_like(energies)
            occupations[order[:below_count]] = maximum_occupation
            if shell_size:
                occupations[
                    order[below_count : below_count + shell_size]
                ] = shell_occupation
            if not np.isclose(
                float(occupations.sum()),
                float(electron_count),
                rtol=0.0,
                atol=1.0e-8,
            ):
                raise ValueError("ensemble occupations do not conserve electrons")
            return occupations

        def fixed_get_occ(
            mo_energy: object,
            mo_coeff: object = None,
        ) -> np.ndarray:
            del mo_coeff
            energies = np.asarray(mo_energy, dtype=float)
            if solver_density_kind(solver) == "UKS":
                if energies.ndim != 2 or energies.shape[0] != 2:
                    raise ValueError("invalid UKS ensemble orbital energies")
                return np.stack(
                    tuple(
                        channel_occupations(
                            energies[channel],
                            electron_count,
                            1.0,
                            channel,
                        )
                        for channel, electron_count in enumerate(
                            (alpha_electrons, beta_electrons)
                        )
                    )
                )
            if energies.ndim != 1:
                raise ValueError("invalid restricted ensemble orbital energies")
            return channel_occupations(
                energies,
                mol.nelectron,
                2.0,
                0,
            )

        solver.get_occ = fixed_get_occ

    def run_attempt(
        label: str,
        solver: object,
        initial_density: np.ndarray | None = None,
    ) -> tuple[float, bool]:
        nonlocal last_valid_density, last_successful_solver, last_finite_energy
        print(f"SCF attempt: {label}")
        prepared_density = compatible_density(initial_density, solver)
        if prepared_density is None and heavy_atom:
            prepared_density = core_hamiltonian_density(solver)
        remember_seed_density(label, solver, prepared_density)
        try:
            energy_value = float(solver.kernel(dm0=prepared_density))
        except Exception as exc:
            detail = f"{label}: {type(exc).__name__}: {exc}"
            attempt_errors.append(detail)
            print(f"SCF strategy failed safely ({detail}); trying the next strategy.")
            return float("nan"), False

        updated_density = density_from_solver(solver, last_valid_density)
        if updated_density is not None:
            last_valid_density = updated_density
            last_successful_solver = solver
        if np.isfinite(energy_value):
            last_finite_energy = energy_value
        if bool(getattr(solver, "converged", False)):
            print(f"SCF converged with {label}.")
            return energy_value, True
        print(f"SCF did not converge with {label}; preparing next strategy.")
        return energy_value, False

    density_cache_arrays = ("dm_alpha", "dm_beta")

    def cached_spin_density(
        cached: dict[str, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Validate cached alpha/beta matrices against the current AO basis."""
        if cached is None:
            return None
        dm_alpha_cached = validate_density(cached.get("dm_alpha"))
        dm_beta_cached = validate_density(cached.get("dm_beta"))
        expected_shape = (number_orbitals, number_orbitals)
        if (
            dm_alpha_cached is None
            or dm_beta_cached is None
            or dm_alpha_cached.ndim != 2
            or dm_beta_cached.ndim != 2
            or dm_alpha_cached.shape != expected_shape
            or dm_beta_cached.shape != expected_shape
        ):
            return None
        return dm_alpha_cached, dm_beta_cached

    exact_loaded = load_array_bundle(exact_density_path, density_cache_arrays)
    exact_cached = None if exact_loaded is None else exact_loaded[0]
    exact_metadata = {} if exact_loaded is None else exact_loaded[1]
    exact_spin_density = cached_spin_density(exact_cached)
    if exact_cached is not None and exact_spin_density is not None:
        try:
            cached_energy = float(exact_metadata["dft_energy"])
            cached_info = exact_metadata["calculation_info"]
            if not np.isfinite(cached_energy) or not isinstance(cached_info, dict):
                raise ValueError("cached SCF metadata is not finite")
            cached_info = dict(cached_info)
            cached_info["quality_profile"] = QUALITY_PROFILE.lower()
            cached_info["quality_level"] = QUALITY_LEVEL
            cached_info["density_cache"] = "reused"
            print("Persistent cache hit: reusable atomic SCF density.")
            return (
                mol,
                None,
                exact_spin_density[0],
                exact_spin_density[1],
                cached_energy,
                cached_info,
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"SCF density cache metadata was invalid; recalculating: {exc}")

    warm_loaded = load_array_bundle(warm_start_path, density_cache_arrays)
    warm_cached = None if warm_loaded is None else warm_loaded[0]
    warm_spin_density = cached_spin_density(warm_cached)

    primary_kind = "RKS" if SPIN == 0 else "UKS"
    occupation_model = "integer"
    initial_guess_model = "default"
    mean_field = make_solver(primary_kind, "standard")
    initial_density = (
        None
        if warm_spin_density is None
        else np.stack(warm_spin_density)
    )
    if initial_density is not None:
        initial_density = compatible_density(initial_density, mean_field)
        last_valid_density = initial_density
        initial_guess_model = "cached warm start"
        print("Persistent cache hit: using a prior SCF density as the warm start.")
    elif heavy_atom:
        initial_density, initial_guess_model = select_heavy_initial_density(
            mean_field
        )
        last_valid_density = initial_density

    # Refine the physical seed before either displaying it directly or using it
    # to launch high-accuracy ensemble SCF. This work is bounded by both an
    # update count and a time budget from the selected quality profile.
    if preview_fallback_available:
        refined_density, refinement_steps = refine_heavy_preview_density(
            mean_field,
            initial_density,
        )
        if refined_density is not None:
            initial_density = refined_density
            last_valid_density = refined_density
        if refinement_steps > 0:
            initial_guess_model = (
                f"{initial_guess_model} + {refinement_steps} "
                "fixed-occupation Fock update"
                f"{'s' if refinement_steps != 1 else ''}"
            )

    # Select exactly one primary path. Subsequent blocks are retry/fallback
    # stages and execute only if this primary path did not converge.
    if deterministic_preview_path:
        print(
            f"Using deterministic {preview_policy} preview path for {symbol}; "
            "skipping occupation-driven SCF kernels."
        )
        remember_seed_density(initial_guess_model, mean_field, initial_density)
        energy = float("nan")
        converged = False
    elif high_accuracy_ensemble_scf:
        mean_field = make_solver(primary_kind, "ensemble")
        install_fixed_average_configuration_occupations(mean_field)
        mean_field.damp = 0.20
        mean_field.level_shift = 0.15
        mean_field.diis_start_cycle = 3
        mean_field.diis_space = 14
        energy, converged = run_attempt(
            f"fixed average-of-configuration ensemble {primary_kind}",
            mean_field,
            initial_density,
        )
        if converged:
            occupation_model = "fixed zero-temperature average-of-configuration ensemble"
    else:
        energy, converged = run_attempt(
            f"standard {primary_kind}/DIIS",
            mean_field,
            initial_density,
        )

    # First retry: preserve the selected occupation model but add damping and a
    # level shift to suppress oscillation between nearly degenerate solutions.
    if not converged and not deterministic_preview_path:
        last_density = density_from_solver(mean_field, last_valid_density)
        mean_field = make_solver(primary_kind, "stabilised")
        mean_field.damp = SCF_DAMPING
        mean_field.level_shift = SCF_LEVEL_SHIFT
        mean_field.diis_start_cycle = 4
        mean_field.diis_space = 14
        if high_accuracy_ensemble_scf:
            install_fixed_average_configuration_occupations(mean_field)
            attempt_label = (
                "damped, level-shifted fixed-ensemble " f"{primary_kind}"
            )
        else:
            attempt_label = f"damped, level-shifted {primary_kind}"
        energy, converged = run_attempt(
            attempt_label,
            mean_field,
            last_density,
        )
        if converged and high_accuracy_ensemble_scf:
            occupation_model = "fixed zero-temperature average-of-configuration ensemble"

    # Second retry: a small electronic temperature produces a smoother warm
    # density. Temperature is then removed; the reported state is always a
    # zero-temperature integer or fixed-ensemble solution.
    if not converged and heavy_atom and not deterministic_preview_path:
        last_density = density_from_solver(mean_field, last_valid_density)
        smeared_solver = make_solver(primary_kind, "smearing")
        smeared_solver.damp = 0.20
        smeared_solver.level_shift = 0.20
        try:
            scf.addons.smearing_(
                smeared_solver,
                sigma=SCF_SMEARING_SIGMA,
                method="fermi",
                fix_spin=SPIN != 0,
            )
            _thermal_energy, _thermal_converged = run_attempt(
                f"finite-temperature {primary_kind} preconditioner",
                smeared_solver,
                last_density,
            )
            warm_density = density_from_solver(
                smeared_solver,
                last_valid_density,
            )
        except Exception as exc:
            attempt_errors.append(
                f"smearing setup: {type(exc).__name__}: {exc}"
            )
            print(
                "Smearing preconditioner was unavailable; continuing without it "
                f"({type(exc).__name__}: {exc})."
            )
            warm_density = last_valid_density

        mean_field = make_solver(primary_kind, "warm")
        mean_field.damp = 0.15
        mean_field.level_shift = 0.10
        if high_accuracy_ensemble_scf:
            install_fixed_average_configuration_occupations(mean_field)
            warm_label = f"zero-temperature fixed-ensemble {primary_kind}"
        else:
            warm_label = f"zero-temperature integer-occupation {primary_kind}"
        energy, converged = run_attempt(
            warm_label,
            mean_field,
            warm_density,
        )
        if converged and high_accuracy_ensemble_scf:
            occupation_model = "fixed zero-temperature average-of-configuration ensemble"

    # Third retry: Newton SCF can converge a stationary state after DIIS/damping
    # has brought the density sufficiently close to self-consistency.
    if not converged and not deterministic_preview_path:
        last_density = density_from_solver(mean_field, last_valid_density)
        try:
            mean_field = mean_field.newton()
            mean_field.conv_tol = scf_tolerance
            mean_field.max_cycle = newton_cycle_limit
            mean_field.max_cycle_inner = 10
            mean_field.max_stepsize = 0.05
            energy, converged = run_attempt(
                f"second-order Newton {primary_kind}",
                mean_field,
                last_density,
            )
            if converged and high_accuracy_ensemble_scf:
                occupation_model = (
                    "fixed zero-temperature average-of-configuration ensemble"
                )
        except Exception as exc:
            attempt_errors.append(
                f"Newton {primary_kind} setup: {type(exc).__name__}: {exc}"
            )
            print(
                "Newton setup failed safely; continuing "
                f"({type(exc).__name__}: {exc})."
            )

    # ROKS is a spin-pure fallback for open shells whose unrestricted
    # occupations keep changing.  It preserves the requested N_alpha/N_beta.
    if (
        not converged
        and not deterministic_preview_path
        and SPIN != 0
        and ALLOW_ROKS_FALLBACK
    ):
        mean_field = make_solver("ROKS", "roks")
        mean_field.damp = SCF_DAMPING
        mean_field.level_shift = SCF_LEVEL_SHIFT
        mean_field.diis_start_cycle = 4
        energy, converged = run_attempt(
            "damped, level-shifted ROKS",
            mean_field,
            initial_density,
        )

        if not converged:
            last_density = density_from_solver(mean_field, last_valid_density)
            try:
                mean_field = mean_field.newton()
                mean_field.conv_tol = scf_tolerance
                mean_field.max_cycle = newton_cycle_limit
                mean_field.max_cycle_inner = 10
                mean_field.max_stepsize = 0.05
                energy, converged = run_attempt(
                    "second-order Newton ROKS",
                    mean_field,
                    last_density,
                )
            except Exception as exc:
                attempt_errors.append(
                    f"Newton ROKS setup: {type(exc).__name__}: {exc}"
                )
                print(
                    "ROKS Newton setup failed safely; continuing "
                    f"({type(exc).__name__}: {exc})."
                )

    # An isolated atom with a partially filled, exactly degenerate shell is
    # correctly represented by a zero-temperature ensemble.  This final heavy-
    # atom fallback fractionally occupies only numerically degenerate HOMOs,
    # while preserving the selected alpha and beta electron counts.
    if not converged and heavy_atom and not deterministic_preview_path:
        last_density = density_from_solver(mean_field, last_valid_density)
        try:
            number_orbitals = int(mol.nao_nr())
            alpha_electrons, beta_electrons = map(int, mol.nelec)
            if primary_kind == "RKS":
                occupation_counts = (mol.nelectron // 2,)
            else:
                occupation_counts = (alpha_electrons, beta_electrons)
            fractional_occupations_safe = all(
                0 < count < number_orbitals for count in occupation_counts
            )
        except Exception as exc:
            fractional_occupations_safe = False
            attempt_errors.append(
                f"occupation bounds: {type(exc).__name__}: {exc}"
            )

        if fractional_occupations_safe:
            try:
                mean_field = make_solver(primary_kind, "ensemble")
                install_safe_degenerate_occupations(mean_field, tolerance=2.0e-3)
                mean_field.damp = 0.20
                mean_field.level_shift = 0.15
                energy, converged = run_attempt(
                    f"degenerate-shell ensemble {primary_kind}",
                    mean_field,
                    last_density,
                )
            except Exception as exc:
                attempt_errors.append(
                    f"ensemble setup: {type(exc).__name__}: {exc}"
                )
                print(
                    "Degenerate-shell ensemble setup was skipped safely "
                    f"({type(exc).__name__}: {exc})."
                )
        else:
            print(
                "Degenerate-shell occupations were skipped because an occupied "
                "channel reaches a finite-basis boundary."
            )

        if not converged and fractional_occupations_safe:
            last_density = density_from_solver(mean_field, last_valid_density)
            try:
                mean_field = mean_field.newton()
                mean_field.conv_tol = scf_tolerance
                mean_field.max_cycle = newton_cycle_limit
                mean_field.max_cycle_inner = 12
                mean_field.max_stepsize = 0.035
                energy, converged = run_attempt(
                    f"second-order degenerate-shell ensemble {primary_kind}",
                    mean_field,
                    last_density,
                )
            except Exception as exc:
                attempt_errors.append(
                    f"ensemble Newton setup: {type(exc).__name__}: {exc}"
                )
                print(
                    "Ensemble Newton setup failed safely; continuing "
                    f"({type(exc).__name__}: {exc})."
                )
        if converged:
            occupation_model = "zero-temperature degenerate-shell ensemble"

    # From here onward the code guarantees a finite renderable density. A true
    # SCF solution is preferred; difficult heavy elements may instead use the
    # best finite density retained during the attempts, followed by the refined
    # deterministic seed and finally an explicitly labelled emergency density.
    calculation_converged = converged
    energy_model = "converged DFT total energy"
    if not converged:
        charge_text = "" if ionic_charge == 0 else f"{abs(ionic_charge)}" + (
            "+" if ionic_charge > 0 else "-"
        )
        attempted = (
            f"the deterministic {preview_policy} density construction"
            if deterministic_preview_path
            else "DIIS, stabilised SCF, and Newton SCF"
        )
        if heavy_atom and not deterministic_preview_path:
            attempted += (
                ", with SAP, smearing warm start, and both DIIS and Newton "
                "ensemble occupations"
            )
        if (
            not deterministic_preview_path
            and SPIN != 0
            and ALLOW_ROKS_FALLBACK
        ):
            attempted += ", including ROKS"
        preview_density: np.ndarray | None = None
        preview_solver: object | None = None
        preview_energy = float("nan")
        preview_source = ""
        preview_energy_model = ""
        if preview_fallback_available:
            preview_solver = retained_seed_solver or mean_field
            preview_source = (
                retained_seed_label
                or initial_guess_model
                or "screened heavy-element seed"
            )
            preview_density = prepare_preview_density(
                (
                    retained_seed_density
                    if retained_seed_density is not None
                    else initial_density
                ),
                preview_solver,
            )
            if preview_density is None:
                targets = (
                    (alpha_electrons, beta_electrons)
                    if solver_density_kind(preview_solver) == "UKS"
                    else (mol.nelectron,)
                )
                try:
                    preview_density = validate_density(
                        _overlap_metric_preview_density(
                            np.asarray(preview_solver.get_ovlp()),
                            targets,
                        )
                    )
                    if preview_density is not None:
                        preview_source = "overlap-metric emergency fallback"
                except Exception as exc:
                    attempt_errors.append(
                        f"overlap-metric preview: {type(exc).__name__}: {exc}"
                    )
            if preview_density is not None:
                preview_energy = _thomas_fermi_preview_energy(
                    atomic_number,
                    mol.nelectron,
                )
                preview_energy_model = "Thomas-Fermi preview estimate"
        if heavy_atom and last_valid_density is not None and np.isfinite(
            last_finite_energy
        ):
            candidate_solver = last_successful_solver or retained_seed_solver
            if candidate_solver is not None:
                preview_density = prepare_preview_density(
                    last_valid_density,
                    candidate_solver,
                )
                if preview_density is not None:
                    preview_solver = candidate_solver
                    preview_energy = last_finite_energy
                    preview_source = "last finite stabilised"
                    preview_energy_model = "last finite SCF kernel energy"

        # Any heavy atom can raise inside get_occ before a kernel energy exists,
        # irrespective of charge, spin, solver type, or whether X2C decoration
        # succeeded.  Its SAP/Slater/core seed is nevertheless a finite,
        # physically populated AO density.  Evaluate that density directly:
        # energy_tot(dm=...) does not invoke the broken occupation callback.
        if heavy_atom and (
            preview_density is None or not np.isfinite(preview_energy)
        ):
            try:
                direct_solver = make_solver(primary_kind, "standard")
            except Exception as exc:
                direct_solver = retained_seed_solver
                attempt_errors.append(
                    f"preview solver: {type(exc).__name__}: {exc}"
                )
            candidates = (
                ("last finite stabilised", last_valid_density),
                (
                    retained_seed_label or initial_guess_model or "physical seed",
                    retained_seed_density,
                ),
            )
            if direct_solver is not None:
                for label, candidate in candidates:
                    (
                        candidate_density,
                        candidate_energy,
                        candidate_energy_model,
                    ) = evaluate_preview_density(
                        direct_solver,
                        candidate,
                        allow_dft_energy=not deterministic_preview_path,
                    )
                    if candidate_density is not None and np.isfinite(candidate_energy):
                        preview_density = candidate_density
                        preview_solver = direct_solver
                        preview_energy = candidate_energy
                        preview_source = label
                        preview_energy_model = candidate_energy_model
                        break

        if preview_density is not None and preview_solver is not None and np.isfinite(
            preview_energy
        ):
            if deterministic_preview_path:
                print(
                    "SCF was intentionally skipped at this quality level; using "
                    f"the finite {preview_source} density for a fast preview."
                )
                scf_fallback_status = "SCF not attempted"
            else:
                print(
                    "All strict convergence routes were exhausted. Continuing with "
                    f"the finite {preview_source} density for a safe fallback."
                )
                scf_fallback_status = "SCF not converged"
            mean_field = preview_solver
            energy = preview_energy
            energy_model = preview_energy_model
            occupation_model = (
                f"{preview_source} approximate density ({scf_fallback_status})"
            )
            density_matrix = preview_density
        else:
            if preview_fallback_available:
                # Absolute last-resort rendering density.  This deliberately
                # avoids every PySCF guess, energy, and occupation routine.
                identity_density = np.eye(number_orbitals, dtype=np.float64)
                try:
                    overlap_trace = float(
                        np.real(np.trace(identity_density @ mean_field.get_ovlp()))
                    )
                except Exception:
                    overlap_trace = float(number_orbitals)
                if not np.isfinite(overlap_trace) or overlap_trace <= 1.0e-12:
                    overlap_trace = float(number_orbitals)
                unit_density = identity_density / overlap_trace
                density_matrix = (
                    np.stack(
                        (
                            alpha_electrons * unit_density,
                            beta_electrons * unit_density,
                        )
                    )
                    if primary_kind == "UKS"
                    else mol.nelectron * unit_density
                )
                energy = _thomas_fermi_preview_energy(
                    atomic_number,
                    mol.nelectron,
                )
                energy_model = "Thomas-Fermi emergency preview estimate"
                emergency_status = (
                    "SCF not attempted"
                    if deterministic_preview_path
                    else "SCF not converged"
                )
                occupation_model = (
                    "identity-metric heavy-element preview "
                    f"({emergency_status})"
                )
                print(
                    "Using the emergency identity-metric heavy-element preview; "
                    "rendering will continue with an explicitly labelled fallback."
                )
            else:
                detail = "; ".join(attempt_errors[-3:])
                suffix = f" Last safe failures: {detail}." if detail else ""
                raise RuntimeError(
                    f"Atomic DFT could not produce a finite density for "
                    f"{symbol}{charge_text} ({mol.nelectron} electrons, 2S={SPIN}) "
                    f"after {attempted}.{suffix}"
                )
    else:
        density_matrix = density_from_solver(mean_field, last_valid_density)
        if density_matrix is None:
            raise RuntimeError(
                "The SCF solver converged but did not return a finite density matrix."
            )

    if density_matrix.ndim == 2 and SPIN != 0 and hasattr(mean_field, "make_rdm1s"):
        try:
            spin_density = validate_density(mean_field.make_rdm1s())
            if spin_density is not None and spin_density.ndim == 3:
                density_matrix = spin_density
        except Exception as exc:
            print(
                "Spin-resolved ROKS density was unavailable; using the total "
                f"density split ({type(exc).__name__}: {exc})."
            )

    if density_matrix.ndim == 2:
        alpha_fraction = alpha_electrons / mol.nelectron
        beta_fraction = beta_electrons / mol.nelectron
        dm_alpha = alpha_fraction * density_matrix
        dm_beta = beta_fraction * density_matrix
    elif density_matrix.ndim == 3 and density_matrix.shape[0] == 2:
        dm_alpha, dm_beta = density_matrix[0], density_matrix[1]
    else:
        raise ValueError(f"Unexpected DFT density-matrix shape {density_matrix.shape}.")

    calculation_info = {
        "basis": basis_name,
        "engine_revision": ENGINE_REVISION,
        "relativistic": relativistic,
        "superheavy_fast_path": bool(
            superheavy_fast_path and deterministic_preview_path
        ),
        "f_block_preview_path": bool(
            f_block_preview_path and deterministic_preview_path
        ),
        "high_accuracy_ensemble_scf": bool(high_accuracy_ensemble_scf),
        "deterministic_preview_policy": preview_policy,
        "heavy_atom_policy": (
            "fixed average-of-configuration ensemble SCF with deterministic fallback"
            if high_accuracy_ensemble_scf
            else (
                "deterministic finite-density preview; SCF not attempted"
                if deterministic_preview_path
                else (
                    "staged heavy-atom SCF with finite-density fallback"
                    if heavy_atom
                    else "standard SCF only"
                )
            )
        ),
        "hamiltonian": (
            "spin-free one-electron X2C"
            if x2c_active
            else (
                "all-electron non-X2C fallback"
                if relativistic
                else "nonrelativistic"
            )
        ),
        "radial_solver": "scalar ZORA" if relativistic else "Schrödinger",
        "energy_model": energy_model,
        "occupations": occupation_model,
        "convergence": (
            "converged"
            if calculation_converged
            else (
                "not attempted — deterministic preview"
                if deterministic_preview_path
                else "preview fallback"
            )
        ),
        "quality_profile": QUALITY_PROFILE.lower(),
        "quality_level": QUALITY_LEVEL,
        "scf_grid_level": scf_grid_level,
        "scf_tolerance": scf_tolerance,
        "scf_cycle_policy": "staged" if heavy_atom else "profile default",
        "initial_guess": initial_guess_model,
        "density_cache": "calculated",
    }
    density_payload = {
        "dm_alpha": np.asarray(dm_alpha, dtype=np.float64),
        "dm_beta": np.asarray(dm_beta, dtype=np.float64),
    }
    density_metadata = {
        "dft_energy": float(energy),
        "calculation_info": calculation_info,
        "converged": bool(calculation_converged),
    }
    atomic_save_array_bundle(
        exact_density_path,
        density_payload,
        density_metadata,
    )
    warm_cache_exists = warm_start_path is not None and warm_start_path.is_dir()
    if calculation_converged or not warm_cache_exists:
        atomic_save_array_bundle(
            warm_start_path,
            density_payload,
            density_metadata,
        )
    return mol, mean_field, dm_alpha, dm_beta, energy, calculation_info


