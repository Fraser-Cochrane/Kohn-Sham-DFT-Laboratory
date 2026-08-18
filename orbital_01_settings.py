# =============================================================================
# PART 1 OF 9: IMPORTS, USER SETTINGS, QUALITY PROFILES, AND ELEMENT DATA
# =============================================================================
#
# Loaded first by atomic_orbital_master.py.  It imports every third-party
# dependency and defines the mutable settings shared by the calculation and
# Flask request handlers.  Do not run this file directly: the master script
# executes all nine parts in order inside one shared Python namespace.
#
"""DFT-derived atomic orbital and spatially dependent effective charge.

Edit USER SETTINGS and run:

    python atomic_orbital_master.py

Install:

    python -m pip install numpy scipy plotly scikit-image pyscf flask basis-set-exchange

The program performs an atomic LDA Kohn-Sham calculation, spherical-averages
the DFT spin density, and constructs

    v_KS,sigma(r) = -Z/r + v_H(r) + v_xc,sigma(r)
    Z_eff,sigma(r) = -r v_KS,sigma(r).

It numerically solves the radial Kohn-Sham equation for the requested (n,l),
combines the radial function with Y_l^m, and plots the 90% probability surface.
Elements from Rb onward use an all-electron spin-free X2C Hamiltonian for the
DFT density and a scalar-ZORA radial kinetic operator. The graph can colour the
surface by orbital phase or Z_eff(r). No Slater or Aufbau screening estimate is
used.

Atomic DFT fields, radial families, angular data, Cartesian grids, surface
meshes, and derived plot data are cached separately according to their physical
dependencies. Hot numerical arrays use uncompressed, memory-mapped NPY files;
exact rendered selections are reused across server restarts.

FINAL INTERACTIVE VERSION: LOGIC MAP

1. The Flask form validates the atom, charge, spin, quantum numbers and quality.
2. A quality profile applies one consistent set of SCF and rendering parameters.
3. Physical cache keys are built from only the inputs that can change each layer.
4. Ordinary atoms run staged Kohn-Sham SCF. F-block and superheavy atoms use a
   deterministic finite-density preview at levels 1-3; levels 4-5 first attempt
   fixed average-of-configuration ensemble SCF and retain the preview as a safe
   fallback.
5. The converged or fallback spin density is spherically averaged to construct
   the radial Kohn-Sham potential and numerical radial orbital family.
6. The selected radial state is combined with its spherical harmonic, sampled
   on a Cartesian grid, converted to an isosurface, and cached by dependency.
7. Only the initial result representation is built synchronously. Other tabs
   are generated lazily from cached wavefunction data when the user opens them.
8. The built-in launcher is for local use. A public deployment should import
   ``app`` through a production WSGI server and initially use one worker because
   request parameters are protected as shared module-level state by one lock.

Effective nuclear charge is not a unique observable. The definition above is
explicit, reproducible, spin-dependent, and tied to the selected local density
functional. This implementation deliberately requires LDA: GGA and hybrid
functionals require differential or nonlocal operator terms that cannot simply
be represented by the scalar v_xc(r) used here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import webbrowser
from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html
try:
    from plotly.offline import get_plotlyjs
except ImportError:  # Compatibility with older Plotly releases
    from plotly.offline.offline import get_plotlyjs
from plotly.subplots import make_subplots
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import eigh_tridiagonal
from scipy.ndimage import label as connected_components
from skimage.measure import marching_cubes
from flask import Flask, Response, redirect, render_template_string, request, send_from_directory, url_for

try:
    from scipy.special import sph_harm_y
except ImportError:
    sph_harm_y = None

try:
    from scipy.special import sph_harm
except ImportError:
    sph_harm = None

try:
    import pyscf
    from pyscf import dft, gto, scf
except ImportError as exc:
    raise ImportError(
        "PySCF is required. Install it with: python -m pip install pyscf"
    ) from exc


# =============================================================================
# USER SETTINGS
# =============================================================================

ION = "Ne"                    # Examples: H, He+, O-, Ne, Na+, Fe2+
SPIN = 0                       # N_alpha - N_beta = 2S, not multiplicity
ORBITAL_SPIN = "ALPHA"        # "ALPHA" or "BETA"

N_QUANTUM = 2                 # n >= 1
L_QUANTUM = 1                 # 0 <= l < n
M_QUANTUM = 0                 # -l <= m <= l
ORBITAL_FORM = "REAL"         # "REAL" or "COMPLEX"

BASIS = "def2-TZVPPD"
RELATIVISTIC_Z_THRESHOLD = 37
SUPERHEAVY_FAST_PATH_Z = 100
F_BLOCK_PREVIEW_RANGES = ((57, 71), (89, 103))
ENGINE_REVISION = "heavy-refined-v24"
HEAVY_PREVIEW_MAX_FOCK_UPDATES = 2
HEAVY_PREVIEW_TIME_BUDGET_SECONDS = 7.0
HEAVY_PREVIEW_GRID_LEVEL = 0
HEAVY_PREVIEW_NEW_DENSITY_WEIGHT = 0.65
HEAVY_ENSEMBLE_ENERGY_TOLERANCE = 5.0e-3
RELATIVISTIC_BASIS_CANDIDATES = ("dyall-v2z", "ano-rcc")
SPEED_OF_LIGHT_AU = 137.035999084
DFT_FUNCTIONAL = "LDA,VWN"    # Must be LDA in this implementation
QUALITY_LEVEL = 3              # 1=preview, 3=balanced, 5=maximum detail
QUALITY_PROFILE = "Balanced"
DFT_GRID_LEVEL = 3             # Reduced quadrature grid for fast atomic previews
SCF_TOLERANCE = 5.0e-8
SCF_MAX_CYCLES = 180
SCF_DAMPING = 0.35             # Used only after the ordinary SCF attempt fails
SCF_LEVEL_SHIFT = 0.40         # Hartree; stabilises small-gap occupations
SCF_NEWTON_MAX_CYCLES = 70     # Second-order SCF outer iterations
SCF_CYCLE_POLICY_VERSION = 2   # Invalidates densities made with shorter limits
SCF_SMEARING_SIGMA = 0.015     # Hartree; preconditioner only, never final result
ALLOW_ROKS_FALLBACK = True     # Open-shell fallback after UKS retries fail

RADIAL_MAX_BOHR = 24.0         # Increase for very diffuse anions
RADIAL_POINTS = 1600           # Balanced radial resolution for smooth orbitals
ANGULAR_DIRECTIONS = 38        # Balanced spherical-density quadrature
RADIAL_BLOCK_SIZE = 120        # Larger AO batches reduce Python overhead
ANGULAR_PLOT_THETA_POINTS = 161
ANGULAR_PLOT_PHI_POINTS = 321

ENCLOSED_FRACTION = 0.90
SURFACE_GRID_POINTS = 81       # Smooth default mesh while remaining interactive
SURFACE_BOX_HALF_WIDTH_BOHR = None  # None chooses from radial probability
SURFACE_GRID_BLOCK_SIZE = 16    # Slab depth; prevents large 3D temporaries
MARCHING_CUBES_STEP = 1
SURFACE_COMPONENT_MIN_VOXELS = 8
SURFACE_COMPONENT_MIN_FRACTION = 2.0e-4
SURFACE_FILTER_VERSION = 1
SURFACE_ALPHA = 0.55
DOT_MAP_POINTS = 8_000          # Probability-weighted dots in density view
DOT_MAP_SEED = 12345            # Reproducible dot sampling

ENABLE_PERSISTENT_CACHE = True
CACHE_FORMAT_VERSION = 19       # Increment whenever cached numerics change
RESULT_RENDER_VERSION = 17      # Increment whenever displayed formatting changes
RUNTIME_MODEL_VERSION = 3       # Ignore calibration ratios from older cost models
CACHE_DIRECTORY = Path(os.environ.get("ATOMIC_ORBITAL_CACHE_DIR", ".atomic_orbital_cache"))
CACHE_MAX_BYTES = 2 * 1024**3   # Oldest cache entries are removed above 2 GiB
COMMON_ORBITAL_MAX_N = 7        # Highest n cached in each requested spin/l family
RESULT_CACHE_MIN_BYTES = 20_000 # Reject incomplete/corrupt HTML cache entries
PYSCF_MAX_MEMORY_MB = 2_000
PYSCF_VERBOSE = 2

# Each slider stop is an atomic configuration: apply_quality_level() changes all
# of these values together while CALCULATION_LOCK is held. This prevents a user
# from accidentally combining, for example, a fine surface grid with a coarse
# DFT density. Runtime prediction is derived from the same workload parameters.
QUALITY_PROFILES: dict[int, dict[str, object]] = {
    1: {
        "name": "Preview",
        "description": "Lowest-cost exploratory shape",
        "dft_grid_level": 1,
        "scf_tolerance": 2.0e-5,
        "scf_max_cycles": 80,
        "scf_newton_cycles": 30,
        "radial_points": 800,
        "angular_directions": 20,
        "radial_block_size": 150,
        "angular_plot_theta_points": 91,
        "angular_plot_phi_points": 181,
        "surface_grid_points": 51,
        "surface_grid_block_size": 20,
        "marching_cubes_step": 1,
        "dot_points": 3_000,
        "heavy_preview_grid_level": 0,
        "heavy_preview_fock_updates": 1,
        "heavy_preview_time_budget_seconds": 3.0,
    },
    2: {
        "name": "Fast",
        "description": "Responsive everyday visualization",
        "dft_grid_level": 2,
        "scf_tolerance": 2.0e-6,
        "scf_max_cycles": 120,
        "scf_newton_cycles": 45,
        "radial_points": 1_100,
        "angular_directions": 26,
        "radial_block_size": 150,
        "angular_plot_theta_points": 121,
        "angular_plot_phi_points": 241,
        "surface_grid_points": 65,
        "surface_grid_block_size": 17,
        "marching_cubes_step": 1,
        "dot_points": 5_000,
        "heavy_preview_grid_level": 0,
        "heavy_preview_fock_updates": 1,
        "heavy_preview_time_budget_seconds": 5.0,
    },
    3: {
        "name": "Balanced",
        "description": "Recommended accuracy/runtime balance",
        "dft_grid_level": 3,
        "scf_tolerance": 5.0e-8,
        "scf_max_cycles": 180,
        "scf_newton_cycles": 70,
        "radial_points": 1_600,
        "angular_directions": 38,
        "radial_block_size": 120,
        "angular_plot_theta_points": 161,
        "angular_plot_phi_points": 321,
        "surface_grid_points": 81,
        "surface_grid_block_size": 16,
        "marching_cubes_step": 1,
        "dot_points": 8_000,
        "heavy_preview_grid_level": 0,
        "heavy_preview_fock_updates": 2,
        "heavy_preview_time_budget_seconds": 7.0,
    },
    4: {
        "name": "Accurate",
        "description": "Tighter SCF and smoother spatial fields",
        "dft_grid_level": 5,
        "scf_tolerance": 5.0e-10,
        "scf_max_cycles": 300,
        "scf_newton_cycles": 120,
        "radial_points": 3_200,
        "angular_directions": 74,
        "radial_block_size": 80,
        "angular_plot_theta_points": 241,
        "angular_plot_phi_points": 481,
        "surface_grid_points": 121,
        "surface_grid_block_size": 10,
        "marching_cubes_step": 1,
        "dot_points": 18_000,
        "heavy_preview_grid_level": 1,
        "heavy_preview_fock_updates": 3,
        "heavy_preview_time_budget_seconds": 15.0,
    },
    5: {
        "name": "Maximum",
        "description": "Highest resolution available in the website",
        "dft_grid_level": 7,
        "scf_tolerance": 1.0e-11,
        "scf_max_cycles": 450,
        "scf_newton_cycles": 180,
        "radial_points": 4_400,
        "angular_directions": 110,
        "radial_block_size": 60,
        "angular_plot_theta_points": 321,
        "angular_plot_phi_points": 641,
        "surface_grid_points": 161,
        "surface_grid_block_size": 8,
        "marching_cubes_step": 1,
        "dot_points": 30_000,
        "heavy_preview_grid_level": 2,
        "heavy_preview_fock_updates": 4,
        "heavy_preview_time_budget_seconds": 30.0,
    },
}


def apply_quality_level(level: int) -> dict[str, object]:
    """Apply one validated slider profile while the calculation lock is held."""
    global QUALITY_LEVEL, QUALITY_PROFILE, DFT_GRID_LEVEL, SCF_TOLERANCE
    global SCF_MAX_CYCLES, SCF_NEWTON_MAX_CYCLES, RADIAL_POINTS
    global ANGULAR_DIRECTIONS, RADIAL_BLOCK_SIZE, SURFACE_GRID_POINTS
    global SURFACE_GRID_BLOCK_SIZE, MARCHING_CUBES_STEP, DOT_MAP_POINTS
    global ANGULAR_PLOT_THETA_POINTS, ANGULAR_PLOT_PHI_POINTS
    global HEAVY_PREVIEW_GRID_LEVEL, HEAVY_PREVIEW_MAX_FOCK_UPDATES
    global HEAVY_PREVIEW_TIME_BUDGET_SECONDS

    if level not in QUALITY_PROFILES:
        raise ValueError("Accuracy level must be an integer from 1 to 5.")
    profile = QUALITY_PROFILES[level]
    QUALITY_LEVEL = level
    QUALITY_PROFILE = str(profile["name"])
    DFT_GRID_LEVEL = int(profile["dft_grid_level"])
    SCF_TOLERANCE = float(profile["scf_tolerance"])
    SCF_MAX_CYCLES = int(profile["scf_max_cycles"])
    SCF_NEWTON_MAX_CYCLES = int(profile["scf_newton_cycles"])
    RADIAL_POINTS = int(profile["radial_points"])
    ANGULAR_DIRECTIONS = int(profile["angular_directions"])
    RADIAL_BLOCK_SIZE = int(profile["radial_block_size"])
    ANGULAR_PLOT_THETA_POINTS = int(profile["angular_plot_theta_points"])
    ANGULAR_PLOT_PHI_POINTS = int(profile["angular_plot_phi_points"])
    SURFACE_GRID_POINTS = int(profile["surface_grid_points"])
    SURFACE_GRID_BLOCK_SIZE = int(profile["surface_grid_block_size"])
    MARCHING_CUBES_STEP = int(profile["marching_cubes_step"])
    DOT_MAP_POINTS = int(profile["dot_points"])
    HEAVY_PREVIEW_GRID_LEVEL = int(profile["heavy_preview_grid_level"])
    HEAVY_PREVIEW_MAX_FOCK_UPDATES = int(profile["heavy_preview_fock_updates"])
    HEAVY_PREVIEW_TIME_BUDGET_SECONDS = float(
        profile["heavy_preview_time_budget_seconds"]
    )
    return profile

WEB_HOST = "127.0.0.1"
WEB_PORT = 5000
ISOTOPE_MASS_NUMBER: int | None = None


ELEMENT_SYMBOLS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
ATOMIC_NUMBERS = {symbol: i for i, symbol in enumerate(ELEMENT_SYMBOLS, 1)}

# Recommended neutral-atom 2S values from the conventional ground-state
# configurations and Hund coupling.  The menu still presents every spin that
# is compatible with integer alpha/beta electron populations for the chosen
# ion; this table only chooses the most useful initial selection.
NEUTRAL_GROUND_STATE_2S = (
    1, 0,  # H-He
    1, 0, 1, 2, 3, 2, 1, 0,  # Li-Ne
    1, 0, 1, 2, 3, 2, 1, 0,  # Na-Ar
    1, 0, 1, 2, 3, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0,  # K-Kr
    1, 0, 1, 2, 5, 6, 5, 4, 3, 0, 1, 0, 1, 2, 3, 2, 1, 0,  # Rb-Xe
    1, 0,  # Cs-Ba
    1, 2, 3, 4, 5, 6, 7, 8, 5, 4, 3, 2, 1, 0, 1,  # La-Lu
    2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0,  # Hf-Rn
    1, 0,  # Fr-Ra
    1, 2, 3, 4, 5, 6, 7, 8, 5, 4, 3, 2, 1, 0, 1,  # Ac-Lr
    2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0,  # Rf-Og
)
GROUND_STATE_2S = {
    symbol: NEUTRAL_GROUND_STATE_2S[index]
    for index, symbol in enumerate(ELEMENT_SYMBOLS)
}

