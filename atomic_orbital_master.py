"""Master launcher for the multi-file Atomic Orbital Explorer.

Run only this file.  It loads the nine numbered source files from the same
directory and then starts the existing Flask webpage:

    python atomic_orbital_master.py

In a JupyterLab notebook cell, the equivalent command is:

    %run atomic_orbital_master.py

Why the files share one namespace
---------------------------------
The final single-file application intentionally keeps the selected atom,
quality profile, cache handles, and request lock as module-level state.  A
normal import-based refactor would copy or relocate some of that mutable state
and could subtly change results.  The small loader below instead compiles each
readable part and executes it in this master module's global namespace.  This
preserves the original initialization order and runtime behaviour while making
the source much easier to navigate in JupyterLab.

File order
----------
1. orbital_01_settings.py
2. orbital_02_cache_and_validation.py
3. orbital_03_atomic_dft.py
4. orbital_04_radial_solver.py
5. orbital_05_spatial_grid.py
6. orbital_06_figures.py
7. orbital_07_result_pipeline.py
8. orbital_08_web_templates.py
9. orbital_09_web_server.py

The numbered files are components, not independent programs.  Keep all ten
Python files together and launch this master file only.
"""

from __future__ import annotations

from pathlib import Path


# Resolve components relative to this file, not the notebook's current working
# directory.  This lets `%run path/to/atomic_orbital_master.py` work reliably.
PROJECT_DIRECTORY = Path(__file__).resolve().parent

# The tuple is the application's dependency order.  A later part may use names
# defined by any earlier part, matching their order in the original monolith.
SCRIPT_PARTS = (
    "orbital_01_settings.py",
    "orbital_02_cache_and_validation.py",
    "orbital_03_atomic_dft.py",
    "orbital_04_radial_solver.py",
    "orbital_05_spatial_grid.py",
    "orbital_06_figures.py",
    "orbital_07_result_pipeline.py",
    "orbital_08_web_templates.py",
    "orbital_09_web_server.py",
)


def _execute_project_part(filename: str) -> None:
    """Compile and execute one component in the master's shared namespace.

    Supplying the component path to compile() preserves useful filenames in
    tracebacks, so an error points to the relevant readable source file.
    """
    script_path = PROJECT_DIRECTORY / filename
    if not script_path.is_file():
        raise FileNotFoundError(
            f"Required project file is missing: {script_path}. "
            "Keep every numbered Python file beside the master script."
        )
    source = script_path.read_text(encoding="utf-8")
    compiled = compile(source, str(script_path), "exec")
    exec(compiled, globals(), globals())


# Loading happens at import time so a production WSGI server can use
# `atomic_orbital_master:app` without starting the local development server.
for _script_filename in SCRIPT_PARTS:
    _execute_project_part(_script_filename)


# Running the master starts the same webpage as the final single-file version.
# Importing the master exposes `app` but deliberately does not launch a browser.
if __name__ == "__main__":
    launch_web_application()
