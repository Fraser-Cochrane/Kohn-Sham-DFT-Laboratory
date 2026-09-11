# Kohn-Sham-DFT-Laboratory
An interactive atomic-orbital explorer using PySCF density-functional theory and numerical Kohn-Sham solvers to model atoms and ions. Its periodic-table GUI lets users choose ion, spin, quantum numbers and accuracy, then explore radial functions, angular wavefunctions, density maps, contours and 3D probability isosurfaces with caching.

## Installation

Python 3.10 or newer is recommended. Create an isolated environment and install
only the runtime dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir --upgrade pip
.venv/bin/python -m pip install --no-cache-dir --no-compile -r requirements.txt
.venv/bin/python scripts/prune_runtime.py
```

Run the local interface with:

```bash
.venv/bin/python atomic_orbital_master.py
```

## Compact storage

The scientific engine retains PySCF, NumPy and SciPy, but avoids two much
larger general-purpose data/vision packages. PyMCubes supplies the one compiled
isosurface operation required by the application. PySCF's bundled basis library
supplies the configured orbital bases directly.

Cached numerical arrays and Plotly figure JSON are compressed. Result pages
refer to one application-served Plotly runtime instead of embedding another
multi-megabyte copy in every result. The cache uses a 256 MiB least-recently-used
limit by default and discards application-owned entries from obsolete cache
formats automatically.

The following environment variables tune storage without editing the source:

- `ATOMIC_ORBITAL_CACHE_DIR` selects the persistent cache directory.
- `ATOMIC_ORBITAL_CACHE_MAX_MB` sets the cache limit, with a minimum of 64 MiB.
- `ATOMIC_ORBITAL_COMPRESS_CACHE=0` opts into faster, uncompressed NPY storage.

## SRCF deployment

The application supports both a domain root and a path-prefixed reverse proxy.
For the `fwzc2` project index it is mounted at `/atomic-orbitals/`; browser
forms, result links, and lazy Plotly endpoints use path-relative URLs so they
remain within that mount. `run-srcf.sh` starts the production Gunicorn service
on `/home/fwzc2/apps/kohn-sham-dft/web.sock` with one worker and persistent
cache, temporary, and log directories under `/home/fwzc2/var/kohn-sham-dft`.
The supplied launcher enables compressed caching, caps it at 128 MiB, suppresses
access-log growth, and truncates the diagnostic error log at each restart.

After pulling an update, install the compact requirements and restart the
service:

```bash
cd /home/fwzc2/apps/kohn-sham-dft
.venv/bin/python -m pip install --no-cache-dir --no-compile -r requirements.txt
.venv/bin/python scripts/prune_runtime.py
systemctl --user restart kohn-sham-dft.service
```
