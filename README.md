# Kohn-Sham-DFT-Laboratory
An interactive atomic-orbital explorer using PySCF density-functional theory and numerical Kohn-Sham solvers to model atoms and ions. Its periodic-table GUI lets users choose ion, spin, quantum numbers and accuracy, then explore radial functions, angular wavefunctions, density maps, contours and 3D probability isosurfaces with caching.

## SRCF deployment

The application supports both a domain root and a path-prefixed reverse proxy.
For the `fwzc2` project index it is mounted at `/atomic-orbitals/`; browser
forms, result links, and lazy Plotly endpoints use path-relative URLs so they
remain within that mount. `run-srcf.sh` starts the production Gunicorn service
on `/home/fwzc2/apps/kohn-sham-dft/web.sock` with one worker and persistent
cache, temporary, and log directories under `/home/fwzc2/var/kohn-sham-dft`.
