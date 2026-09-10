#!/bin/bash
set -euo pipefail

APP_ROOT="/home/fwzc2/apps/kohn-sham-dft"
RUNTIME_ROOT="/home/fwzc2/var/kohn-sham-dft"
SOCKET_PATH="$APP_ROOT/web.sock"

cd "$APP_ROOT"
source "$APP_ROOT/.venv/bin/activate"
mkdir -p "$RUNTIME_ROOT/cache" "$RUNTIME_ROOT/tmp" "$RUNTIME_ROOT/logs"
rm -f "$SOCKET_PATH"

export ATOMIC_ORBITAL_CACHE_DIR="$RUNTIME_ROOT/cache"
export PYSCF_TMPDIR="$RUNTIME_ROOT/tmp"
export TMPDIR="$RUNTIME_ROOT/tmp"
export PYTHONUNBUFFERED=1

exec gunicorn \
  --bind "unix:$SOCKET_PATH" \
  --workers 1 \
  --worker-class gthread \
  --threads 4 \
  --timeout 600 \
  --access-logfile "$RUNTIME_ROOT/logs/access.log" \
  --error-logfile "$RUNTIME_ROOT/logs/error.log" \
  atomic_orbital_master:app
