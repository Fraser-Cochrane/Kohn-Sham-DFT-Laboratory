#!/bin/bash
set -euo pipefail

APP_ROOT="/home/fwzc2/apps/kohn-sham-dft"
RUNTIME_ROOT="/home/fwzc2/var/kohn-sham-dft"
SOCKET_PATH="$APP_ROOT/web.sock"

cd "$APP_ROOT"
source "$APP_ROOT/.venv/bin/activate"
mkdir -p "$RUNTIME_ROOT/cache" "$RUNTIME_ROOT/tmp" "$RUNTIME_ROOT/logs"
rm -f "$SOCKET_PATH"
: > "$RUNTIME_ROOT/logs/error.log"

export ATOMIC_ORBITAL_CACHE_DIR="$RUNTIME_ROOT/cache"
export ATOMIC_ORBITAL_CACHE_MAX_MB=128
export ATOMIC_ORBITAL_COMPRESS_CACHE=1
export PYSCF_TMPDIR="$RUNTIME_ROOT/tmp"
export TMPDIR="$RUNTIME_ROOT/tmp"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

exec gunicorn \
  --bind "unix:$SOCKET_PATH" \
  --workers 1 \
  --worker-class gthread \
  --threads 4 \
  --timeout 600 \
  --access-logfile /dev/null \
  --error-logfile "$RUNTIME_ROOT/logs/error.log" \
  atomic_orbital_master:app
