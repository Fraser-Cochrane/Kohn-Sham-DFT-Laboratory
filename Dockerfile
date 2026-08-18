# Use a stable Linux Python environment with broad scientific-wheel support.
FROM python:3.12-slim

# Prevent Python from generating unnecessary bytecode files and ensure logs are
# immediately visible in the Render dashboard.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Store reusable calculation results on the Render persistent disk.
ENV ATOMIC_ORBITAL_CACHE_DIR=/var/data/atomic-orbital-cache

# PySCF needs an existing, writable directory for temporary calculation files.
ENV PYSCF_TMPDIR=/tmp/pyscf
ENV TMPDIR=/tmp/pyscf

# Application files are installed here.
WORKDIR /app

# libgomp1 supplies the OpenMP runtime used by scientific Python packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies separately so Docker can reuse this layer when only the
# application source changes.
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        --no-cache-dir \
        --prefer-binary \
        -r /app/requirements.txt

# Copy the master and all nine component scripts into the image.
COPY . /app

# Create the directories during the build. The start command repeats this step
# because Render's persistent-disk mount can replace /var/data at runtime.
RUN mkdir -p \
    /tmp/pyscf \
    /var/data/atomic-orbital-cache

# Render normally supplies PORT=10000.
EXPOSE 10000

# Create runtime directories after all mounts are attached, then start the
# production WSGI server. One worker preserves the application's shared state.
CMD ["sh", "-c", "mkdir -p /tmp/pyscf /var/data/atomic-orbital-cache && exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --worker-class gthread --threads 4 --timeout 600 atomic_orbital_master:app"]
