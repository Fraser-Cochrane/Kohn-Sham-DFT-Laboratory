FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ATOMIC_ORBITAL_CACHE_DIR=/var/data/atomic-orbital-cache
ENV PYSCF_TMPDIR=/tmp/pyscf

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --worker-class gthread --threads 4 --timeout 600 atomic_orbital_master:app"]
