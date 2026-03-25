# -----------------------------
# STAGE 1 — BUILD
# -----------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# install build deps
RUN apt-get update && apt-get install -y gcc libgomp1 && rm -rf /var/lib/apt/lists/*

# copy requirements
COPY backend/requirements.txt .

# install deps into custom dir
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && apt-get purge -y --auto-remove gcc \
    && rm -rf /root/.cache /tmp/pip-* /var/lib/apt/lists/*

# -----------------------------
# STAGE 2 — RUNTIME
# -----------------------------
FROM python:3.11-slim

WORKDIR /app

# install ONLY runtime dependency
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# copy installed python packages
COPY --from=builder /install /usr/local

# copy only backend app, models, and ml (not data, .git, etc.)
COPY backend/ ./backend/
COPY models/ ./models/
COPY ml/ ./ml/

# expose port
EXPOSE 8000

# strip binaries to reduce image size
RUN find /usr/local -type f -name '*.so*' -exec strip --strip-unneeded {} + || true

# run app
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
