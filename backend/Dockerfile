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


# install ONLY runtime dependency and DVC
RUN apt-get update && apt-get install -y \
    libgomp1 git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir dvc[s3]


# copy installed python packages
COPY --from=builder /install /usr/local



# Copy DVC files and configs (from project root context)
COPY .dvc .dvc
COPY dvc.yaml dvc.yaml
COPY dvc.lock dvc.lock
COPY data/loan_2019_20.csv.dvc data/loan_2019_20.csv.dvc


# Copy .git directory so DVC recognizes the repo
COPY .git .git

# Debug: list contents to verify .dvc and .git are present
RUN ls -la /app && ls -la /app/.dvc && ls -la /app/.git


# Pull models and data from DVC remote
RUN dvc pull || exit 1

# Remove .git to keep image clean
RUN rm -rf .git

# copy backend app and ml (models will be present after dvc pull)
COPY backend/ ./backend/
COPY ml/ ./ml/

# expose port
EXPOSE 8000

# strip binaries to reduce image size
RUN find /usr/local -type f -name '*.so*' -exec strip --strip-unneeded {} + || true

# run app
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
