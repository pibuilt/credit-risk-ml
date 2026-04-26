# Credit Risk ML Platform

Production-grade machine learning platform for predicting loan default risk using a hybrid tabular + text feature pipeline, explainability outputs, and containerized serving.

Dataset reference:
https://www.kaggle.com/datasets/beatafaron/loan-credit-risk-and-population-stability

This repository is built around realistic ML engineering constraints:

- Reproducible training and artifact tracking
- Data/model versioning through DVC + S3
- Inference API with telemetry and structured outputs
- Human-readable risk score + risk band mapping
- Local and production deployment via Docker Compose
- Reverse proxy and TLS handling through nginx

---

## Table of Contents

1. [What This Project Solves](#what-this-project-solves)
2. [High-Level Architecture](#high-level-architecture)
3. [Tech Stack and Why Each Part Exists](#tech-stack-and-why-each-part-exists)
4. [Repository Layout](#repository-layout)
5. [Data and Versioning Strategy](#data-and-versioning-strategy)
6. [ML Training Pipeline](#ml-training-pipeline)
7. [Inference Service Design](#inference-service-design)
8. [Frontend Behavior](#frontend-behavior)
9. [Routing and Reverse Proxy Behavior](#routing-and-reverse-proxy-behavior)
10. [Environment and Secrets](#environment-and-secrets)
11. [Local Development Guide](#local-development-guide)
12. [Training and Experiment Tracking](#training-and-experiment-tracking)
13. [API Reference](#api-reference)
14. [Docker and Deployment](#docker-and-deployment)
15. [Production HTTPS Setup](#production-https-setup)
16. [Testing](#testing)
17. [Operational Troubleshooting](#operational-troubleshooting)
18. [Known Trade-offs and Future Hardening](#known-trade-offs-and-future-hardening)

---

## What This Project Solves

Lending risk decisions typically need more than a single probability output. Real systems need:

- A robust prediction pipeline that can handle sparse or partially missing request fields.
- A business-readable score and category, not just raw model probability.
- Feature-level explainability to support trust and analysis.
- Reproducible training outputs and data lineage.
- Predictable deployment behavior across local/dev/prod environments.

This project addresses those requirements end to end.

---

## High-Level Architecture

```text
Client (Browser)
  |
  | HTTP dev (:8080) or HTTPS prod (:443)
  v
nginx reverse proxy
  |- /             -> frontend service (React dev server)
  '- /api/*        -> backend service (FastAPI)

FastAPI backend
  |- request logging middleware
  |- request metrics accumulation
  |- lazy model load
  |- lazy SHAP explainer load
  '- /v1/predict for inference

ML model artifact
  '- models/credit_model_v1.pkl

DVC metadata + remote storage
  '- S3: s3://dvc-demo-pi/dvcstore
```

### Important routing behavior

The frontend posts to `/api/v1/predict`.

nginx rule for `/api/` uses a trailing slash in `proxy_pass`, so the prefix is stripped before forwarding:

- External: `/api/v1/predict`
- Internal backend: `/v1/predict`

This is intentional and expected.

---

## Tech Stack and Why Each Part Exists

### Core serving

- FastAPI: low-latency API, schema validation integration.
- Uvicorn: ASGI server runtime.
- Pydantic: strict request schema and type enforcement.

### ML and feature engineering

- scikit-learn: preprocessing, pipelines, baseline models.
- LightGBM: high-performance gradient boosting for tabular credit tasks.
- Optuna: focused hyperparameter search.
- SHAP: local/global model explanation artifacts.

### Data and experiment operations

- pandas/numpy: data manipulation and numerical operations.
- MLflow: experiment run logging and artifact indexing.
- DVC: data/model/report lineage and reproducibility.

### Frontend and networking

- React (CRA): user interface and client-side validation.
- Axios: API calls.
- nginx: reverse proxy, HTTPS termination, route segmentation.

### Packaging and runtime

- Docker: deterministic container builds.
- Docker Compose: multi-service orchestration in dev and prod profiles.

---

## Repository Layout

```text
ml-credit-risk/
|- backend/
|  |- Dockerfile
|  |- requirements.txt
|  '- app/
|     |- main.py
|     |- schemas/prediction.py
|     '- services/model_service.py
|
|- ml/
|  |- data.py
|  |- features.py
|  '- train.py
|
|- frontend/
|  |- Dockerfile
|  |- package.json
|  '- src/App.js
|
|- nginx/
|  |- nginx.dev.conf
|  '- nginx.conf
|
|- data/
|  '- loan_2019_20.csv.dvc
|- models/
|- reports/
|- tests/test_api.py
|- dvc.yaml
|- dvc.lock
|- docker-compose.yml
'- docker-compose.prod.yml
```

### Key ownership boundaries

- `ml/` owns offline training and reporting.
- `backend/` owns online inference behavior and API contracts.
- `frontend/` owns user interactions and payload shaping.
- `nginx/` owns all external routing and TLS policy.
- `dvc.yaml` owns reproducible train-stage workflow.

---

## Data and Versioning Strategy

### Why DVC is critical here

This project has large and evolving assets:

- raw dataset
- trained model
- evaluation reports/plots

Storing these directly in Git would bloat history and reduce collaboration speed. DVC keeps these artifacts in remote object storage while Git tracks lightweight pointers and pipeline metadata.

### Current DVC remote

Configured in `.dvc/config`:

- remote name: `myremote`
- remote URL: `s3://dvc-demo-pi/dvcstore`

### Tracked artifacts

- `data/loan_2019_20.csv`
- `models/credit_model_v1.pkl`
- `reports/`

### Typical lifecycle

```bash
# Fetch data and artifacts for local run
 dvc pull

# Recompute training stage if dependencies changed
 dvc repro

# Push updated data/model/report artifacts
 dvc push
```

### Why backend Docker build copies `.git`

DVC expects repository context when resolving tracked assets.
The backend image build copies DVC metadata and `.git`, runs `dvc pull`, then removes `.git` to keep runtime image cleaner.

---

## ML Training Pipeline

The pipeline is defined in `ml/train.py` and composed through functions from `ml/data.py` and `ml/features.py`.

### Step 1: target preparation

- Keep rows where `loan_status` is one of:
  - `Fully Paid`
  - `Charged Off`
  - `Default`
- Create binary `default` target:
  - 1 for `Charged Off` and `Default`
  - 0 for `Fully Paid`

### Step 2: leakage and missingness reduction

- Remove known leakage columns (post-outcome signals and downstream payment details).
- Drop columns with missingness above threshold (`> 50%`).

### Step 3: split policy

- Stratified train/validation/test split with 70/15/15 proportions.
- Preserves class ratio across splits.

### Step 4: preprocessing graph

The `ColumnTransformer` includes multiple branches:

1. Numeric branch
   - median imputation
   - standard scaling

2. Categorical branch
   - most-frequent imputation
   - one-hot encoding with unknown category safety

3. Text branches
   - columns: `emp_title`, `title`
   - null-safe string conversion
   - TF-IDF vectorization with bounded vocabulary size

4. Synthetic branch
   - custom `RiskClusterTransformer` over numeric features
   - KMeans cluster ID as an additional feature

### Step 5: model competition

Three candidate models are evaluated through cross-validation:

- Logistic Regression
- Random Forest
- LightGBM

Selection criterion prioritizes PR-AUC behavior (helpful for imbalanced default contexts).

### Step 6: tuning path

When LightGBM is selected, Optuna runs targeted hyperparameter optimization.

### Step 7: evaluation outputs

Validation metrics include:

- ROC-AUC
- PR-AUC
- F1 score
- Brier score
- confusion matrix

Plots generated:

- confusion matrix image
- ROC curve image
- precision-recall curve image
- SHAP summary image

### Step 8: artifacts

Saved outputs:

- `models/credit_model_v1.pkl`
- `models/metadata.json`
- `reports/metrics.json`
- report plots in `reports/`

---

## Inference Service Design

Backend entrypoint is `backend/app/main.py`.

### Runtime behavior summary

- Initializes logging.
- Initializes in-memory metrics on startup.
- Uses lock-protected lazy loading for model and SHAP explainer.
- Assigns per-request UUID and logs latency in middleware.

### Input schema strategy

Request schema in `backend/app/schemas/prediction.py` accepts batch payload:

```json
{"data": [{...loan application fields...}]}
```

Only a minimum field subset is mandatory for valid prediction requests:

- `loan_amnt`
- `annual_inc`
- `dti`
- `fico_range_low`

Other fields are optional to support lighter client payloads.

### Column alignment strategy

Before inference, backend aligns incoming frame to the model pipeline expected feature order (`feature_names_in_`). Missing expected columns are auto-added with nulls, then preprocessor handles imputation.

### Explainability output strategy

For each request, SHAP values are computed from transformed features and returned as top factors (`feature_i` labels).

### Business score mapping

Probability is transformed to score:

- `risk_score = 850 - probability * 550`

Risk band mapping:

- score >= 750: Low Risk
- score >= 650: Medium Risk
- score >= 550: High Risk
- else: Very High Risk

---

## Frontend Behavior

Frontend logic in `frontend/src/App.js`:

1. Captures minimal required values.
2. Performs client-side validation (required, numeric positivity/range basics).
3. Calls `POST /api/v1/predict`.
4. Renders returned score, level, probability, and cluster.

### Why call `/api/...` from frontend

Frontend should avoid hardcoded backend host/port assumptions.

Using a relative nginx path:

- improves portability between local and production
- centralizes routing policy in nginx
- avoids CORS complexity at browser layer in proxied mode

---

## Routing and Reverse Proxy Behavior

### Development nginx (`nginx/nginx.dev.conf`)

- listens on port 80 (container)
- `/api/` -> backend:8000
- `/` -> frontend:3000

Compose maps host `8080:80`, so local entrypoint is `http://localhost:8080`.

### Production nginx (`nginx/nginx.conf`)

- `80` redirects to HTTPS
- `443` serves TLS traffic
- `/` proxied to frontend
- `/api/` proxied to backend
- certificates read from `/etc/letsencrypt/...`

---

## Environment and Secrets

Template provided in `.env.example`:

```env
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
```

### Local setup

```bash
cp .env.example .env
```

Fill actual credentials for S3 access.

### Security guidance

- Do not commit `.env`.
- Rotate keys if accidentally exposed.
- In cloud deployments, prefer IAM role-based access instead of static keys.

---

## Local Development Guide

### Prerequisites

- Docker Desktop or Docker Engine + compose plugin
- Python 3.11
- Node.js 18+

### Path A: full stack with Docker Compose

```bash
docker-compose up --build
```

What starts:

- backend service (FastAPI)
- frontend service (React dev server)
- nginx reverse proxy

Where to open:

- app: `http://localhost:8080`

Health check:

```bash
curl http://localhost:8080/api/health
```

Prediction check:

```bash
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"data":[{"loan_amnt":10000,"annual_inc":60000,"dti":15,"fico_range_low":680}]}'
```

### Path B: run services manually

Backend:

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm start
```

Important:

- Direct backend testing route is `/v1/predict`.
- Proxied route is `/api/v1/predict`.

---

## Training and Experiment Tracking

### Direct training

```bash
python -m ml.train
```

### DVC-managed training

```bash
dvc repro
```

### MLflow UI

```bash
mlflow ui
```

Open:

- `http://localhost:5000`

MLflow tracks:

- selected model name
- tuning parameters
- final metrics
- generated report artifacts

---

## API Reference

### GET /health

Purpose: liveness check.

Response:

```json
{ "status": "ok" }
```

### GET /metrics

Purpose: expose in-memory serving telemetry.

Response fields:

- `request_count`
- `error_count`
- `avg_latency`

### POST /v1/predict (backend direct)

Request body:

```json
{
  "data": [
    {
      "loan_amnt": 10000,
      "annual_inc": 60000,
      "dti": 15.0,
      "fico_range_low": 680
    }
  ]
}
```

Response shape:

```json
{
  "request_id": "uuid",
  "predictions": [
    {
      "default_probability": 0.123,
      "risk_score": 782,
      "risk_level": "Low Risk",
      "risk_cluster": 2
    }
  ],
  "top_factors": [{ "feature": "feature_12", "impact": 0.08 }]
}
```

### POST /api/v1/predict (browser/proxy path)

Same payload/response semantics as backend direct path.

---

## Docker and Deployment

### Dev compose file

`docker-compose.yml`:

- backend builds with `backend/Dockerfile`
- frontend builds with `frontend/Dockerfile`
- nginx uses dev config and exposes host port 8080

### Prod compose file

`docker-compose.prod.yml`:

- nginx exposes 80 and 443
- mounts production nginx config
- mounts host letsencrypt cert directory

Run production stack:

```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

---

## Production HTTPS Setup

Production nginx expects:

- `/etc/letsencrypt/live/creditriskml.duckdns.org/fullchain.pem`
- `/etc/letsencrypt/live/creditriskml.duckdns.org/privkey.pem`

Issue certificate on host before starting production compose:

```bash
sudo certbot certonly --standalone -d creditriskml.duckdns.org
```

Then start services:

```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

Validation checks:

```bash
curl -I http://creditriskml.duckdns.org
curl -I https://creditriskml.duckdns.org
```

Expected:

- HTTP redirects to HTTPS
- HTTPS serves frontend
- `/api/` endpoints reachable via TLS

---

## Testing

Run all tests:

```bash
pytest -q
```

Current API tests cover:

- health endpoint
- valid prediction response path
- empty payload behavior
- invalid payload type behavior
- metrics endpoint

---

## Operational Troubleshooting

### DVC pull failures during backend build

Symptoms:

- backend image build exits at `dvc pull`

Likely causes:

- invalid AWS credentials
- missing S3 permissions
- blocked outbound network

Checks:

```bash
dvc remote list
dvc pull -v
```

### Runtime model not found

Symptoms:

- backend raises model file not found/load errors

Likely causes:

- `dvc pull` not executed or failed
- training never generated model in expected path

Fix:

- verify `models/credit_model_v1.pkl` exists
- rerun `dvc pull` or `python -m ml.train`

### Prediction route confusion

Symptoms:

- 404 or unexpected route mismatch

Fix:

- use `/api/v1/predict` through nginx
- use `/v1/predict` when calling backend directly

### SHAP latency overhead

Symptoms:

- prediction endpoint slower under load

Cause:

- SHAP computation per request

Mitigation options:

- make explainability optional via query/body flag
- precompute explanations in selected workflows
- disable SHAP in high-throughput mode

---

## Known Trade-offs and Future Hardening

Current implementation is intentionally practical but leaves room for production hardening.

1. Frontend container runs CRA dev server.
   - Improvement: build static assets and serve directly via nginx.

2. In-memory telemetry resets on backend restart.
   - Improvement: export metrics to Prometheus or a persistent store.

3. SHAP output labels are generic (`feature_i`).
   - Improvement: map transformed indices back to meaningful feature names.

4. Prediction schema is permissive for optional fields.
   - Improvement: add tighter schema checks for categorical vocabularies and ranges.

5. Current docs do not include CI/CD because no workflow files are present in this repository snapshot.
   - Improvement: add GitHub Actions for tests/build/deploy and document those stages.

---

## Request Lifecycle Deep Dive

This section traces a single prediction request from browser click to response payload.

### Step 1: Browser event and payload creation

In the frontend, user clicks Predict.

Client builds payload:

```json
{
  "data": [
    {
      "loan_amnt": 12000,
      "annual_inc": 70000,
      "dti": 18.2,
      "fico_range_low": 690
    }
  ]
}
```

### Step 2: Client sends request to proxy route

Frontend issues:

- `POST /api/v1/predict`

No backend hostname is embedded in frontend code.

### Step 3: nginx route rewrite

nginx receives `/api/v1/predict` and forwards to backend with stripped prefix due to trailing slash behavior:

- upstream receives `/v1/predict`

### Step 4: FastAPI middleware instrumentation

Before route handler runs:

- request UUID generated
- start timestamp captured

After response:

- request counter incremented
- total latency accumulated
- structured log emitted

### Step 5: Schema validation

FastAPI validates request body via Pydantic model.

If payload shape/types are invalid:

- route handler is not executed
- response is HTTP 422 with validation details

### Step 6: Lazy model + explainer retrieval

On first prediction request:

- model is loaded from `models/credit_model_v1.pkl`
- SHAP tree explainer initialized

On later requests:

- cached objects are reused

### Step 7: Input alignment and inference

Backend creates DataFrame and aligns columns to `pipeline.feature_names_in_`, filling missing expected columns with nulls.

Pipeline runs:

- preprocessing transform
- model probability output

### Step 8: Business projection + explainability

Backend returns:

- probability
- derived risk score
- risk level
- risk cluster
- top SHAP factors

### Step 9: Frontend render

Frontend displays result panel with risk fields for user interpretation.

---

## Error Contract Examples

### 422 validation error example

Request with wrong top-level shape:

```json
{ "data": "not-a-list" }
```

Typical response (shape simplified):

```json
{
  "detail": [
    {
      "loc": ["body", "data"],
      "msg": "Input should be a valid list"
    }
  ]
}
```

### 400 empty data error

Request:

```json
{ "data": [] }
```

Behavior:

- route-level guard returns 400 for empty batch

### 500 inference error

Returned when unexpected runtime exceptions happen in inference flow (for example model loading failure or incompatible artifact).

---

## Developer SOP

Use this checklist for safe local changes.

1. Pull latest code.
2. Refresh environment file if needed.
3. Run `dvc pull`.
4. Run `pytest -q`.
5. If ML code changed, run `python -m ml.train` and inspect reports.
6. If model/report outputs changed intentionally, run `dvc repro` and `dvc push`.
7. Bring up stack with `docker-compose up --build` and verify prediction endpoint.

---

## Incident Response Mini-Runbook

### Incident: predictions suddenly fail after deploy

Immediate checks:

1. Inspect backend logs.
2. Verify model file exists inside backend container.
3. Verify request route used (`/api/v1/predict` through proxy).
4. Verify input payload still matches schema.

Containment actions:

1. Roll back to previous image/tag if available.
2. Restore previously known-good DVC model artifact hash.
3. Restart backend and re-run smoke test.

### Incident: large latency spike

Checks:

1. Inspect request volume and backend CPU.
2. Check SHAP overhead under current traffic.
3. Verify no repeated model/explainer reloads.

Mitigations:

1. Temporarily disable explainability path (code flag rollout).
2. Increase backend workers/resources.
3. Introduce queueing or rate limits at edge.

---
