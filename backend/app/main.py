import logging
import time
import uuid

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.app.schemas.prediction import PredictionRequest
import pandas as pd
import shap


from backend.app.services.model_service import ModelService
from threading import Lock

_model_lock = Lock()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------
# LOGGING SETUP
# -----------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


setup_logging()
logger = logging.getLogger(__name__)


# -----------------------------------
# REQUEST MIDDLEWARE (TRACE + LATENCY)
# -----------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.time()

    try:
        response = await call_next(request)
    except Exception as e:
        app.state.metrics["error_count"] += 1  # ✅ track errors
        logger.error(f"request_id={request_id} unhandled_error={str(e)}")
        raise

    latency = time.time() - start_time

    # ✅ update metrics
    app.state.metrics["request_count"] += 1
    app.state.metrics["total_latency"] += latency

    logger.info(
        f"request_id={request_id} "
        f"method={request.method} "
        f"path={request.url.path} "
        f"status={response.status_code} "
        f"latency={latency:.4f}s"
    )

    return response


# -----------------------------------
# STARTUP (SAFE MODEL LOADING + METRICS)
# -----------------------------------


# Lazy loading for model and explainer
def get_model_service():
    with _model_lock:
        if not hasattr(app.state, "model_service"):
            logger.info("Loading model_service...")
            app.state.model_service = ModelService("models/credit_model_v1.pkl")
    return app.state.model_service

def get_explainer(model_service):
    with _model_lock:
        if not hasattr(app.state, "explainer"):
            logger.info("Loading SHAP explainer...")
            model = model_service.pipeline.named_steps["model"]
            app.state.explainer = shap.TreeExplainer(model)
    return app.state.explainer

# Metrics always initialized at startup
@app.on_event("startup")
def startup_event():
    logger.info("Starting FastAPI application")
    app.state.metrics = {
        "request_count": 0,
        "error_count": 0,
        "total_latency": 0.0
    }
    logger.info("Metrics initialized successfully")

# -----------------------------------
# INPUT ADAPTER
# -----------------------------------

def build_full_dataframe(input_data, model_service):
    df = pd.DataFrame(input_data)

    expected_cols = model_service.pipeline.feature_names_in_

    missing_cols = []

    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
            missing_cols.append(col)

    if missing_cols:
        logger.warning(
            f"Missing columns auto-filled (first 10): {missing_cols[:10]}"
        )

    df = df[expected_cols]

    return df


# -----------------------------------
# SHAP EXPLAINABILITY
# -----------------------------------

def get_top_features(df, model_service, explainer, top_n=5):

    pipeline = model_service.pipeline

    X_transformed = pipeline.named_steps["preprocessor"].transform(df)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    values = shap_values[0]

    feature_names = [f"feature_{i}" for i in range(len(values))]

    top_idx = sorted(
        range(len(values)),
        key=lambda i: abs(values[i]),
        reverse=True
    )[:top_n]

    return [
        {
            "feature": feature_names[i],
            "impact": float(values[i])
        }
        for i in top_idx
    ]


# -----------------------------------
# HEALTH ENDPOINT
# -----------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


# -----------------------------------
# METRICS ENDPOINT
# -----------------------------------

@app.get("/metrics")
def get_metrics():
    metrics = app.state.metrics

    request_count = metrics["request_count"]
    error_count = metrics["error_count"]
    total_latency = metrics["total_latency"]

    avg_latency = (
        total_latency / request_count
        if request_count > 0 else 0
    )

    return {
        "request_count": request_count,
        "error_count": error_count,
        "avg_latency": round(avg_latency, 4)
    }


# -----------------------------------
# PREDICTION ENDPOINT
# -----------------------------------

@app.post("/v1/predict")
def predict(request: PredictionRequest, req: Request):

    request_id = req.state.request_id


    try:
        model_service = get_model_service()
        explainer = get_explainer(model_service)

        df = build_full_dataframe(
            [item.model_dump() for item in request.data],
            model_service
        )

        if df.empty:
            raise HTTPException(status_code=400, detail="Empty input data")

        results = model_service.predict_with_risk(df)

        # SHAP
        top_factors = get_top_features(df, model_service, explainer)

        # logging prediction
        logger.info(
            f"request_id={request_id} "
            f"prediction={results[0]['default_probability']:.4f} "
            f"risk={results[0]['risk_level']}"
        )

        return {
            "request_id": request_id,
            "predictions": results,
            "top_factors": top_factors
        }

    except ValueError as e:
        app.state.metrics["error_count"] += 1  # ✅ track errors
        logger.error(
            f"request_id={request_id} validation_error={str(e)}"
        )
        raise HTTPException(status_code=400, detail="Invalid input data")

    except HTTPException:
        raise

    except Exception as e:
        app.state.metrics["error_count"] += 1  # ✅ track errors
        logger.error(
            f"request_id={request_id} inference_error={str(e)}"
        )
        logger.error(f"Columns received: {request.data}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )