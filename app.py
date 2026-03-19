import logging
import time
import uuid

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import pandas as pd

from inference import ModelService

app = FastAPI()


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
        logger.error(f"request_id={request_id} unhandled_error={str(e)}")
        raise

    latency = time.time() - start_time

    logger.info(
        f"request_id={request_id} "
        f"method={request.method} "
        f"path={request.url.path} "
        f"status={response.status_code} "
        f"latency={latency:.4f}s"
    )

    return response


# -----------------------------------
# STARTUP (SAFE MODEL LOADING)
# -----------------------------------

@app.on_event("startup")
def startup_event():
    logger.info("Starting FastAPI application")

    app.state.model_service = ModelService("models/credit_model_v1.pkl")

    logger.info("Model loaded successfully")


# -----------------------------------
# REQUEST SCHEMA
# -----------------------------------

class PredictionRequest(BaseModel):
    data: list[dict]


# -----------------------------------
# INPUT ADAPTER (CRITICAL FIX)
# -----------------------------------

def build_full_dataframe(input_data, model_service):
    df = pd.DataFrame(input_data)

    expected_cols = model_service.pipeline.feature_names_in_

    missing_cols = []

    # Add missing columns
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
            missing_cols.append(col)

    # Log missing columns (observability)
    if missing_cols:
        logger.warning(
            f"Missing columns auto-filled (showing first 10): {missing_cols[:10]}"
        )

    # Reorder columns to match training schema
    df = df[expected_cols]

    return df


# -----------------------------------
# HEALTH ENDPOINT
# -----------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


# -----------------------------------
# PREDICTION ENDPOINT
# -----------------------------------

@app.post("/api/v1/predict")
def predict(request: PredictionRequest, req: Request):

    request_id = req.state.request_id

    try:
        model_service = app.state.model_service

        #FIX: Use schema adapter instead of raw DataFrame
        df = build_full_dataframe(request.data, model_service)

        if df.empty:
            raise HTTPException(status_code=400, detail="Empty input data")

        # Run inference
        results = model_service.predict_with_risk(df)

        logger.info(
            f"request_id={request_id} "
            f"inference_success rows={len(df)}"
        )

        return {
            "request_id": request_id,
            "predictions": results
        }

    except ValueError as e:
        logger.error(
            f"request_id={request_id} validation_error={str(e)}"
        )
        raise HTTPException(status_code=400, detail="Invalid input data")

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            f"request_id={request_id} inference_error={str(e)}"
        )
        logger.error(f"Columns received: {request.data}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )