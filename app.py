import logging
import time
import uuid

from fastapi import FastAPI, Request

from inference import ModelService

app = FastAPI()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


setup_logging()
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time

    logger.info(
        f"request_id={request_id} "
        f"method={request.method} "
        f"path={request.url.path} "
        f"status={response.status_code} "
        f"latency={latency:.4f}s"
    )

    return response

@app.on_event("startup")
def startup_event():
    logger.info("Starting FastAPI application")

    app.state.model_service = ModelService("models/credit_model_v1.pkl")

    logger.info("Model loaded successfully")


@app.get("/health")
def health_check():
    return {"status": "ok"}