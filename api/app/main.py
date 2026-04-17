from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.model_service import ModelService
from app.schemas import PredictionResponse, TopProbability

model_service = ModelService(
    checkpoint_path=settings.model_path,
    img_size=settings.img_size,
    threshold=settings.confidence_threshold,
    top_k=settings.top_k,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title="PlantVillage Inference API",
    description="FastAPI backend for plant disease image classification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    max_bytes = 10 * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    try:
        result = model_service.predict(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictionResponse(
        label=result.label,
        confidence=result.confidence,
        isUnknown=result.is_unknown,
        threshold=settings.confidence_threshold,
        topProbabilities=[TopProbability(**item) for item in result.top_probabilities],
    )
