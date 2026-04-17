from pydantic import BaseModel, Field


class TopProbability(BaseModel):
    label: str
    value: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    isUnknown: bool
    threshold: float = Field(ge=0.0, le=1.0)
    topProbabilities: list[TopProbability]
