# PlantVillage FastAPI Backend

This backend serves image inference for your PlantVillage CNN checkpoint.

## 1) Setup

```bash
cd api
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2) Configure environment

Copy `.env.example` to `.env` and adjust values if needed.

```bash
copy .env.example .env
```

Important settings:

- `MODEL_PATH`: path to `.pth` checkpoint
- `CONFIDENCE_THRESHOLD`: unknown-class threshold (default `0.5`)
- `TOP_K`: number of top predictions to return
- `ALLOWED_ORIGINS`: frontend origins for CORS

## 3) Run API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /health`
- `POST /predict` with `multipart/form-data`, field name: `file`

### Sample response

```json
{
  "label": "Potato___Late_blight",
  "confidence": 0.9231,
  "isUnknown": false,
  "threshold": 0.5,
  "topProbabilities": [
    { "label": "Potato___Late_blight", "value": 0.9231 },
    { "label": "Potato___Early_blight", "value": 0.0524 },
    { "label": "Potato___healthy", "value": 0.0139 }
  ]
}
```

## Frontend integration

From your Next.js frontend, send `FormData` to:

- `http://localhost:8000/predict`

and map directly to your existing UI model fields (`label`, `confidence`, `isUnknown`, `topProbabilities`).
