# PlantVillage Disease Classification Project

## Group Project Overview

This project was developed as a deep learning coursework submission for plant disease classification using leaf images.

Our goal was to:

- Train a CNN-based model for PlantVillage disease detection
- Save the trained model for inference
- Build a user-friendly web interface where anyone can upload a leaf image and get a prediction
- Add a practical unknown-class rule: if highest confidence is below 50%, we treat the image as outside known classes

The work is organized in this repository.

## Kaggle Notebook and Dataset Work

The full training notebook, experiments, and analysis are available on Kaggle:

- Kaggle notebook: https://www.kaggle.com/code/emmanueladoum/plant-village-disease-classification-99-32

This notebook includes:

- Data preparation and transformations
- Model training and validation
- Saved checkpoint export
- Evaluation plots and metrics

## Repository Structure

- frontend: Next.js app for image upload and prediction display
- api: FastAPI backend for model inference
- model: Saved notebook and trained model checkpoint

## What We Built

### 1) Model Training

We trained a deep learning model on PlantVillage images and saved the checkpoint for later inference.

Saved artifacts include:

- Jupyter notebook with complete workflow
- Trained model file in the model folder

### 2) Backend API (FastAPI)

The backend handles:

- Loading the trained checkpoint at startup
- Receiving uploaded images
- Running preprocessing and inference
- Returning top predictions and confidence scores
- Applying unknown-class logic using threshold = 0.5

Main endpoints:

- GET /health
- POST /predict

### 3) Frontend Interface (Next.js)

The frontend supports:

- Drag and drop or browse image upload
- Preview of selected image
- One-click classification
- Confidence visualization for top predictions
- Displaying unknown class if confidence is lower than threshold

## Unknown-Class Assumption

From our experiments, we observed that out-of-distribution images often received a top confidence below 50%.

Based on that observation, we applied this rule in the system:

- If top confidence is less than 0.5, report as unknown or out of known classes

This is a practical project assumption and can be improved in future work with stronger OOD detection methods.

## Local Setup Guide

## Prerequisites

- Python 3.10+ (or compatible with installed torch)
- Node.js 18+
- npm

## Backend Setup

1. Open terminal in api folder
2. Create and activate virtual environment
3. Install requirements
4. Copy env template
5. Run server

Example commands:

Windows cmd:

- python -m venv .venv
- .venv\Scripts\activate
- pip install -r requirements.txt
- copy .env.example .env
- uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

## Frontend Setup

1. Open terminal in frontend folder
2. Install dependencies
3. Set API base url
4. Run dev server

Example commands:

- npm install
- copy .env.local.example .env.local
- npm run dev

Default frontend URL:

- http://localhost:3000

Default backend URL:

- http://localhost:8000

## Testing the App

1. Start backend server
2. Start frontend server
3. Open frontend in browser
4. Upload a leaf image
5. Click classify
6. Read predicted class and confidence

Expected behavior:

- Known class: normal prediction panel
- Unknown class: shown when top confidence is below threshold

## Notes for Grading

This repository is prepared for grading from GitHub and includes all major deliverables:

- Training notebook reference
- Trained model file
- Inference backend
- User-facing frontend
- Threshold-based unknown handling

Kaggle notebook link is provided above for reproducibility and verification of training workflow.

## Limitations and Future Improvements

Current limitations:

- Threshold rule is heuristic and not a formal OOD detector
- Model confidence may still be high on some unseen samples
- No authentication or rate-limiting in API

Future improvements:

- Calibrate probabilities (temperature scaling)
- Add uncertainty estimation and OOD-specific methods
- Add batch inference and logging dashboard
- Containerize deployment (Docker) for easier hosting

## Contributors

This was completed as a student group project submission for deep learning coursework.
