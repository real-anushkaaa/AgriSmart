
# AgriSmart — AI-Powered Agriculture Assistant

AgriSmart is an agriculture assistance platform designed to combine
machine learning, deep learning, and generative AI.

## Features

- Crop yield prediction
- Crop price prediction and insights
- Tomato leaf disease detection
- Soybean leaf disease detection
- Crop calendar generation
- AI agriculture chatbot

## Technology Stack

- Python
- FastAPI
- MySQL
- SQLAlchemy
- Scikit-learn
- TensorFlow / Keras
- Gemini API

## Run Locally

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
test -f .env || cp .env.example .env
```

Set `DATABASE_URL` and `GEMINI_API_KEY` in `.env`. Create the MySQL database
and user first, then initialize the tables:

```bash
python -m app.db.init_db
```

Start the development server:

```bash
python -m uvicorn app.main:app --reload
```

## API Documentation

Visit http://127.0.0.1:8000/docs

## Datasets and Models

The training datasets and generated models are included through Git LFS. After
cloning, install Git LFS and download the assets with `git lfs pull`. They are
also available locally in `data/` and `artifacts/`.

## Run Tests

```bash
python -m pytest -v
```

## Project Status

The FastAPI backend, database layer, training scripts, prediction endpoints,
and Gemini-powered guidance endpoints are implemented. Dataset and model
assets are included for local use.
