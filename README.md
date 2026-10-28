
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
```

Copy `.env.example` to `.env`, then set `DB_PASSWORD` and
`GEMINI_API_KEY` for the services you plan to use. The application reads
the database settings as `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and
`DB_PASSWORD`.

Start the development server:

```bash
python -m uvicorn app.main:app --reload
```

## API Documentation

Visit http://127.0.0.1:8000/docs

## Machine Learning Models

Training data and generated model files are kept out of Git. Place the crop
yield CSV in `data/raw/`, and place tomato and soybean images under
`data/raw/tomato_disease/` and `data/raw/soybean_disease/`, with one folder
per class. Run the corresponding scripts in `scripts/` to create the model
files in `artifacts/`. Prediction endpoints return an unavailable response
until their model files have been generated.

## Run Tests

```bash
python -m pytest -v
```

## Project Status

The FastAPI backend, database layer, model training scripts, prediction
endpoints, and Gemini-powered guidance endpoints are implemented. Configure
the required services and generate the local model artifacts before using
those features.
