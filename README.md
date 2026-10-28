
# AgriSmart — AI-Powered Agriculture Assistant

AgriSmart is an agriculture assistance platform designed to combine
machine learning, deep learning, and generative AI.

## Planned Features

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

Start the development server:

```bash
python -m uvicorn app.main:app --reload
```

## API Documentation

Visit http://127.0.0.1:8000/docs

## Run Tests

```bash
python -m pytest -v
```

## Project Status

The backend foundation and initial API endpoints are implemented.
The database and agriculture modules will be developed in subsequent phases.
