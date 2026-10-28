# AgriSmart

AgriSmart is a FastAPI agriculture assistant with yield prediction, crop disease
classification, Gemini-powered guidance, and MySQL crop records.

## Project contents

- `app/` contains the API, configuration, database, model, schema, and service code.
- `scripts/` contains dataset inspection and model training commands.
- `tests/` contains the API and service tests.
- `data/` contains the yield and disease training datasets.
- `artifacts/` contains the trained models and evaluation reports.

Datasets and model binaries are stored with Git LFS. GitHub ZIP/tar archives
contain LFS pointer files by default; clone the repository and run `git lfs
pull` to retrieve the assets. A repository admin can enable LFS objects in
archives in **Settings → General → Archives**; archive downloads then count
against Git LFS bandwidth.

## Run locally

Use Python 3.12, create a virtual environment, and install the runtime packages:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
```

Set `DATABASE_URL` and `GEMINI_API_KEY` in `.env`. Create the MySQL database and
user first, then initialize its tables and start the API:

```bash
python -m app.db.init_db
python -m uvicorn app.main:app --reload
```

Open the API documentation at <http://127.0.0.1:8000/docs>.

In production, set a strong `API_ACCESS_TOKEN`; versioned API endpoints require
it in the `X-API-Key` header. Set `CORS_ORIGINS` to a comma-separated allowlist
for browser clients. Image uploads default to a 10 MiB limit; configure
`MAX_IMAGE_UPLOAD_BYTES` to change it.

## Tests

Install development dependencies and run the suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -v
```

The tests mock database and external API calls; model inference tests use small
stubs and do not need a Gemini key or trained TensorFlow models.

## Retraining

Run commands from the repository root. The yield trainer uses the prepared CSVs;
the disease trainers read one subdirectory per class and save only the final
model plus class labels and evaluation metrics under `artifacts/`.

```bash
python scripts/prepare_yield_data.py
python scripts/train_yield_model.py
python scripts/train_tomato_model.py
python scripts/train_soybean_model.py
```

The soybean dataset is small, and its held-out test set contains only 15 images
per class. Treat its reported scores as preliminary and collect more labeled
images before relying on it for farming decisions. Yield metrics are reported
overall and by crop because crop units and outliers can differ. In the current
yield data, Coconut values are far larger than most other crop values; confirm
the source units before treating its predictions or the overall MAE as
representative.
