# AgriSmart

AgriSmart is a FastAPI agriculture assistant with yield prediction, crop disease
classification, Gemini-powered guidance, MySQL crop records, and a small browser
interface for farmers.

## Project contents

- `app/` contains the API, configuration, database, model, schema, and service code.
- `app/web/` contains the browser interface served at `/ui/`.
- `Dockerfile` and `compose.yaml` start the API and MySQL locally.
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

### Docker Compose

Install Docker Desktop and Git LFS, then from the repository root:

```bash
git lfs install
git lfs pull
cp .env.example .env
```

Edit `.env`: replace the local database passwords and add your `GEMINI_API_KEY`.
Then build and start the API and MySQL database:

```bash
docker compose up --build
```

Compose waits for MySQL, creates the crop-record table, and then starts the API.
Open the farmer interface at <http://127.0.0.1:8000/ui/> or the API docs at
<http://127.0.0.1:8000/docs>. The database data persists in a Docker volume.
The app port is bound to localhost for local development.

### Run Python directly

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
The farmer interface is available at <http://127.0.0.1:8000/ui/>.

In production, set a strong `API_ACCESS_TOKEN`; versioned API endpoints require
it in the `X-API-Key` header. Set `CORS_ORIGINS` to a comma-separated allowlist
for browser clients. Image uploads default to a 10 MiB limit; configure
`MAX_IMAGE_UPLOAD_BYTES` to change it.

The sample passwords in `.env.example` are for local development only. Replace
them before using the Docker setup beyond your own machine.

## Tests

Install development dependencies and run the suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -v
```

API tests use in-memory SQLite, and Gemini and model inference calls use stubs.
The suite needs no MySQL server, Gemini key, or trained TensorFlow models.
GitHub Actions runs the same suite on pushes and pull requests to `main` without
fetching the large Git LFS artifacts.

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
