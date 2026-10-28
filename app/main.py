
from fastapi import FastAPI

app = FastAPI(
    title="AgriSmart API",
    description=(
        "An AI-powered agriculture assistant "
        "for crop predictions and farming guidance."
    ),
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "Welcome to AgriSmart",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }
