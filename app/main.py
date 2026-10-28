
from fastapi import FastAPI

app = FastAPI(
    title="AgriSmart API",
    description="AI-powered agriculture assistant",
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
