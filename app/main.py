
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.api.upload_limit import ImageUploadLimitMiddleware
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    description=(
        "An AI-powered agriculture assistant "
        "for crop predictions and farming guidance."
    ),
    version="1.0.0",
)

app.add_middleware(ImageUploadLimitMiddleware)

if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

app.include_router(router)


@app.get("/ui-config", include_in_schema=False)
def ui_config():
    return {
        "api_v1_prefix": "/" + settings.api_v1_prefix.strip("/"),
    }


app.mount(
    "/ui",
    StaticFiles(directory=Path(__file__).resolve().parent / "web", html=True),
    name="web-ui",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
