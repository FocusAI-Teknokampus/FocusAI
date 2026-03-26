# backend/api/main.py
#
# FastAPI uygulamasinin giris noktasi.
# Tum router'lar buraya baglanir.
# uvicorn bu dosyayi calistirir:
#   uvicorn backend.api.main:app --reload

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers import (
    analytics,
    camera,
    chat,
    dashboard,
    feedback,
    history,
    session,
    upload,
    welcome,
)
from backend.core.config import settings
from backend.core.database import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI Destekli Kisisel Ogrenme Mentoru - REST API",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(session.router, prefix="/session", tags=["Session"])
    app.include_router(camera.router, prefix="/camera", tags=["Camera"])
    app.include_router(chat.router, prefix="/chat", tags=["Chat"])
    app.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])
    app.include_router(upload.router, prefix="/upload", tags=["Upload"])
    app.include_router(history.router, tags=["History"])
    app.include_router(analytics.router, tags=["Analytics"])
    app.include_router(welcome.router, tags=["Welcome"])
    app.include_router(feedback.router, tags=["Feedback"])

    @app.get("/health", tags=["System"])
    def health_check() -> dict[str, str]:
        return {"status": "ok", "version": settings.app_version}

    return app


app = create_app()
