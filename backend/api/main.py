# backend/api/main.py
#
# FastAPI uygulamasının giriş noktası.
# Tüm router'lar buraya bağlanır.
# uvicorn bu dosyayı çalıştırır:
#   uvicorn backend.api.main:app --reload

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.api.routers import session, chat, dashboard, upload

from backend.core.database import init_db




app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI Destekli Kişisel Öğrenme Mentörü — REST API",
)

init_db()

# ── CORS ──────────────────────────────────────────────────────────────────────
# Frontend (Vite dev server) farklı port'tan istek attığı için CORS gerekli.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Router'ları bağla ─────────────────────────────────────────────────────────
app.include_router(session.router,   prefix="/session",   tags=["Session"])
app.include_router(chat.router,      prefix="/chat",       tags=["Chat"])
app.include_router(dashboard.router, prefix="/dashboard",  tags=["Dashboard"])
app.include_router(upload.router, prefix="/upload", tags=["Upload"])


# ── Sağlık kontrolü ───────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """Servisin ayakta olup olmadığını kontrol eder."""
    return {"status": "ok", "version": settings.app_version}
