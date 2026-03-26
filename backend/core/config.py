# backend/core/config.py
#
# Tüm ayarlar buradan okunur.
# .env dosyasından otomatik yüklenir.
# Yeni bir ayar eklenince .env.example'a da eklenmeli.
#
# Sahip: K3

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):

    # ── Uygulama ──────────────────────────────
    debug_mode: bool = True
    app_name: str = "FocusAI"
    app_version: str = "2.0.0"

    # ── LLM ───────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_vision_model: str = "gpt-4o"          # Fotoğraf analizi için
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Kamera / CV Engine ────────────────────
    camera_id: int = 0
    camera_enabled: bool = False                  # Varsayılan kapalı, kullanıcı açar
    fps_target: int = 5
    window_size_sec: int = 5

    # ── State Modeling ────────────────────────
    default_uncertainty_threshold: float = 0.75   # Adaptive threshold başlangıç değeri
    min_confidence_to_intervene: float = 0.75
    intervention_cooldown_seconds: int = 120      # Aynı kullanıcıya max 2 dk'da bir müdahale

    # ── RAG ───────────────────────────────────
    rag_chunk_size: int = 500                     # Her chunk kaç karakter?
    rag_chunk_overlap: int = 50                   # Chunk'lar arası örtüşme
    rag_top_k: int = 3                            # Kaç chunk dönsün?
    faiss_index_path: str = "data/faiss_index"   # Index dosyasının yolu

    # ── Memory ────────────────────────────────
    mem0_api_key: str = ""                        # Mem0 cloud için (boş = local mod)
    short_term_max_messages: int = 20             # Session context'te max kaç mesaj?

    # ── Storage ───────────────────────────────
    database_url: str = "sqlite:///./data/mentor.db"
    data_dir: str = "data"                        # Tüm local dosyalar buraya

    # ── API ───────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]  # Vite dev server

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Settings singleton. Her import'ta yeniden okumaz.
    Kullanım: from backend.core.config import get_settings
              settings = get_settings()
    """
    return Settings()


# Modül seviyesinde erişim için kısayol
settings = get_settings()
