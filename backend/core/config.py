# backend/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    debug_mode: bool = True
    camera_id: int = 0
    openai_api_key: str
    fps_target: int = 5
    window_size_sec: int = 5

    class Config:
        env_file = ".env"

# Projenin her yerinden bu settings objesini çağıracağız
settings = Settings()