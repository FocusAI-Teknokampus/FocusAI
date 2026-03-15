# backend/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    debug_mode: bool = True
    camera_id: int = 0
    openai_api_key: str
    fps_target: int = 5
    window_size_sec: int = 5

    # Pydantic'e tanımadığı .env değişkenlerini görmezden gelmesini (ignore) söylüyoruz
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()