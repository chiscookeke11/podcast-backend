from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # AI
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-20250514"

    # ElevenLabs
    elevenlabs_api_key: str
    elevenlabs_model: str = "eleven_turbo_v2_5"

    # App
    app_env: str = "development"
    audio_output_dir: Path = Path("./audio_cache")
    max_episode_turns: int = 12
    tts_concurrency: int = 3
    cors_origins: list[str] = ["http://localhost:5173"]

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"


Settings = Settings()
