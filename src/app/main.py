from fastapi import FastAPI
import structlog
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from config import Settings
from app.routers.episodes import router as episodes_router

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Settings.audio_output_dir.mkdir(parents=True, exist_ok=True)
    log.info("startup", audio_dir=str(Settings.audio_output_dir), env=Settings.app_env)
    yield
    log.info("shutdown")


app = FastAPI(title="AI Podcast API", version="0.1.0", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(episodes_router)


@app.get("/health")
async def health():
    return {"status": "ok", "env": Settings.app_env}
