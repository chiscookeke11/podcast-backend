from fastapi import FastAPI
from fastapi.responses import FileResponse
import structlog
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers.episodes import router as episodes_router

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.audio_output_dir.mkdir(parents=True, exist_ok=True)
    log.info("startup", audio_dir=str(settings.audio_output_dir), env=settings.app_env)
    yield
    log.info("shutdown")


app = FastAPI(title="AI Podcast API", version="0.1.0", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(episodes_router)


@app.get("/")
async def frontend():
    return {"message": "Backend is up and running!"}


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.app_env}
