"""
routers/episodes.py
FastAPI route handlers for episode generation and retrieval.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
from uuid import UUID

from app.models import EpisodeRequest, Episode, EpisodeStatus
from pipeline.ochestrator import run_episode_pipeline
from config import settings

router = APIRouter(prefix="/episodes", tags=["episodes"])

# In-memory store — replace with Redis or a DB for production
_episodes: dict[str, Episode] = {}


@router.post("/generate", response_model=Episode, status_code=202)
async def generate_episode(
    request: EpisodeRequest,
    background_tasks: BackgroundTasks,
):
    """
    Kick off episode generation.
    Returns immediately with a pending Episode (id + status=pending).
    Generation runs in the background; poll GET /episodes/{id} for status.
    """
    # Create a placeholder episode right away so the client has an ID to poll
    episode = Episode(
        title=request.episode_title,
        status=EpisodeStatus.PENDING,
    )
    _episodes[str(episode.id)] = episode

    async def _run():
        result = await run_episode_pipeline(request)
        # Merge result back — preserve the same ID
        result.id = episode.id
        _episodes[str(episode.id)] = result

    background_tasks.add_task(_run)

    return episode


@router.post("/generate/sync", response_model=Episode)
async def generate_episode_sync(request: EpisodeRequest):
    """
    Synchronous version — waits for the full pipeline and returns the result.
    Useful for testing and short episodes. May time out on long episodes.
    """
    episode = await run_episode_pipeline(request)
    _episodes[str(episode.id)] = episode
    return episode


@router.get("/{episode_id}", response_model=Episode)
async def get_episode(episode_id: UUID):
    """Poll for episode status and retrieve the completed episode."""
    ep = _episodes.get(str(episode_id))
    if not ep:
        raise HTTPException(status_code=404, detail="Episode not found")
    return ep


@router.get("/{episode_id}/audio/{filename}")
async def serve_audio(episode_id: UUID, filename: str):
    """Serve a synthesised audio file for playback."""
    audio_path = settings.audio_output_dir / str(episode_id) / filename

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Basic path traversal guard
    try:
        audio_path.resolve().relative_to(settings.audio_output_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")

    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        filename=filename,
    )
