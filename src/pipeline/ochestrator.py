"""
pipeline/orchestrator.py
End-to-end coordinator: source document → transcript → synthesised audio.
This is the single entry point the FastAPI router calls.
"""

from __future__ import annotations
import uuid
import structlog
from pathlib import Path
from config import settings
from app.models import (
    EpisodeRequest,
    Episode,
    EpisodeStatus,
    Transcript,
    SynthesisJob,
)
from services.document_processor import process_document
from services.script_generator import generate_transcript
from services.tts_service import synthesise_transcript

log = structlog.get_logger()


async def run_episode_pipeline(request: EpisodeRequest) -> Episode:
    """
    Full pipeline — runs synchronously through three stages:

      1. Document processing  — clean + summarise the source text
      2. Script generation    — Claude writes the full transcript
      3. Audio synthesis      — ElevenLabs renders each line to MP3

    Returns a completed Episode with transcript and audio_files populated.
    On any failure, returns an Episode with status=FAILED and error set.
    """
    episode_id = uuid.uuid4()
    episode = Episode(id=episode_id, title=request.episode_title)

    log.info(
        "pipeline_start",
        episode_id=str(episode_id),
        host_a=request.host_a.name,
        host_b=request.host_b.name,
        source_chars=len(request.source_text),
    )

    # ── Stage 1: Document processing ─────────────────────────────────────────
    try:
        episode.status = EpisodeStatus.GENERATING
        log.info("stage_document_processing", episode_id=str(episode_id))

        _clean_text, summary = await process_document(
            request.source_text, is_file=False
        )

        log.info(
            "document_processed",
            episode_id=str(episode_id),
            summary_chars=len(summary),
        )

    except Exception as e:
        log.error("document_processing_failed", error=str(e))
        episode.status = EpisodeStatus.FAILED
        episode.error = f"Document processing failed: {e}"
        return episode

    # ── Stage 2: Script generation ────────────────────────────────────────────
    try:
        log.info("stage_script_generation", episode_id=str(episode_id))

        transcript: Transcript = await generate_transcript(
            host_a=request.host_a,
            host_b=request.host_b,
            source_summary=summary,
            num_turns=request.num_turns,
            episode_title=request.episode_title,
        )
        episode.transcript = transcript

        log.info(
            "script_generated",
            episode_id=str(episode_id),
            total_lines=transcript.total_turns,
        )

    except Exception as e:
        log.error("script_generation_failed", error=str(e))
        episode.status = EpisodeStatus.FAILED
        episode.error = f"Script generation failed: {e}"
        return episode

    # ── Stage 3: Audio synthesis ──────────────────────────────────────────────
    try:
        episode.status = EpisodeStatus.SYNTHESISING
        log.info("stage_audio_synthesis", episode_id=str(episode_id))

        output_dir = settings.audio_output_dir / str(episode_id)
        hosts_by_name = {
            request.host_a.name: request.host_a,
            request.host_b.name: request.host_b,
        }

        job = SynthesisJob(
            episode_id=episode_id,
            total_lines=transcript.total_turns,
        )

        completed_job = await synthesise_transcript(
            transcript=transcript,
            hosts=hosts_by_name,
            output_dir=output_dir,
            job=job,
        )

        # Collect ordered audio file paths (skip failed lines gracefully)
        episode.audio_files = [r.audio_path for r in completed_job.results if r.success]

        log.info(
            "audio_synthesised",
            episode_id=str(episode_id),
            audio_files=len(episode.audio_files),
        )

    except Exception as e:
        log.error("audio_synthesis_failed", error=str(e))
        episode.status = EpisodeStatus.FAILED
        episode.error = f"Audio synthesis failed: {e}"
        return episode

    # ── Done ──────────────────────────────────────────────────────────────────
    episode.status = EpisodeStatus.READY
    log.info(
        "pipeline_complete",
        episode_id=str(episode_id),
        status=episode.status,
        lines=transcript.total_turns,
        audio_files=len(episode.audio_files),
    )

    return episode
