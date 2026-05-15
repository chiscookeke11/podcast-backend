"""
tts_service.py
TTS synthesis supporting both ElevenLabs and OpenAI.
Automatically falls back to OpenAI if ElevenLabs fails.
"""

from __future__ import annotations
import asyncio
import httpx
import structlog
from dataclasses import dataclass
from pathlib import Path
from config import settings
from app.models import (
    HostConfig,
    TranscriptLine,
    Transcript,
    SynthesisResult,
    SynthesisJob,
    Tone,
)

log = structlog.get_logger()


# ── Tone presets ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToneSettings:
    stability: float
    style_exaggeration: float
    speaking_rate: float


TONE_PRESETS: dict[Tone, ToneSettings] = {
    Tone.DEFAULT: ToneSettings(
        stability=0.45, style_exaggeration=0.40, speaking_rate=1.00
    ),
    Tone.ANGRY: ToneSettings(
        stability=0.20, style_exaggeration=0.85, speaking_rate=1.20
    ),
    Tone.SERIOUS: ToneSettings(
        stability=0.70, style_exaggeration=0.20, speaking_rate=0.88
    ),
    Tone.AMUSED: ToneSettings(
        stability=0.30, style_exaggeration=0.60, speaking_rate=1.05
    ),
    Tone.HUSHED: ToneSettings(
        stability=0.75, style_exaggeration=0.10, speaking_rate=0.82
    ),
    Tone.TAUNTING: ToneSettings(
        stability=0.25, style_exaggeration=0.75, speaking_rate=1.10
    ),
}


def _merge_tone(host: HostConfig, tone: Tone) -> dict:
    """
    Merge base host voice settings with tone preset overrides.
    Host-level tone_overrides (from config) take final priority.
    """
    base = host.voice
    preset = TONE_PRESETS.get(tone, TONE_PRESETS[Tone.DEFAULT])

    # Check if the host has a custom override for this tone
    custom = host.tone_overrides.get(tone.value, {})

    return {
        "stability": custom.get("stability", preset.stability),
        "similarity_boost": custom.get("similarity_boost", base.similarity_boost),
        "style": custom.get("style", preset.style_exaggeration),
        "use_speaker_boost": True,
        "speed": custom.get("speed", preset.speaking_rate),
    }


# ── ElevenLabs API call ───────────────────────────────────────────────────────


async def _call_elevenlabs(
    text: str,
    voice_id: str,
    voice_settings: dict,
    model_id: str,
    api_key: str,
) -> bytes:
    """Single ElevenLabs /text-to-speech call. Returns raw MP3 bytes."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": voice_settings,
    }
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.content


# ── OpenAI TTS API call ───────────────────────────────────────────────────────

OPENAI_VOICE_MAP = {
    "Nova": "nova",  # Friendly, warm
    "Titan": "onyx",  # Deep, authoritative
}


async def _call_openai_tts(
    text: str,
    speaker_name: str,
    speaking_rate: float,
    api_key: str,
) -> bytes:
    """OpenAI TTS /audio/speech call. Returns raw MP3 bytes."""
    url = "https://api.openai.com/v1/audio/speech"

    # Map speaker name to OpenAI voice
    voice = OPENAI_VOICE_MAP.get(speaker_name, "alloy")

    payload = {
        "model": "tts-1",  # tts-1 is faster, tts-1-hd for higher quality
        "input": text,
        "voice": voice,
        "speed": speaking_rate,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.content


# ── Synthesise one line ───────────────────────────────────────────────────────


async def synthesise_line(
    line: TranscriptLine,
    host: HostConfig,
    output_dir: Path,
) -> SynthesisResult:
    """
    Synthesise one transcript line to MP3.
    Tries ElevenLabs first, falls back to OpenAI if it fails.
    Writes to {output_dir}/{line_index:03d}_{speaker}.mp3
    Returns a SynthesisResult (success or failure).
    """
    filename = f"{line.line_index:03d}_{line.speaker.lower()}.mp3"
    output_path = output_dir / filename

    # Try ElevenLabs first if configured
    if settings.tts_provider == "elevenlabs" and settings.elevenlabs_api_key:
        try:
            voice_settings = _merge_tone(host, line.tone)
            audio_bytes = await _call_elevenlabs(
                text=line.text,
                voice_id=host.voice.voice_id,
                voice_settings=voice_settings,
                model_id=host.voice.model_id,
                api_key=settings.elevenlabs_api_key,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)

            log.debug(
                "line_synthesised",
                provider="elevenlabs",
                line_index=line.line_index,
                speaker=line.speaker,
                tone=line.tone,
                bytes=len(audio_bytes),
            )

            return SynthesisResult(
                line_index=line.line_index,
                speaker=line.speaker,
                audio_path=str(output_path),
            )

        except httpx.HTTPStatusError as e:
            log.warning(
                "elevenlabs_failed_trying_openai",
                status=e.response.status_code,
                error=e.response.text[:200],
            )

    # OpenAI TTS (fallback or primary)
    try:
        if not settings.openai_api_key:
            return SynthesisResult(
                line_index=line.line_index,
                speaker=line.speaker,
                audio_path="",
                error="No TTS provider configured. Set ELEVENLABS_API_KEY or OPENAI_API_KEY",
            )

        preset = TONE_PRESETS.get(line.tone, TONE_PRESETS[Tone.DEFAULT])
        audio_bytes = await _call_openai_tts(
            text=line.text,
            speaker_name=line.speaker,
            speaking_rate=preset.speaking_rate,
            api_key=settings.openai_api_key,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio_bytes)

        log.debug(
            "line_synthesised",
            provider="openai",
            line_index=line.line_index,
            speaker=line.speaker,
            bytes=len(audio_bytes),
        )

        return SynthesisResult(
            line_index=line.line_index,
            speaker=line.speaker,
            audio_path=filename,  # Store only filename, not full path
        )

    except Exception as e:
        log.error("tts_synthesis_error", error=str(e))
        return SynthesisResult(
            line_index=line.line_index,
            speaker=line.speaker,
            audio_path="",
            error=str(e),
        )


# ── Synthesise full transcript ────────────────────────────────────────────────


async def synthesise_transcript(
    transcript: Transcript,
    hosts: dict[str, HostConfig],  # keyed by host name
    output_dir: Path,
    job: SynthesisJob,
    concurrency: int | None = None,
) -> SynthesisJob:
    """
    Synthesise all lines in a transcript with bounded concurrency.
    Updates job in-place as lines complete.
    Preserves episode order in job.results regardless of completion order.
    """
    concurrency = concurrency or settings.tts_concurrency
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(line: TranscriptLine) -> SynthesisResult:
        host = hosts[line.speaker]
        async with semaphore:
            result = await synthesise_line(line, host, output_dir)
            job.completed += 1
            job.results.append(result)
            log.info(
                "synthesis_progress",
                progress=f"{job.progress_pct}%",
                line=line.line_index,
            )
            return result

    await asyncio.gather(*[_bounded(line) for line in transcript.lines])

    # Sort results back into episode order
    job.results.sort(key=lambda r: r.line_index)

    failures = [r for r in job.results if not r.success]
    if failures:
        log.warning("synthesis_failures", count=len(failures))

    log.info(
        "synthesis_complete",
        total=job.total_lines,
        succeeded=len([r for r in job.results if r.success]),
        failed=len(failures),
    )

    return job
