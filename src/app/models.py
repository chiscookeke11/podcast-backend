# All Pydantic data models for the podcast backend


from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from uuid import UUID, uuid4
from enum import Enum

# --------------------------------------------- Voice * Character ----------------------------------------------


class VoiceConfig(BaseModel):
    """ElevenLabs voice parameters for one host."""

    voice_id: str = Field(..., description="ElevenLabs voice ID")
    stability: float = Field(0.45, ge=0.0, le=1.0)
    similarity_boost: float = Field(0.75, ge=0.0, le=1.0)
    style_exaggeration: float = Field(0.40, ge=0.0, le=1.0)
    speaking_rate: float = Field(1.0, ge=0.5, le=2.0)
    model_id: str = "eleven_turbo_v2_5"


class CharacterConfig(BaseModel):
    """
    Claude persona definition for one host.
    Everything here becomes part of the system prompt.
    """

    worldview: str = Field(
        ...,
        description="Core beliefs and how they see the world",
        examples=[
            "You believe discipline and competition are the foundation of everything."
        ],
    )
    speaking_style: str = Field(
        ...,
        description="How they talk - sentence length, rhythm, verbal tics",
        examples=[
            "short declarative sentences. Never hedge. Open points with 'Listen' or 'Brother'."
        ],
    )

    debate_style: str = Field(
        ...,
        description="How they engage with the other host",
        examples=[
            "Pushes back hard on weak thinking. Finds the competitive angle in any topic."
        ],
    )
    forbidden: str = Field(
        default="",
        description="Things this host never says or does",
        examples=[
            "Never says 'I think' or 'maybe'. Never sounds sympathetic to failure."
        ],
    )


class HostConfig(BaseModel):
    """Complete definition of one podcast host — voice + character."""

    name: str = Field(..., min_length=1, max_length=50)
    voice: VoiceConfig
    character: CharacterConfig
    tone_overrides: dict[str, dict] = Field(
        default_factory=dict,
        description="Per-tone voice setting overrides, keyed by tone name (angry, serious, etc.)",
    )


# ── Transcript ────────────────────────────────────────────────────────────────


class Tone(str, Enum):
    DEFAULT = "default"
    ANGRY = "angry"
    SERIOUS = "serious"
    AMUSED = "amused"
    HUSHED = "hushed"
    TAUNTING = "taunting"


class TranscriptLine(BaseModel):
    """one spoken turn in the episode."""

    line_index: int
    speaker: str  # matches HostConfig.name
    text: str  # may contain ElevenLabs expression tags
    tone: Tone = Tone.DEFAULT
    ambient_action: str | None = None  # "water", "whiskey" — for UI hints


class Transcript(BaseModel):
    lines: list[TranscriptLine]
    host_a_name: str
    host_b_name: str
    source_title: str = ""

    @property
    def total_turns(self) -> int:
        return len(self.lines)


# ── Episode request / response ────────────────────────────────────────────────
class EpisodeRequest(BaseModel):
    """
    Everything the user submits to kick off an episode.
    Source content comes as a separate file upload or inline text.
    """

    host_a: HostConfig
    host_b: HostConfig
    source_text: str = Field(
        ..., min_length=100, description="The document/article the hosts will discuss"
    )
    num_turns: int = Field(
        default=8,
        ge=2,
        le=20,
        description="Number of back-and-forth exchanges (not counting intro/outro)",
    )
    episode_title: str = Field(default="", description="Optional episode title")

    @field_validator("host_a", "host_b")
    @classmethod
    def names_must_differ(cls, v: HostConfig, info) -> HostConfig:
        # Cross-field validation runs in model_validator instead
        return v

    def model_post_init(self, __context) -> None:
        if self.host_a.name.lower() == self.host_b.name.lower():
            raise ValueError("host_a and host_b must have different names")


class EpisodeStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"  # Claude writing the script
    SYNTHESISING = "synthesising"  # ElevenLabs rendering audio
    READY = "ready"
    FAILED = "failed"


class Episode(BaseModel):
    """Full episode — returned to the client when ready."""

    id: UUID = Field(default_factory=uuid4)
    status: EpisodeStatus = EpisodeStatus.PENDING
    title: str = ""
    transcript: Transcript | None = None
    audio_files: list[str] = Field(
        default_factory=list,
        description="Ordered list of audio file URLs/paths, one per transcript line",
    )
    error: str | None = None


# ── Audio synthesis ───────────────────────────────────────────────────────────


class SynthesisResult(BaseModel):
    """Result of synthesising one transcript line."""

    line_index: int
    speaker: str
    audio_path: str  # path relative to audio_output_dir
    duration_ms: int | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


class SynthesisJob(BaseModel):
    """Tracks synthesis progress for a full episode."""

    episode_id: UUID
    total_lines: int
    completed: int = 0
    results: list[SynthesisResult] = Field(default_factory=list)

    @property
    def progress_pct(self) -> float:
        if self.total_lines == 0:
            return 0.0
        return round(self.completed / self.total_lines * 100, 1)

    @property
    def all_done(self) -> bool:
        return self.completed >= self.total_lines
