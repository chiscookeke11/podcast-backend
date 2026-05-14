"""
script_generator.py
Orchestrates Claude API calls to generate a full podcast transcript.
Two hosts alternate turns; Claude plays each one with a distinct system prompt.
Full transcript is generated before any audio synthesis begins.
"""

from __future__ import annotations
import re
import anthropic
import structlog
from config import settings
from app.models import HostConfig, Transcript, TranscriptLine, Tone
from services.prompt_builder import (
    build_system_prompt,
    build_intro_prompt,
    build_outro_prompt,
)

log = structlog.get_logger()


# ── Expression tag parsing ────────────────────────────────────────────────────

_TONE_PREFIX = re.compile(
    r"^\[(angry|serious|amused|hushed|taunting|default)\]\s*",
    re.IGNORECASE,
)

_AMBIENT_KEYWORDS: dict[str, list[str]] = {
    "water": ["water", "throat"],
    "whiskey": ["whiskey", "sip", "something stronger", "drink"],
}

_ELEVENLABS_TAGS = {
    "<laugh>",
    "<chuckle>",
    "<sigh>",
    "<cough>",
    "<throat-clearing>",
    "<gasp>",
    "<hmm>",
}


def _parse_raw_line(
    speaker: str,
    raw: str,
    line_index: int,
) -> TranscriptLine:
    """Extract tone prefix, detect ambient actions, normalise tag spacing."""
    raw = raw.strip()

    # Extract tone prefix
    tone = Tone.DEFAULT
    m = _TONE_PREFIX.match(raw)
    if m:
        tone = Tone(m.group(1).lower())
        raw = raw[m.end() :]

    # Normalise ElevenLabs tags — ensure trailing space
    for tag in _ELEVENLABS_TAGS:
        raw = raw.replace(tag, f"{tag} ")
    raw = re.sub(r"  +", " ", raw).strip()

    # Detect ambient actions for UI hints
    ambient = None
    lower = raw.lower()
    for action, keywords in _AMBIENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            ambient = action
            break

    return TranscriptLine(
        line_index=line_index,
        speaker=speaker,
        text=raw,
        tone=tone,
        ambient_action=ambient,
    )


# ── Single Claude turn ────────────────────────────────────────────────────────


async def _call_claude(
    system_prompt: str,
    conversation_history: list[dict],
    max_tokens: int = 300,
) -> str:
    """Make one Claude API call and return the text response."""
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    response = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=conversation_history,
    )
    return response.content[0].text.strip()


# ── Full transcript generation ────────────────────────────────────────────────


async def generate_transcript(
    host_a: HostConfig,
    host_b: HostConfig,
    source_summary: str,
    num_turns: int = 8,
    episode_title: str = "",
) -> Transcript:
    """
    Generate the full podcast transcript using Claude.

    Structure:
      - Opening line from host_a
      - num_turns alternating exchanges (host_b goes first in the loop)
      - Closing line from host_b

    Each host has a dedicated system prompt; they share a conversation history
    so each sees what the other said.
    """
    log.info(
        "generating_transcript",
        host_a=host_a.name,
        host_b=host_b.name,
        num_turns=num_turns,
    )

    lines: list[TranscriptLine] = []
    # Shared conversation history — both Claude calls read this
    history: list[dict] = []

    # ── Opening line (host_a) ─────────────────────────────────────────────────
    intro_prompt = build_intro_prompt(
        host_a, host_b.name, source_summary, episode_title
    )
    opening_text = await _call_claude(
        system_prompt=intro_prompt,
        conversation_history=[{"role": "user", "content": "Start the episode."}],
    )

    opening_line = _parse_raw_line(host_a.name, opening_text, line_index=0)
    lines.append(opening_line)

    # Seed history so host_b can respond to the opening
    history.append({"role": "user", "content": f"{host_a.name} said: {opening_text}"})

    log.debug("opening_line", speaker=host_a.name, text=opening_text[:80])

    # ── Alternating turns ─────────────────────────────────────────────────────
    # Turn 0 = host_b responds to opening
    # Turn 1 = host_a responds to host_b
    # ...and so on

    for turn in range(num_turns):
        is_host_b_turn = turn % 2 == 0
        current_host = host_b if is_host_b_turn else host_a
        other_host = host_a if is_host_b_turn else host_b

        system_prompt = build_system_prompt(
            current_host, other_host.name, source_summary, episode_title
        )
        prompt_text = (
            f"Respond to {other_host.name}'s last point. "
            "2-4 sentences. Stay in character."
        )
        history.append({"role": "user", "content": prompt_text})

        reply_text = await _call_claude(
            system_prompt=system_prompt,
            conversation_history=history,
        )

        history.append({"role": "assistant", "content": reply_text})

        line = _parse_raw_line(
            current_host.name,
            reply_text,
            line_index=len(lines),
        )
        lines.append(line)

        log.debug(
            "turn_complete",
            turn=turn,
            speaker=current_host.name,
            text=reply_text[:80],
        )

    # ── Closing line (host_b) ─────────────────────────────────────────────────
    outro_prompt = build_outro_prompt(
        host_b, host_a.name, source_summary, episode_title
    )
    history.append(
        {"role": "user", "content": "Close the episode with a final thought."}
    )

    closing_text = await _call_claude(
        system_prompt=outro_prompt,
        conversation_history=history,
        max_tokens=200,
    )

    closing_line = _parse_raw_line(host_b.name, closing_text, line_index=len(lines))
    lines.append(closing_line)

    log.info(
        "transcript_complete",
        total_lines=len(lines),
        host_a_lines=sum(1 for l in lines if l.speaker == host_a.name),
        host_b_lines=sum(1 for l in lines if l.speaker == host_b.name),
    )

    return Transcript(
        lines=lines,
        host_a_name=host_a.name,
        host_b_name=host_b.name,
        source_title=episode_title,
    )
