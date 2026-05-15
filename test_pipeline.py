"""
test_pipeline.py
Run the full episode pipeline from the terminal — no server needed.
Useful for validating API keys and tuning host characters before building the frontend.

Usage:
    python test_pipeline.py
"""

import asyncio
import json
from app.models import EpisodeRequest, HostConfig, VoiceConfig, CharacterConfig
from pipeline.ochestrator import run_episode_pipeline

# ── Sample hosts ──────────────────────────────────────────────────────────────

NOVA = HostConfig(
    name="Nova",
    voice=VoiceConfig(
        voice_id="21m00Tcm4TlvDq8ikWAM",  # replace with your ElevenLabs voice ID
        stability=0.40,
        similarity_boost=0.75,
        style_exaggeration=0.50,
        speaking_rate=1.05,
    ),
    character=CharacterConfig(
        worldview=(
            "You believe ideas are the most powerful force in the world. "
            "You're endlessly curious and find surprising angles in everything. "
            "You're optimistic but not naive."
        ),
        speaking_style=(
            "Enthusiastic, quick. Rhetorical questions you then answer yourself. "
            "Uses 'okay but—' to pivot. Short bursts of energy followed by a genuine pause."
        ),
        debate_style=(
            "Finds the unexpected angle and runs with it. "
            "Genuinely excited when Titan pushes back — treats it as a puzzle to solve."
        ),
        forbidden="Never sounds defeated. Never hedges without immediately recovering.",
    ),
)

TITAN = HostConfig(
    name="Titan",
    voice=VoiceConfig(
        voice_id="DaHSjSULwPSikisUTxBf",  # replace with your ElevenLabs voice ID
        stability=0.30,
        similarity_boost=0.80,
        style_exaggeration=0.65,
        speaking_rate=1.10,
    ),
    character=CharacterConfig(
        worldview=(
            "You believe most people are mentally weak and comfort-seeking. "
            "Discipline, competition, and real-world consequences are the only metrics that matter. "
            "You respect competence above all else."
        ),
        speaking_style=(
            "Short declarative sentences. You never trail off. "
            "Opens points with 'Listen' or 'Brother'. "
            "Speaks in absolutes: 'nobody', 'always', 'the top 1%'. "
            "Never says 'I think' or 'maybe' — you assert."
        ),
        debate_style=(
            "Pushes back hard on anything that sounds like an excuse. "
            "Finds the competitive or hierarchical angle in any topic. "
            "Occasionally admits when Nova makes a strong point — but always adds a harder layer."
        ),
        forbidden=(
            "Never hedge or qualify excessively. "
            "Never sound sympathetic to failure without pivoting to what should have been done differently."
        ),
    ),
    tone_overrides={
        "angry": {"stability": 0.15, "style": 0.90},
        "hushed": {"stability": 0.80, "style": 0.05},
    },
)

# ── Sample source text ────────────────────────────────────────────────────────

SOURCE_TEXT = """
The 10,000-Hour Rule, popularised by Malcolm Gladwell in Outliers, claims that
mastery in any field requires roughly 10,000 hours of deliberate practice.
The idea swept through popular culture and became gospel in productivity circles.

However, the original researcher behind the concept, Anders Ericsson, repeatedly
stated that Gladwell misrepresented his findings. Ericsson's research focused
specifically on deliberate practice — structured, effortful, feedback-driven work —
not mere repetition. Playing a video game for 10,000 hours does not make you a master.

More recent studies have found that practice accounts for only about 12% of
performance differences in music, 18% in sports, and even less in domains like chess.
Genetics, starting age, and coaching quality all play significant roles that the
10,000-hour rule ignores.

The debate matters because millions of people have structured their lives around
the belief that effort alone determines success. If that's wrong, what does it mean
for how we raise children, build schools, and think about talent?
"""


# ── Run ───────────────────────────────────────────────────────────────────────


async def main():
    print("\n=== AI PODCAST PIPELINE TEST ===\n")
    print(f"Hosts: {NOVA.name} vs {TITAN.name}")
    print(f"Source: {len(SOURCE_TEXT)} characters\n")

    request = EpisodeRequest(
        host_a=NOVA,
        host_b=TITAN,
        source_text=SOURCE_TEXT,
        num_turns=6,  # keep short for testing
        episode_title="The 10,000-Hour Lie",
    )

    print("Running pipeline (document → transcript → audio)...\n")
    episode = await run_episode_pipeline(request)

    print(f"\nStatus: {episode.status}")

    if episode.status == "ready":
        print(f"\n── TRANSCRIPT ({'='*40})")
        for line in episode.transcript.lines:
            tone_tag = f"[{line.tone}] " if line.tone != "default" else ""
            print(f"\n{line.speaker} {tone_tag}(line {line.line_index}):")
            print(f"  {line.text}")
            if line.ambient_action:
                print(f"  ↳ ambient: {line.ambient_action}")

        print(f"\n── AUDIO FILES ({len(episode.audio_files)} files) {'='*30}")
        for path in episode.audio_files:
            print(f"  {path}")
    else:
        print(f"\nError: {episode.error}")


if __name__ == "__main__":
    import structlog

    structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
    asyncio.run(main())
