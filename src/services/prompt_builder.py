"""
prompt_builder.py
Assembles Claude system prompts from HostConfig objects.
Separated from script_generator so prompts are easy to tweak and test in isolation.
"""

from app.models import HostConfig, CharacterConfig

# ── Expression instructions injected into every host prompt ──────────────────

_EXPRESSION_GUIDE = """
NATURAL EXPRESSIONS — inject these tags directly into your spoken text:
  <laugh>            genuine laugh
  <chuckle>          dry, quiet amusement
  <sigh>             frustration, resignation, or heavy emphasis
  <cough>            brief interruption or comic timing
  <throat-clearing>  before making a major point
  <gasp>             surprise or disbelief
  <hmm>              skepticism or thinking out loud

Use sparingly — one or two per episode feels real, every line feels performed.

TONE PREFIXES — start a line with one of these to adjust vocal delivery.
Do NOT speak these words aloud; they are metadata the system strips before synthesis:
  [angry]     forceful, raised energy
  [serious]   slow and deliberate, quiet authority
  [amused]    loose and light, enjoying the moment
  [hushed]    quiet intensity, like sharing a hard truth
  [taunting]  playful provocation

Example:
  [angry] <sigh> Nobody who's ever built anything thinks like that.

AMBIENT MOMENTS — use at most once per episode, never forced:
  <cough> Pass me that water. Anyway —
  Let me take a sip before I say what I actually think.
  <throat-clearing> Right. Where were we.
"""

# ── Format rules applied to every host ───────────────────────────────────────

_FORMAT_RULES = """
RESPONSE RULES
- Speak only your line. No labels, no stage directions, no "(laughs)".
- Keep every turn to 2–4 sentences maximum.
- Refer to your co-host by name when you agree, challenge, or hand off a point.
- Never break character. Never explain what you're doing.
- Do not start with your own name.
"""


def build_system_prompt(
    host: HostConfig,
    other_host_name: str,
    source_summary: str,
    episode_title: str = "",
) -> str:
    """
    Build the full Claude system prompt for one host.

    host              — the host whose turn it is
    other_host_name   — used so this host can reference their co-host by name
    source_summary    — compressed version of the source document
    episode_title     — optional, included if provided
    """
    c: CharacterConfig = host.character

    title_line = f'TODAY\'S EPISODE: "{episode_title}"\n' if episode_title else ""

    prompt = f"""You are {host.name}, a podcast host.

{title_line}YOUR CO-HOST: {other_host_name}
Refer to {other_host_name} by name when you push back, agree, or hand off a point.

─── WHO YOU ARE ────────────────────────────────────────────────────────────────

WORLDVIEW
{c.worldview.strip()}

SPEAKING STYLE
{c.speaking_style.strip()}

HOW YOU DEBATE
{c.debate_style.strip()}
"""

    if c.forbidden.strip():
        prompt += f"""
WHAT YOU NEVER DO
{c.forbidden.strip()}
"""

    prompt += f"""
─── TODAY'S TOPIC ──────────────────────────────────────────────────────────────

{source_summary.strip()}

─── EXPRESSION GUIDE ───────────────────────────────────────────────────────────
{_EXPRESSION_GUIDE}
{_FORMAT_RULES}"""

    return prompt.strip()


def build_intro_prompt(
    host: HostConfig,
    other_host_name: str,
    source_summary: str,
    episode_title: str = "",
) -> str:
    """
    Specialised prompt for the opening line of the episode.
    The first host sets the tone for the whole conversation.
    """
    base = build_system_prompt(host, other_host_name, source_summary, episode_title)
    return (
        base + "\n\nThis is your OPENING LINE. In 2 sentences, introduce the topic "
        "with energy and give the audience a reason to keep listening. "
        "Reference the topic directly — don't do generic podcast small talk."
    )


def build_outro_prompt(
    host: HostConfig,
    other_host_name: str,
    source_summary: str,
    episode_title: str = "",
) -> str:
    """
    Specialised prompt for the closing line of the episode.
    """
    base = build_system_prompt(host, other_host_name, source_summary, episode_title)
    return (
        base + "\n\nThis is your CLOSING LINE. In 2 sentences, land the episode with "
        "a memorable final thought that's true to your character. "
        "Don't say 'thanks for listening' or generic podcast signoffs."
    )
