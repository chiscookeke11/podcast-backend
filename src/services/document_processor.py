"""
document_processor.py
Ingests raw source content (plain text, PDF, DOCX) and produces a
clean summary suitable for injection into Claude host prompts.
"""

from __future__ import annotations
import re
import anthropic
from pathlib import Path
from config import settings

# ── Text extraction ───────────────────────────────────────────────────────────


def extract_text_from_file(path: Path) -> str:
    """Extract plain text from PDF, DOCX, or plain text files."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix in (".docx", ".doc"):
        return _extract_docx(path)
    else:
        return path.read_text(encoding="utf-8", errors="replace")


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ── Text cleaning ─────────────────────────────────────────────────────────────


def clean_text(raw: str) -> str:
    """Normalise whitespace, remove junk characters."""
    text = re.sub(r"\r\n|\r", "\n", raw)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", "", text)
    return text.strip()


# ── Summarisation via Claude ──────────────────────────────────────────────────

_SUMMARY_SYSTEM = """
You are a research assistant preparing a briefing for two podcast hosts.
Extract the most interesting, debatable, and surprising content from the document.
Return a structured briefing in plain text with these sections:

CORE ARGUMENT
What is the document's main claim or thesis? (2-3 sentences)

KEY POINTS
3-5 specific claims, findings, or ideas worth discussing on air.

TENSIONS & DEBATES
What is controversial, counterintuitive, or likely to spark disagreement?

MEMORABLE DETAILS
Specific numbers, quotes, examples, or stories that will make good podcast moments.

Be concrete. No generic summaries. Hosts will use this to have a live argument.
""".strip()


async def summarise_document(text: str, max_chars: int = 80_000) -> str:
    """
    Use Claude to produce a rich, opinionated briefing from the source text.
    Truncates very long documents before sending.
    """
    truncated = text[:max_chars]
    if len(text) > max_chars:
        truncated += f"\n\n[Document truncated at {max_chars} characters]"

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    message = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1500,
        system=_SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": f"Document to brief:\n\n{truncated}"}],
    )

    return message.content[0].text


# ── Main entry point ──────────────────────────────────────────────────────────


async def process_document(
    source: str | Path,
    is_file: bool = False,
) -> tuple[str, str]:
    """
    Full pipeline: extract → clean → summarise.

    Returns (clean_full_text, summary).
    The summary goes into host prompts; the full text is stored for reference.
    """
    if is_file:
        raw = extract_text_from_file(Path(source))
    else:
        raw = str(source)

    clean = clean_text(raw)
    summary = await summarise_document(clean)

    return clean, summary
