"""Citizen reports in Hindi or Kannada: which language a text is in, and an English translation for officers.

The original text is always kept; the translation is extra and comes from the locally served LLM (the same
client and endpoint as the chatbot), timed like every other stage."""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

LANGUAGES = {"en": "English", "hi": "Hindi", "kn": "Kannada"}

DEVANAGARI = range(0x0900, 0x0980)
KANNADA = range(0x0C80, 0x0D00)


def detect_language(text: str, declared: str = "en") -> str:
    """The script decides when there is one: Devanagari -> hi, Kannada -> kn. Latin text keeps the declared
    language (Hindi is often typed in Latin letters), and anything unknown falls back to English."""
    deva = sum(ord(ch) in DEVANAGARI for ch in text)
    kan = sum(ord(ch) in KANNADA for ch in text)
    if deva or kan:
        return "hi" if deva >= kan else "kn"
    return declared if declared in LANGUAGES else "en"


class Translation(BaseModel):
    english: str = Field(max_length=4000)


def translate_prompt(text: str, lang: str) -> str:
    name = LANGUAGES.get(lang, "an Indian language")
    return (
        f"Translate this citizen's police report from {name} into plain English. Translate faithfully: keep "
        "every fact, place name, time and number exactly as written, add nothing, and do not summarise or "
        "comment. If parts are already English, keep them.\n\n"
        f"Report:\n{text}\n\nReturn JSON with the field 'english'."
    )


def translate_to_english(client, text: str, lang: str) -> tuple[str | None, float]:
    """Returns (english or None on failure, ms)."""
    t0 = time.perf_counter()
    ans = client.structured([{"role": "user", "content": translate_prompt(text, lang)}], Translation,
                            max_tokens=min(1024, 64 + 2 * len(text)))
    english = ans.english.strip() if ans is not None and ans.english.strip() else None
    return english, (time.perf_counter() - t0) * 1000
