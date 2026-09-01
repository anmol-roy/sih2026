"""
Legal Disclaimer  (Phase 9)
────────────────────────────
Hard-coded disclaimer added by the backend to every answer.
The LLM never generates the disclaimer — it is always appended here.

This prevents the LLM from accidentally omitting, weakening,
or rewording the disclaimer.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# English disclaimer (canonical)
# ─────────────────────────────────────────────────────────────────────────────

DISCLAIMER_EN = (
    "Information provided by IP-SAKTI Sahayak is for general informational "
    "purposes only and does not constitute legal advice or create an "
    "attorney-client relationship. "
    "For legal decisions, filing, prosecution, compliance, or other "
    "professional matters, consult a qualified IP professional."
)

# ─────────────────────────────────────────────────────────────────────────────
# Translated disclaimers (hard-coded — never LLM-generated)
# ─────────────────────────────────────────────────────────────────────────────

DISCLAIMER_HI = (
    "IP-SAKTI Sahayak द्वारा प्रदान की गई जानकारी केवल सामान्य सूचना के उद्देश्य से है "
    "और कानूनी सलाह नहीं है। "
    "कानूनी निर्णयों, फाइलिंग, अनुपालन या अन्य पेशेवर मामलों के लिए "
    "योग्य IP पेशेवर से परामर्श करें।"
)

DISCLAIMER_KN = (
    "IP-SAKTI Sahayak ಒದಗಿಸಿದ ಮಾಹಿತಿಯು ಸಾಮಾನ್ಯ ಮಾಹಿತಿ ಉದ್ದೇಶಗಳಿಗಾಗಿ ಮಾತ್ರ ಮತ್ತು "
    "ಕಾನೂನು ಸಲಹೆಯಲ್ಲ. "
    "ಕಾನೂನು ನಿರ್ಧಾರಗಳು, ಫೈಲಿಂಗ್, ಅನುಸರಣೆ ಅಥವಾ ಇತರ ವೃತ್ತಿಪರ ವಿಷಯಗಳಿಗೆ "
    "ಅರ್ಹ IP ವೃತ್ತಿಪರರನ್ನು ಸಂಪರ್ಕಿಸಿ."
)

DISCLAIMERS: dict[str, str] = {
    "en": DISCLAIMER_EN,
    "hi": DISCLAIMER_HI,
    "kn": DISCLAIMER_KN,
}


def get_disclaimer(language: str = "en") -> str:
    """Return the disclaimer for the given language code."""
    return DISCLAIMERS.get(language.lower(), DISCLAIMER_EN)


def add_disclaimer(answer: str, language: str = "en") -> str:
    """Append the disclaimer to *answer* — always called by the backend."""
    disclaimer = get_disclaimer(language)
    return f"{answer}\n\n---\n{disclaimer}"


def strip_disclaimer(text: str) -> str:
    """
    Remove appended disclaimer from text (for tests / re-translation).
    Strips everything after the last `---\n` separator.
    """
    sep = "\n\n---\n"
    if sep in text:
        return text[:text.rfind(sep)]
    return text
