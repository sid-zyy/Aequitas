"""The citizen portal's emergency assistant. Rule-based on purpose: it asks whether the person needs help now,
and either shows the official helpline numbers or explains how to file a normal report.

No LLM writes these replies, so the assistant can never invent a phone number or claim help is on the way.
Every number shown comes from HELPLINES below."""

from __future__ import annotations

from typing import Literal

Lang = Literal["en", "hi", "kn"]

# India's national numbers. 112 is the single emergency number (ERSS); the others still work.
HELPLINES = [
    ("112", {"en": "Emergency: police, fire, ambulance", "hi": "आपातकाल: पुलिस, अग्निशमन, एम्बुलेंस",
             "kn": "ತುರ್ತು: ಪೊಲೀಸ್, ಅಗ್ನಿಶಾಮಕ, ಆಂಬ್ಯುಲೆನ್ಸ್"}),
    ("100", {"en": "Police", "hi": "पुलिस", "kn": "ಪೊಲೀಸ್"}),
    ("101", {"en": "Fire", "hi": "अग्निशमन (फ़ायर ब्रिगेड)", "kn": "ಅಗ್ನಿಶಾಮಕ ದಳ"}),
    ("108", {"en": "Ambulance", "hi": "एम्बुलेंस", "kn": "ಆಂಬ್ಯುಲೆನ್ಸ್"}),
    ("1091", {"en": "Women's helpline", "hi": "महिला हेल्पलाइन", "kn": "ಮಹಿಳಾ ಸಹಾಯವಾಣಿ"}),
]

TEXT = {
    "greeting": {
        "en": "Hello. Do you need emergency help right now?",
        "hi": "नमस्ते। क्या आपको अभी आपातकालीन मदद चाहिए?",
        "kn": "ನಮಸ್ಕಾರ. ನಿಮಗೆ ಈಗಲೇ ತುರ್ತು ಸಹಾಯ ಬೇಕೇ?",
    },
    "emergency": {
        "en": "If you or someone else is in danger right now, call 112 immediately. This chat and the report "
              "form cannot send help. If you can, move to a safe place. When you are safe, you can file a "
              "report below: it will be marked urgent for officers.",
        "hi": "अगर आप या कोई और अभी खतरे में है, तो तुरंत 112 पर कॉल करें। यह चैट और रिपोर्ट फ़ॉर्म मदद नहीं "
              "भेज सकते। हो सके तो किसी सुरक्षित जगह पर चले जाएँ। सुरक्षित होने पर आप नीचे रिपोर्ट दर्ज कर "
              "सकते हैं: अधिकारियों के लिए उसे अत्यावश्यक चिह्नित किया जाएगा।",
        "kn": "ನೀವು ಅಥವಾ ಬೇರೆ ಯಾರಾದರೂ ಈಗ ಅಪಾಯದಲ್ಲಿದ್ದರೆ, ತಕ್ಷಣ 112 ಗೆ ಕರೆ ಮಾಡಿ. ಈ ಚಾಟ್ ಮತ್ತು ವರದಿ ಫಾರ್ಮ್ "
              "ಸಹಾಯವನ್ನು ಕಳುಹಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ. ಸಾಧ್ಯವಾದರೆ ಸುರಕ್ಷಿತ ಸ್ಥಳಕ್ಕೆ ಹೋಗಿ. ನೀವು ಸುರಕ್ಷಿತವಾದ ನಂತರ ಕೆಳಗೆ "
              "ವರದಿ ಸಲ್ಲಿಸಬಹುದು: ಅಧಿಕಾರಿಗಳಿಗೆ ಅದನ್ನು ತುರ್ತು ಎಂದು ಗುರುತಿಸಲಾಗುತ್ತದೆ.",
    },
    "report": {
        "en": "Okay. You can file a report with the form on this page: add the video (the original file from "
              "your phone), say what happened, and where and when if you know. You will get a report ID and a "
              "tracking code to check the status. If it becomes an emergency, call 112.",
        "hi": "ठीक है। आप इस पेज के फ़ॉर्म से रिपोर्ट दर्ज कर सकते हैं: वीडियो जोड़ें (अपने फ़ोन की मूल फ़ाइल), "
              "बताएँ कि क्या हुआ, और अगर पता हो तो कहाँ और कब। स्थिति देखने के लिए आपको एक रिपोर्ट आईडी और "
              "ट्रैकिंग कोड मिलेगा। अगर आपात स्थिति बन जाए, तो 112 पर कॉल करें।",
        "kn": "ಸರಿ. ಈ ಪುಟದ ಫಾರ್ಮ್ ಬಳಸಿ ನೀವು ವರದಿ ಸಲ್ಲಿಸಬಹುದು: ವೀಡಿಯೊ ಸೇರಿಸಿ (ನಿಮ್ಮ ಫೋನ್‌ನ ಮೂಲ ಫೈಲ್), "
              "ಏನಾಯಿತು ಎಂದು ತಿಳಿಸಿ, ಮತ್ತು ಗೊತ್ತಿದ್ದರೆ ಎಲ್ಲಿ ಮತ್ತು ಯಾವಾಗ ಎಂದು ಬರೆಯಿರಿ. ಸ್ಥಿತಿಯನ್ನು ಪರಿಶೀಲಿಸಲು "
              "ನಿಮಗೆ ವರದಿ ಐಡಿ ಮತ್ತು ಟ್ರ್ಯಾಕಿಂಗ್ ಕೋಡ್ ಸಿಗುತ್ತದೆ. ತುರ್ತು ಪರಿಸ್ಥಿತಿ ಎದುರಾದರೆ 112 ಗೆ ಕರೆ ಮಾಡಿ.",
    },
    "unclear": {
        "en": "Sorry, I did not understand. Do you need emergency help right now? Please answer yes or no.",
        "hi": "माफ़ कीजिए, मैं समझ नहीं पाया। क्या आपको अभी आपातकालीन मदद चाहिए? कृपया हाँ या नहीं में जवाब दें।",
        "kn": "ಕ್ಷಮಿಸಿ, ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ. ನಿಮಗೆ ಈಗಲೇ ತುರ್ತು ಸಹಾಯ ಬೇಕೇ? ದಯವಿಟ್ಟು ಹೌದು ಅಥವಾ ಇಲ್ಲ ಎಂದು ಉತ್ತರಿಸಿ.",
    },
}
CHOICES = {"en": ("Yes", "No"), "hi": ("हाँ", "नहीं"), "kn": ("ಹೌದು", "ಇಲ್ಲ")}

YES = {"yes", "y", "yeah", "yep", "haan", "han", "ha", "haa", "houdu",
       "हाँ", "हां", "हा", "जी", "ಹೌದು", "ಹೌದ", "ಹು"}
NO = {"no", "n", "nope", "nah", "nahi", "nahin", "na", "illa", "नहीं", "नही", "ना", "ಇಲ್ಲ", "ಬೇಡ"}
# Words that mean "help now" in any of the three languages (and common Latin-letter Hindi/Kannada).
DANGER = (
    "help", "emergency", "danger", "urgent", "attack", "fire", "accident", "bleeding", "injured", "hurt",
    "kidnap", "weapon", "knife", "gun", "unconscious", "sos", "following me", "chasing",
    "bachao", "madad", "khatra", "aag", "kaapaadi", "kapadi", "sahaya",
    "मदद", "बचाओ", "बचाइए", "आपातकाल", "खतरा", "खतरे", "आग", "दुर्घटना", "हादसा", "घायल", "खून", "हमला",
    "चाकू", "बंदूक", "अपहरण",
    "ಸಹಾಯ", "ಕಾಪಾಡಿ", "ತುರ್ತು", "ಅಪಾಯ", "ಬೆಂಕಿ", "ಅಪಘಾತ", "ಗಾಯ", "ರಕ್ತ", "ಹಲ್ಲೆ", "ಚಾಕು", "ಅಪಹರಣ",
)
_PUNCT = ".,!?।॥;:'\"()-"


def _lang(lang: str) -> str:
    return lang if lang in CHOICES else "en"


def intent(message: str) -> Literal["yes", "no", "unclear"]:
    """yes = needs help now. A plain yes/no word decides first (so "no, I don't need help" is a no),
    then any danger word anywhere means yes."""
    text = message.strip().lower()
    words = [w.strip(_PUNCT) for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return "unclear"
    if words[0] in YES:
        return "yes"
    if words[0] in NO:
        return "no"
    if any(d in text for d in DANGER):
        return "yes"
    if any(w in NO for w in words):
        return "no"
    if any(w in YES for w in words):
        return "yes"
    return "unclear"


def helplines(lang: str) -> list[dict]:
    lang = _lang(lang)
    return [{"number": n, "label": labels[lang]} for n, labels in HELPLINES]


def reply(lang: str, message: str | None = None, choice: str | None = None) -> dict:
    """One assistant turn. `choice` ("yes"/"no") is a button press; `message` is typed text.
    Neither: the opening question."""
    lang = _lang(lang)
    yes, no = CHOICES[lang]
    buttons = [{"value": "yes", "label": yes}, {"value": "no", "label": no}]
    if choice is None and not (message or "").strip():
        return {"state": "ask", "reply": TEXT["greeting"][lang], "choices": buttons, "helplines": [], "urgent": False}
    said = choice if choice in ("yes", "no") else intent(message or "")
    if said == "yes":
        return {"state": "emergency", "reply": TEXT["emergency"][lang], "choices": [],
                "helplines": helplines(lang), "urgent": True}
    if said == "no":
        return {"state": "report", "reply": TEXT["report"][lang], "choices": [], "helplines": [], "urgent": False}
    return {"state": "ask", "reply": TEXT["unclear"][lang], "choices": buttons, "helplines": [], "urgent": False}
