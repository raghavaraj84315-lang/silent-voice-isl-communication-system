"""Grammar restructuring and inflection mapping for sign token sequences."""
import os
import re
from typing import Dict, Optional, Tuple

import requests

PLACE_NAMES = {
    "hospital": {"en": "hospital", "hi": "अस्पताल", "kn": "ಆಸ್ಪತ್ರೆ"},
    "school": {"en": "school", "hi": "स्कूल", "kn": "ಶಾಲೆ"},
    "home": {"en": "home", "hi": "घर", "kn": "ಮನೆ"},
    "market": {"en": "market", "hi": "बाज़ार", "kn": "ಮಾರುಕಟ್ಟೆ"},
}

PERSON_NAMES = {
    "mother": {"en": "mother", "hi": "माँ", "kn": "ಅಮ್ಮ"},
    "father": {"en": "father", "hi": "पिताजी", "kn": "ತಂದೆ"},
    "brother": {"en": "brother", "hi": "भाई", "kn": "ಸಹೋದರ"},
    "sister": {"en": "sister", "hi": "बहन", "kn": "ಸಹೋದರಿ"},
    "friend": {"en": "friend", "hi": "दोस्त", "kn": "ಸ್ನೇಹಿತ"},
    "teacher": {"en": "teacher", "hi": "शिक्षक", "kn": "ಶಿಕ್ಷಕರು"},
    "student": {"en": "student", "hi": "छात्र", "kn": "ವಿದ್ಯಾರ್ಥಿ"},
    "doctor": {"en": "doctor", "hi": "डॉक्टर", "kn": "ವೈದ್ಯರು"},
}

FOOD_NAMES = {
    "water": {"en": "water", "hi": "पानी", "kn": "ನೀರು"},
    "food": {"en": "food", "hi": "खाना", "kn": "ಊಟ"},
    "tea": {"en": "tea", "hi": "चाय", "kn": "ಚಹಾ"},
}

# Pre-defined high-frequency ISL colloquial patterns and their natural translations
IDIOMATIC_PATTERNS = [
    # Requests & Need
    (r"^(?:please\s+)?water\s+drink(?:\s+please)?$",
     "Please give me some water to drink.",
     "कृपया मुझे पीने के लिए पानी दें।",
     "ದಯವಿಟ್ಟು ನನಗೆ ಕುಡಿಯಲು ನೀರು ಕೊಡಿ."),
    (r"^(?:please\s+)?food\s+eat(?:\s+please)?$",
     "Please give me something to eat.",
     "कृपया मुझे कुछ खाने को दें।",
     "ದಯವಿಟ್ಟು ನನಗೆ ತಿನ್ನಲು ಏನಾದರೂ ಕೊಡಿ."),
    (r"^(?:please\s+)?tea(?:\s+drink)?(?:\s+please)?$",
     "Could I please have a cup of tea?",
     "क्या मुझे एक कप चाय मिल सकती है?",
     "ದಯವಿಟ್ಟು ನನಗೆ ಒಂದು ಕಪ್ ಚಹಾ ಸಿಗಬಹುದೇ?"),
    (r"^(?:help\s+please|please\s+help)$",
     "Please help me.",
     "कृपया मेरी मदद करें।",
     "ದಯವಿಟ್ಟು ನನಗೆ ಸಹಾಯ ಮಾಡಿ."),
    (r"^(?:doctor\s+help|help\s+doctor|doctor\s+please)$",
     "Please call a doctor immediately, I need medical help.",
     "कृपया तुरंत डॉक्टर को बुलाएँ, मुझे चिकित्सा सहायता चाहिए।",
     "ದಯವಿಟ್ಟು ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಕರೆಯಿರಿ, ನನಗೆ ವೈದ್ಯಕೀಯ ಸಹಾಯ ಬೇಕು."),
    (r"^(?:please\s+)?sit(?:\s+please)?$",
     "Please take a seat.",
     "कृपया बैठ जाइए।",
     "ದಯವಿಟ್ಟು ಕುಳಿತುಕೊಳ್ಳಿ."),
    (r"^(?:please\s+)?stand(?:\s+please)?$",
     "Please stand up.",
     "कृपया खड़े हो जाइए।",
     "ದಯವಿಟ್ಟು ಎದ್ದು ನಿಲ್ಲಿ."),

    # Questions (ISL often puts question words at the end)
    (r"^hospital\s+where$|^where\s+hospital$",
     "Where is the nearest hospital?",
     "निकटतम अस्पताल कहाँ है?",
     "ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆ ಎಲ್ಲಿದೆ?"),
    (r"^school\s+where$|^where\s+school$",
     "Where is the school located?",
     "स्कूल कहाँ स्थित है?",
     "ಶಾಲೆ ಎಲ್ಲಿದೆ?"),
    (r"^home\s+where$|^where\s+home$",
     "Where is your home?",
     "आपका घर कहाँ है?",
     "ನಿಮ್ಮ ಮನೆ ಎಲ್ಲಿದೆ?"),
    (r"^market\s+where$|^where\s+market$",
     "Where is the market?",
     "बाज़ार कहाँ है?",
     "ಮಾರುಕಟ್ಟೆ ಎಲ್ಲಿದೆ?"),
    (r"^(?:you\s+how|how\s+you|how\s+are\s+you)$",
     "How are you doing today?",
     "आप आज कैसे हैं?",
     "ಇಂದು ನೀವು ಹೇಗಿದ್ದೀರಿ?"),
    (r"^(?:what\s+name|name\s+what|you\s+name\s+what)$",
     "What is your name?",
     "आपका नाम क्या है?",
     "ನಿಮ್ಮ ಹೆಸರು ಏನು?"),
    (r"^(?:what\s+time|time\s+what)$",
     "What time is it right now?",
     "अभी क्या समय हुआ है?",
     "ಈಗ ಸಮಯ ಎಷ್ಟು?"),
    (r"^(?:come\s+when|when\s+come|you\s+come\s+when)$",
     "When are you coming?",
     "आप कब आ रहे हैं?",
     "ನೀವು ಯಾವಾಗ ಬರುತ್ತೀರಿ?"),
    (r"^(?:go\s+when|when\s+go)$",
     "When are you leaving?",
     "आप कब जा रहे हैं?",
     "ನೀವು ಯಾವಾಗ ಹೋಗುತ್ತೀರಿ?"),
    (r"^(?:what\s+today|today\s+what)$",
     "What is happening today?",
     "आज क्या कार्यक्रम है?",
     "ಇವತ್ತು ಏನು ವಿಶೇಷ?"),

    # Movements & Intentions
    (r"^(?:me\s+)?home\s+go$|^go\s+home$",
     "I am going home now.",
     "मैं अब घर जा रहा हूँ।",
     "ನಾನು ಈಗ ಮನೆಗೆ ಹೋಗುತ್ತಿದ್ದೇನೆ."),
    (r"^(?:me\s+)?school\s+go$|^go\s+school$",
     "I am going to school.",
     "मैं स्कूल जा रहा हूँ।",
     "ನಾನು ಶಾಲೆಗೆ ಹೋಗುತ್ತಿದ್ದೇನೆ."),
    (r"^(?:me\s+)?hospital\s+go$|^go\s+hospital$",
     "I need to go to the hospital.",
     "मुझे अस्पताल जाना है।",
     "ನಾನು ಆಸ್ಪತ್ರೆಗೆ ಹೋಗಬೇಕು."),
    (r"^(?:come\s+home|home\s+come)$",
     "Please come over to my house.",
     "कृपया मेरे घर आइए।",
     "ದಯವಿಟ್ಟು ನಮ್ಮ ಮನೆಗೆ ಬನ್ನಿ."),

    # Greetings & Courtesies
    (r"^hello$",
     "Hello! Nice to meet you.",
     "नमस्ते! आपसे मिलकर अच्छा लगा।",
     "ನಮಸ್ಕಾರ! ನಿಮ್ಮನ್ನು ಭೇಟಿಯಾಗಿದ್ದು ಸಂತೋಷ."),
    (r"^good\s+morning$",
     "Good morning! Have a wonderful day.",
     "शुभ प्रभात! आपका दिन शुभ हो।",
     "ಶುಭೋದಯ! ನಿಮ್ಮ ದಿನ ಸುಖಕರವಾಗಿರಲಿ."),
    (r"^goodbye$",
     "Goodbye! Take care and see you soon.",
     "अलविदा! अपना ध्यान रखें और फिर मिलेंगे।",
     "ವಿದಾಯ! ಜಾಗರೂಕರಾಗಿರಿ ಮತ್ತು ಮತ್ತೆ ಸಿಗೋಣ."),
    (r"^thank\s+you$",
     "Thank you very much!",
     "आपका बहुत-बहुत धन्यवाद!",
     "ತುಂಬಾ ಧನ್ಯವಾದಗಳು!"),
    (r"^sorry$",
     "I am really sorry about that.",
     "मुझे इसके लिए बहुत खेद है।",
     "ನನ್ನಿಂದ ತಪ್ಪಾಗಿದೆ, ದಯವಿಟ್ಟು ಕ್ಷಮಿಸಿ."),
    (r"^okay$",
     "Everything is alright and agreed.",
     "सब कुछ ठीक है।",
     "ಎಲ್ಲವೂ ಸರಿಯಾಗಿದೆ."),
    (r"^yes$",
     "Yes, absolutely.",
     "हाँ, बिल्कुल।",
     "ಹೌದು, ಖಂಡಿತ."),
    (r"^no$",
     "No, thank you.",
     "नहीं, धन्यवाद।",
     "ಇಲ್ಲ, ಧನ್ಯವಾದಗಳು."),
    (r"^stop$",
     "Please stop right here.",
     "कृपया यहीं रुकिए।",
     "ದಯವಿಟ್ಟು ಇಲ್ಲೇ ನಿಲ್ಲಿ."),

    # Family & Relations
    (r"^(?:mother|father)\s+home$",
     "Parents are at home.",
     "माता-पिता घर पर हैं।",
     "ಪೋಷಕರು ಮನೆಯಲ್ಲಿದ್ದಾರೆ."),
    (r"^mother\s+home$",
     "Mother is at home.",
     "माँ घर पर हैं।",
     "ಅಮ್ಮ ಮನೆಯಲ್ಲಿದ್ದಾರೆ."),
    (r"^father\s+home$",
     "Father is at home.",
     "पिताजी घर पर हैं।",
     "ತಂದೆ ಮನೆಯಲ್ಲಿದ್ದಾರೆ."),
    (r"^(?:he|she)\s+friend$|^friend\s+(?:he|she)$",
     "They are my good friend.",
     "वह मेरे अच्छे दोस्त हैं।",
     "ಅವರು ನನ್ನ ಆತ್ಮೀಯ ಸ್ನೇಹಿತರು."),
    (r"^teacher\s+school$",
     "The teacher is at the school.",
     "शिक्षक स्कूल में मौजूद हैं।",
     "ಶಿಕ್ಷಕರು ಶಾಲೆಯಲ್ಲಿದ್ದಾರೆ."),
]


def _call_gemini_api(raw_text: str, api_key: str) -> Optional[Dict[str, str]]:
    """Optional external LLM call if user provided GEMINI_API_KEY."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        prompt = (
            "You are an Indian Sign Language (ISL) translator. Convert the following sequence of sign tokens "
            "into a natural, grammatically fluent spoken sentence in English, and provide accurate natural translations "
            "in Hindi and Kannada. Return ONLY a valid JSON object with keys: english, hindi, kannada.\n\n"
            f"Sign tokens: \"{raw_text}\""
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }
        res = requests.post(url, json=payload, timeout=4)
        if res.status_code == 200:
            import json
            data = res.json()
            candidate = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(candidate)
            if "english" in parsed:
                return {
                    "english": parsed.get("english", "").strip(),
                    "hindi": parsed.get("hindi", "").strip(),
                    "kannada": parsed.get("kannada", "").strip()
                }
    except Exception as e:
        print(f"[Smoother] Gemini API call skipped: {e}")
    return None


def smooth_sentence(raw_text: str, translations_table: Optional[Dict] = None) -> Dict[str, str]:
    """Smooths raw sign tokens into natural English, Hindi, and Kannada.

    Returns:
        {
            "raw": str,
            "smoothed": str,
            "translations": {
                "english": str,
                "hindi": str,
                "kannada": str
            },
            "source": "rules" | "gemini"
        }
    """
    clean_raw = re.sub(r"\s+", " ", raw_text.strip().lower())
    if not clean_raw:
        return {
            "raw": "",
            "smoothed": "",
            "translations": {"english": "", "hindi": "", "kannada": ""},
            "source": "empty"
        }

    # 1. Try Gemini API if key is set
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        api_res = _call_gemini_api(clean_raw, gemini_key)
        if api_res and api_res.get("english"):
            return {
                "raw": raw_text,
                "smoothed": api_res["english"],
                "translations": api_res,
                "source": "gemini"
            }

    # 2. Match idiomatic patterns
    for pattern, en_smooth, hi_smooth, kn_smooth in IDIOMATIC_PATTERNS:
        if re.search(pattern, clean_raw):
            return {
                "raw": raw_text,
                "smoothed": en_smooth,
                "translations": {
                    "english": en_smooth,
                    "hindi": hi_smooth,
                    "kannada": kn_smooth
                },
                "source": "rules"
            }

    # 3. Dynamic Linguistic Smoothing for General Token Combinations
    words = clean_raw.split()

    # Dynamic Location Question: "<place> where" or "where <place>"
    places_found = [w for w in words if w in PLACE_NAMES]
    if "where" in words and places_found:
        place = places_found[0]
        meta = PLACE_NAMES[place]
        en = f"Where is the {meta['en']}?"
        hi = f"{meta['hi']} कहाँ है?"
        kn = f"{meta['kn']} ಎಲ್ಲಿದೆ?"
        return {
            "raw": raw_text,
            "smoothed": en,
            "translations": {"english": en, "hindi": hi, "kannada": kn},
            "source": "rules"
        }

    # Dynamic Needs & Requests: "<food> drink/eat please"
    foods_found = [w for w in words if w in FOOD_NAMES]
    if foods_found and ("drink" in words or "eat" in words or "please" in words):
        item = foods_found[0]
        meta = FOOD_NAMES[item]
        action = "drink" if item in ("water", "tea") else "eat"
        en = f"Please give me some {meta['en']} to {action}."
        hi = f"कृपया मुझे {meta['hi']} दें।"
        kn = f"ದಯವಿಟ್ಟು ನನಗೆ {meta['kn']} ಕೊಡಿ."
        return {
            "raw": raw_text,
            "smoothed": en,
            "translations": {"english": en, "hindi": hi, "kannada": kn},
            "source": "rules"
        }

    # Dynamic Person Location: "<person> <place>"
    persons_found = [w for w in words if w in PERSON_NAMES]
    if persons_found and places_found:
        p_meta = PERSON_NAMES[persons_found[0]]
        l_meta = PLACE_NAMES[places_found[0]]
        en = f"My {p_meta['en']} is currently at the {l_meta['en']}."
        hi = f"{p_meta['hi']} {l_meta['hi']} में हैं।"
        kn = f"{p_meta['kn']} {l_meta['kn']}ದಲ್ಲಿದ್ದಾರೆ."
        return {
            "raw": raw_text,
            "smoothed": en,
            "translations": {"english": en, "hindi": hi, "kannada": kn},
            "source": "rules"
        }

    # 4. General fallback: assemble capitalized fluent sentence with punctuation
    # Capitalize first letter, ensure sentence ends with a period or question mark
    tokens = clean_raw.split()
    capitalized = " ".join(tokens)
    if capitalized:
        capitalized = capitalized[0].upper() + capitalized[1:]

    # Add question mark if starts or contains question words
    if any(q in tokens for q in ("what", "where", "when", "how", "why")):
        capitalized = capitalized.rstrip(".") + "?"
    else:
        capitalized = capitalized.rstrip(".") + "."

    # Standard translations fallback from table
    hi_trans = ""
    kn_trans = ""
    if translations_table:
        from translate import translate
        hi_trans = translate(clean_raw, "hindi", translations_table)
        kn_trans = translate(clean_raw, "kannada", translations_table)

    return {
        "raw": raw_text,
        "smoothed": capitalized,
        "translations": {
            "english": capitalized,
            "hindi": hi_trans or capitalized,
            "kannada": kn_trans or capitalized
        },
        "source": "fallback"
    }
