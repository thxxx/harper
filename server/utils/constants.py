COMPANION_NAME = "Harper"
LLM_MODEL = "api" # "api" or "local"
OPENAI_KEY = ""

DEFAULT_VOICE_PATH = "./audio_samples/output_full.wav"
DEFAULT_KOREAN_VOICE_PATH = "./audio_samples/shogun.wav"

FOLLOWUP_SILENCE_DELAY = 6.0

LANGUAGE_CODE = {
    "Arabic": "ar",
    "German": "de",
    "Greek": "el",
    "Danish": "da",
    "English": "en",
    "Spanish": "es",
    "Finnish": "fi",
    "Hindi": "hi",
    "Hebrew": "he",
    "French": "fr",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Chinese": "zh",
    "Dutch": "nl",
    "Norwegian": "no",
    "Polish": "pl",
    "Portuguese": "pt",
    "Swedish": "sv",
    "Swahili": "sw",
    "Turkish": "tr",
}

LANGUAGE_CODE_REVERSED = {v: k for k, v in LANGUAGE_CODE.items()}


COMMON_STARTERS = [
    "Yeah.. ",
    "Yep.. ",
    "Nah.. ",
    "Right.. ",
    "Okay.. ",
    "Alright.. ",
    "Well.. ",
    "So, ",
    "Anyway, ",
    "By the way, ",
    "Actually, ",
    "Honestly, ",
    "Seriously, ",
    "Basically, ",
    "Like",
    "You know, ",
    "I mean, ",
    "I guess, ",
    "I think, ",
    "Apparently, ",
    "Obviously, ",
    "Literally, ",
    "Maybe, ",
    "Probably, ",
    "Exactly, ",
    "Sure, ",
    "Uh...",
    "Uhm...",
    "Ah...",
    "Oh!",
    "Cool.. ",
    "Nice.. "
]