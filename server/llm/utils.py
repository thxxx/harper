import os
from typing import Callable
import re
import json

ALLOWED_CLASSES = {
    "continuation",
    "observation",
    "reflection",
    "filler",
    "selfCorrection",
    "softQuestion",
    "ambientMood",
    "wait",
}

def extract_companion_followup(model_output):
    try:
        data = json.loads(model_output)
        cls = data.get("class")
        txt = data.get("text", "")

        if cls == "wait":
            return (None, "wait")

        if isinstance(txt, str) and txt.strip():
            return (txt.strip(), cls)
        else:
            return (None, cls)

    except Exception as e:
        print("Parsing error : ", e)
        return (None, None)
