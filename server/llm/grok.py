import os
from typing import Callable
from openai import OpenAI
import time
import re
import json
from utils.constants import COMPANION_NAME, LANGUAGE_CODE_REVERSED

api_key = ""

xclient = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1"  # xAI API 엔드포인트
)
