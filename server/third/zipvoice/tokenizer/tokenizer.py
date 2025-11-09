# Copyright      2023-2024  Xiaomi Corp.        (authors: Zengwei Yao
#                                                         Han Zhu,
#                                                         Wei Kang)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
from abc import ABC, abstractmethod
from functools import reduce
from typing import Dict, List, Optional

import jieba
from lhotse import CutSet
from pypinyin import Style, lazy_pinyin
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

from zipvoice.tokenizer.normalizer import ChineseTextNormalizer, EnglishTextNormalizer

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f \
            https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )

jieba.default_logger.setLevel(logging.INFO)
class KoreanTextNormalizer:
    """
    최소한의 전처리: 공백/문장부호 정규화, 한글 호환 자모를 표준 자모로 치환(필요시).
    필요하다면 숫자->한국어(예: '5'->'오') 치환 규칙을 여기에 확장하세요.
    """
    def normalize(self, text: str) -> str:
        # 중국어와 동일한 문장부호 맵을 재사용
        text = text.replace("，", ",").replace("。", ".").replace("！", "!")
        text = text.replace("？", "?").replace("；", ";").replace("：", ":")
        text = text.replace("、", ",").replace("‘", "'").replace("’", "'")
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("⋯", "…").replace("···", "…").replace("・・・", "…")
        text = text.replace("...", "…")
        # 연속 공백 정규화
        text = re.sub(r"\s+", " ", text).strip()
        return text
# ---- 한글 감지 ----
def is_korean_char(ch: str) -> bool:
    # 음절(U+AC00–U+D7A3)만 대상으로 분해 (자모 영역은 그대로 토큰으로도 쓸 수 있음)
    return "\uAC00" <= ch <= "\uD7A3"

# ---- 한글 음절 -> (초성, 중성, 종성) 분해 ----
_CHO = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
_JUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
_JONG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

def decompose_hangul_syllable(ch: str):
    """AC00 기반 분해. 반환은 (초성 or None, 중성 or None, 종성 or None)"""
    if not is_korean_char(ch):
        return (None, None, None)
    code = ord(ch) - 0xAC00
    cho = code // 588
    jung = (code % 588) // 28
    jong = code % 28
    return (_CHO[cho], _JUNG[jung], _JONG[jong] if _JONG[jong] != '' else None)


class Tokenizer(ABC):
    """Abstract base class for tokenizers, defining common interface."""

    @abstractmethod
    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert list of texts to list of token id sequences."""
        raise NotImplementedError

    @abstractmethod
    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        """Convert list of texts to list of token sequences."""
        raise NotImplementedError

    @abstractmethod
    def tokens_to_token_ids(self, tokens: List[List[str]]) -> List[List[int]]:
        """Convert list of token sequences to list of token id sequences."""
        raise NotImplementedError


class SimpleTokenizer(Tokenizer):
    """The simplpest tokenizer, treat every character as a token,
    without text normalization.
    """

    def __init__(self, token_file: Optional[str] = None):
        """
        Args:
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line.
        """
        # Parse token file
        self.has_tokens = False
        if token_file is None:
            logging.debug(
                "Initialize Tokenizer without tokens file, \
                will fail when map to ids."
            )
            return
        self.token2id: Dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split("\t")
                token, id = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id
        self.pad_id = self.token2id["_"]  # padding
        self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def texts_to_token_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(
        self,
        texts: List[str],
    ) -> List[List[str]]:
        tokens_list = [list(texts[i]) for i in range(len(texts))]
        return tokens_list

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
    ) -> List[List[int]]:
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."

        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.debug(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            token_ids_list.append(token_ids)

        return token_ids_list


class EspeakTokenizer(Tokenizer):
    """A simple tokenizer with Espeak g2p function."""

    def __init__(self, token_file: Optional[str] = None, lang: str = "en-us"):
        """
        Args:
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line.
          lang: the language identifier, see
            https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md
        """
        # Parse token file
        self.has_tokens = False
        self.lang = lang
        if token_file is None:
            logging.debug(
                "Initialize Tokenizer without tokens file, \
                will fail when map to ids."
            )
            return
        self.token2id: Dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split("\t")
                token, id = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id
        self.pad_id = self.token2id["_"]  # padding
        self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def g2p(self, text: str) -> List[str]:
        try:
            tokens = phonemize_espeak(text, self.lang)
            tokens = reduce(lambda x, y: x + y, tokens)
            return tokens
        except Exception as ex:
            logging.warning(f"Tokenization of {self.lang} texts failed: {ex}")
            return []

    def texts_to_token_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(
        self,
        texts: List[str],
    ) -> List[List[str]]:
        tokens_list = [self.g2p(texts[i]) for i in range(len(texts))]
        return tokens_list

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
    ) -> List[List[int]]:
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."

        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.debug(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            token_ids_list.append(token_ids)

        return token_ids_list


class EmiliaTokenizer(Tokenizer):
    def __init__(self, token_file: Optional[str] = None, token_type="phone"):
        """
        Args:
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line.
        """
        assert (
            token_type == "phone"
        ), f"Only support phone tokenizer for Emilia, but get {token_type}."

        self.english_normalizer = EnglishTextNormalizer()
        self.chinese_normalizer = ChineseTextNormalizer()
        self.korean_normalizer  = KoreanTextNormalizer()  # <-- 추가

        self.has_tokens = False
        if token_file is None:
            logging.debug(
                "Initialize Tokenizer without tokens file, \
                    will fail when map to ids."
            )
            return
        self.token2id: Dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split("\t")
                token, id = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id
        self.pad_id = self.token2id["_"]  # padding

        self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def texts_to_token_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    # ---- 기존 preprocess_text는 중국어 구두점 맵만 수행 ----
    def preprocess_text(self, text: str) -> str:
        return self.map_punctuations(text)

    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        for i in range(len(texts)):
            texts[i] = self.preprocess_text(texts[i])

        phoneme_list = []
        for text in texts:
            segments = self.get_segment(text)
            all_phoneme = []
            for seg_text, seg_lang in segments:
                if seg_lang == "zh":
                    phoneme = self.tokenize_ZH(seg_text)
                elif seg_lang == "en":
                    phoneme = self.tokenize_EN(seg_text)
                elif seg_lang == "ko":  # <-- 한국어 분기
                    phoneme = self.tokenize_KO(seg_text)
                elif seg_lang == "pinyin":
                    phoneme = self.tokenize_pinyin(seg_text)
                elif seg_lang == "tag":
                    phoneme = [seg_text]
                else:
                    logging.warning(f"No supported language; skipping: {(seg_text, seg_lang)}")
                    continue
                all_phoneme += phoneme
            phoneme_list.append(all_phoneme)
        return phoneme_list

    # ---- 한국어 토크나이저 ----
    def tokenize_KO(self, text: str) -> List[str]:
        try:
            text = self.korean_normalizer.normalize(text)
            tokens: List[str] = []
            for ch in text:
                if is_korean_char(ch):
                    cho, jung, jong = decompose_hangul_syllable(ch)
                    # 중국어와 동일한 철학: initial에만 '0' suffix
                    if cho is not None:
                        tokens.append(cho + "0")      # 예: ㄱ0, ㅂ0 ...
                    if jung is not None:
                        tokens.append(jung)           # 예: ㅏ, ㅔ, ㅘ ...
                    if jong is not None:
                        tokens.append(jong)           # 예: ㄴ, ㅇ ...
                else:
                    # 한글이 아닌 문자는 그대로 둡니다 (공백/숫자/기호 등)
                    # 필요하면 여기서 숫자를 한국어로 치환하거나 제거 로직을 추가
                    tokens.append(ch)
            return tokens
        except Exception as ex:
            logging.warning(f"Tokenization of Korean texts failed: {ex}")
            return []

    # ---- 세그멘테이션 보강: 한국어 타입 판별 ----
    def get_segment(self, text: str) -> List[str]:
        segments = []
        types = []
        temp_seg = ""
        temp_lang = ""

        _part_pattern = re.compile(r"[<[].*?[>\]]|.")
        text_parts = _part_pattern.findall(text)

        for part in text_parts:
            if self.is_pinyin(part):
                types.append("zh")  # pinyin은 중국어 처리 루틴을 타므로 'zh'로 유지 (아래 split에서 'pinyin'으로 재태깅)
            elif self.is_chinese(part):
                types.append("zh")
            elif self.is_korean(part):                # <-- 추가
                types.append("ko")
            elif self.is_alphabet(part):
                types.append("en")
            else:
                types.append("other")

        assert len(types) == len(text_parts)

        for i in range(len(types)):
            if i == 0:
                temp_seg += text_parts[i]
                temp_lang = types[i]
            else:
                if temp_lang == "other":
                    temp_seg += text_parts[i]
                    temp_lang = types[i]
                else:
                    if types[i] in [temp_lang, "other"]:
                        temp_seg += text_parts[i]
                    else:
                        segments.append((temp_seg, temp_lang))
                        temp_seg = text_parts[i]
                        temp_lang = types[i]
        segments.append((temp_seg, temp_lang))

        # 기존처럼 <>/[] 처리
        segments = self.split_segments(segments)
        return segments

    # ---- 헬퍼: 한국어 판별 ----
    def is_korean(self, part: str) -> bool:
        # part는 한 글자 또는 <>/[] 포함 문자열일 수 있음. 여기서는 한 글자 기준으로만 판별
        return len(part) == 1 and is_korean_char(part)

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
    ) -> List[List[int]]:
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."
        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.debug(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            token_ids_list.append(token_ids)

        return token_ids_list

    def tokenize_ZH(self, text: str) -> List[str]:
        try:
            text = self.chinese_normalizer.normalize(text)
            segs = list(jieba.cut(text))
            full = lazy_pinyin(
                segs,
                style=Style.TONE3,
                tone_sandhi=True,
                neutral_tone_with_five=True,
            )
            phones = []
            for x in full:
                # valid pinyin (in tone3 style) is alphabet + 1 number in [1-5].
                if not (x[0:-1].isalpha() and x[-1] in ("1", "2", "3", "4", "5")):
                    phones.append(x)
                    continue
                else:
                    phones.extend(self.seperate_pinyin(x))
            return phones
        except Exception as ex:
            logging.warning(f"Tokenization of Chinese texts failed: {ex}")
            return []

    def tokenize_EN(self, text: str) -> List[str]:
        try:
            text = self.english_normalizer.normalize(text)
            tokens = phonemize_espeak(text, "en-us")
            tokens = reduce(lambda x, y: x + y, tokens)
            return tokens
        except Exception as ex:
            logging.warning(f"Tokenization of English texts failed: {ex}")
            return []

    def tokenize_pinyin(self, text: str) -> List[str]:
        try:
            assert text.startswith("<") and text.endswith(">")
            text = text.lstrip("<").rstrip(">")
            # valid pinyin (in tone3 style) is alphabet + 1 number in [1-5].
            if not (text[0:-1].isalpha() and text[-1] in ("1", "2", "3", "4", "5")):
                logging.warning(
                    f"Strings enclosed with <> should be pinyin, \
                    but got: {text}. Skipped it. "
                )
                return []
            else:
                return self.seperate_pinyin(text)
        except Exception as ex:
            logging.warning(f"Tokenize pinyin failed: {ex}")
            return []

    def seperate_pinyin(self, text: str) -> List[str]:
        """
        Separate pinyin into initial and final
        """
        pinyins = []
        initial = to_initials(text, strict=False)
        # don't want to share tokens with espeak tokens,
        # so use tone3 style
        final = to_finals_tone3(
            text,
            strict=False,
            neutral_tone_with_five=True,
        )
        if initial != "":
            # don't want to share tokens with espeak tokens,
            # so add a '0' after each initial
            pinyins.append(initial + "0")
        if final != "":
            pinyins.append(final)
        return pinyins

    def map_punctuations(self, text):
        text = text.replace("，", ",")
        text = text.replace("。", ".")
        text = text.replace("！", "!")
        text = text.replace("？", "?")
        text = text.replace("；", ";")
        text = text.replace("：", ":")
        text = text.replace("、", ",")
        text = text.replace("‘", "'")
        text = text.replace("“", '"')
        text = text.replace("”", '"')
        text = text.replace("’", "'")
        text = text.replace("⋯", "…")
        text = text.replace("···", "…")
        text = text.replace("・・・", "…")
        text = text.replace("...", "…")
        return text

    def split_segments(self, segments):
        """
        split segments into smaller parts if special strings enclosed by [] or <>
        are found, where <> denotes pinyin strings, [] denotes other special strings.

        Args:
            segments (list): A list of tuples where each tuple contains:
                - temp_seg (str): The text segment to be split.
                - temp_lang (str): The language code associated with the segment.

        Returns:
            list: A list of smaller segments.
        """
        result = []
        for temp_seg, temp_lang in segments:
            parts = re.split(r"([<[].*?[>\]])", temp_seg)
            for part in parts:
                if not part:
                    continue
                if self.is_pinyin(part):
                    result.append((part, "pinyin"))
                elif self.is_tag(part):
                    result.append((part, "tag"))
                else:
                    result.append((part, temp_lang))
        return result

    def is_chinese(self, char: str) -> bool:
        if char >= "\u4e00" and char <= "\u9fa5":
            return True
        else:
            return False

    def is_alphabet(self, char: str) -> bool:
        if (char >= "\u0041" and char <= "\u005a") or (
            char >= "\u0061" and char <= "\u007a"
        ):
            return True
        else:
            return False

    def is_pinyin(self, part: str) -> bool:
        if part.startswith("<") and part.endswith(">"):
            return True
        else:
            return False

    def is_tag(self, part: str) -> bool:
        if part.startswith("[") and part.endswith("]"):
            return True
        else:
            return False


class DialogTokenizer(EmiliaTokenizer):
    def __init__(self, token_file: Optional[str] = None, token_type="phone"):
        super().__init__(token_file=token_file, token_type=token_type)
        if token_file:
            self.spk_a_id = self.token2id["[S1]"]
            self.spk_b_id = self.token2id["[S2]"]

    def preprocess_text(
        self,
        text: str,
    ) -> str:
        text = re.sub(r"\s*(\[S[12]\])\s*", r"\1", text)
        text = self.map_punctuations(text)
        return text


class LibriTTSTokenizer(Tokenizer):
    def __init__(self, token_file: Optional[str] = None, token_type="char"):
        """
        Args:
          type: the type of tokenizer, e.g., bpe, char, phone.
          tokens: the file that contains information that maps tokens to ids,
            which is a text file with '{token}\t{token_id}' per line if type is
            char or phone, otherwise it is a bpe_model file.
        """
        self.type = token_type
        assert token_type in ["bpe", "char", "phone"]
        try:
            import tacotron_cleaner.cleaners
        except Exception as ex:
            raise RuntimeError(f"{ex}\nPlease run\n" "pip install espnet_tts_frontend")

        self.normalize = tacotron_cleaner.cleaners.custom_english_cleaners

        self.has_tokens = False
        if token_file is None:
            logging.debug(
                "Initialize Tokenizer without tokens file, \
                will fail when map to ids."
            )
            return
        if token_type == "bpe":
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor()
            self.sp.load(token_file)
            self.pad_id = self.sp.piece_to_id("<pad>")
            self.vocab_size = self.sp.get_piece_size()
        else:
            self.token2id: Dict[str, int] = {}
            with open(token_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    info = line.rstrip().split("\t")
                    token, id = info[0], int(info[1])
                    assert token not in self.token2id, token
                    self.token2id[token] = id
            self.pad_id = self.token2id["_"]  # padding
            self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def texts_to_token_ids(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        if self.type == "bpe":
            for i in range(len(texts)):
                texts[i] = self.normalize(texts[i])
            return self.sp.encode(texts)
        else:
            return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(
        self,
        texts: List[str],
    ) -> List[List[str]]:
        for i in range(len(texts)):
            texts[i] = self.normalize(texts[i])

        if self.type == "char":
            tokens_list = [list(texts[i]) for i in range(len(texts))]
        elif self.type == "phone":
            tokens_list = [
                phonemize_espeak(texts[i].lower(), "en-us") for i in range(len(texts))
            ]
        elif self.type == "bpe":
            tokens_list = self.sp.encode(texts, out_type=str)

        return tokens_list

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
    ) -> List[List[int]]:
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."

        assert self.type != "bpe", "BPE tokenizer does not support this function."

        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.debug(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            token_ids_list.append(token_ids)

        return token_ids_list


def add_tokens(cut_set: CutSet, tokenizer: str, lang: str):
    if tokenizer == "emilia":
        tokenizer = EmiliaTokenizer()
    elif tokenizer == "espeak":
        tokenizer = EspeakTokenizer(lang=lang)
    elif tokenizer == "dialog":
        tokenizer = DialogTokenizer()
    elif tokenizer == "libritts":
        tokenizer = LibriTTSTokenizer()
    elif tokenizer == "simple":
        tokenizer = SimpleTokenizer()
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer}.")

    def _prepare_cut(cut):
        # Each cut only contains one supervision
        assert len(cut.supervisions) == 1, (len(cut.supervisions), cut)
        text = cut.supervisions[0].text
        tokens = tokenizer.texts_to_tokens([text])[0]
        cut.supervisions[0].tokens = tokens
        return cut

    cut_set = cut_set.map(_prepare_cut)
    return cut_set


if __name__ == "__main__":
    text = (
        "我们是5年小米人,是吗? Yes I think so! "
        "mr king, 5 years, from 2019 to 2024."
        "霍...啦啦啦超过90%的人<le5>...?!9204"
    )
    tokenizer = EmiliaTokenizer()
    tokens = tokenizer.texts_to_tokens([text])
    print(f"tokens: {'|'.join(tokens[0])}")
