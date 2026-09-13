"""
Text preprocessing pipelines for the bilingual (English + Arabic) fake news
classifier. This module merges the cleaning logic that was originally written
in two separate notebooks:

  * nlp-mini-en-final.ipynb                     -> English (WELFake dataset)
  * nlp-arabic-fake-news-classification.ipynb   -> Arabic  (Arabic fake-news dataset)

A couple of small bugs in the original notebooks were fixed here (e.g. the
English `remove_extra_whitespace` function did not actually do anything), but
the overall approach is unchanged.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.isri import ISRIStemmer


def ensure_nltk_data() -> None:
    """Download the small NLTK resources required by both pipelines."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, pkg in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                # Offline / restricted network: the pipelines fall back to
                # simpler behaviour (see `tokenize_text` below).
                pass


# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def detect_language(text: str) -> str:
    """Very lightweight heuristic: if a meaningful share of characters fall in
    the Arabic Unicode block, treat the text as Arabic, otherwise English."""
    if not text:
        return "en"
    arabic_chars = len(_ARABIC_RE.findall(text))
    letters = sum(ch.isalpha() for ch in text) or 1
    return "ar" if arabic_chars / letters > 0.3 else "en"


# --------------------------------------------------------------------------- #
# English pipeline (ported from nlp-mini-en-final.ipynb)
# --------------------------------------------------------------------------- #

_lemmatizer = None
_en_stopwords = None


def _get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def _get_en_stopwords():
    global _en_stopwords
    if _en_stopwords is None:
        try:
            _en_stopwords = set(stopwords.words("english"))
        except LookupError:
            _en_stopwords = set()
    return _en_stopwords


def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+", "", text)


def remove_html_tags(text: str) -> str:
    return re.sub(r"<.*?>", "", text)


def remove_punctuation_en(text: str) -> str:
    # Keep apostrophes so contractions ("doesn't") survive.
    return text.translate(str.maketrans("", "", string.punctuation.replace("'", "")))


def remove_numbers(text: str) -> str:
    return re.sub(r"\b\d+\b", "", text)


def remove_special_characters_en(text: str) -> str:
    return re.sub(r"[^a-zA-Z'\s]", " ", text)


def remove_extra_whitespace(text: str) -> str:
    return " ".join(text.split())


def tokenize_text(text: str):
    try:
        return word_tokenize(text)
    except LookupError:
        return text.split()


def remove_stopwords_en(tokens):
    stop = _get_en_stopwords()
    return [w for w in tokens if w not in stop]


def lemmatize_tokens(tokens):
    lem = _get_lemmatizer()
    return [lem.lemmatize(w) for w in tokens]


def clean_text_en(text: str) -> str:
    """Full English cleaning pipeline, mirrors nlp-mini-en-final.ipynb."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = remove_urls(text)
    text = remove_html_tags(text)
    text = text.lower()
    text = remove_punctuation_en(text)
    text = remove_numbers(text)
    text = remove_special_characters_en(text)
    text = remove_extra_whitespace(text)

    tokens = tokenize_text(text)
    tokens = remove_stopwords_en(tokens)
    tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# Arabic pipeline (ported from nlp-arabic-fake-news-classification.ipynb)
# --------------------------------------------------------------------------- #

_arabic_diacritics = re.compile(
    r"""ّ|َ|ً|ُ|ٌ|ِ|ٍ|ْ|ـ""",
    re.VERBOSE,
)

_negation_words = {"لن", "لا", "لم", "ليس", "ما", "غير", "بدون", "مو", "مش"}
_important_words = {
    "بين", "على", "قبل", "بعد", "حول", "حتى", "أمام", "خلال", "ضد", "إلى",
    "منذ", "وراء", "تحت", "وسط", "كما", "حيث", "إذا", "حين", "أما", "بسبب",
    "نتيجة",
}
_extra_remove = {
    "انا", "هو", "هي", "هم", "نحن", "كما", "كان", "كانت", "يكون", "قد", "ثم",
    "سوف", "كل", "أي", "أيضا", "ولكن", "مع", "لدى", "لدي", "هذا", "هذه",
    "ذلك", "تلك", "هناك", "هنا", "الى", "في", "عن", "من", "على", "و", "يا",
}

_ar_stopwords = None
_isri_stemmer = None


def _get_ar_stopwords():
    global _ar_stopwords
    if _ar_stopwords is None:
        try:
            base = set(stopwords.words("arabic"))
        except LookupError:
            base = set()
        _ar_stopwords = (base | _extra_remove) - (_important_words | _negation_words)
    return _ar_stopwords


def _get_isri_stemmer():
    global _isri_stemmer
    if _isri_stemmer is None:
        _isri_stemmer = ISRIStemmer()
    return _isri_stemmer


def normalize_arabic(text: str) -> str:
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def clean_text_ar(text: str) -> str:
    """Full Arabic cleaning pipeline, mirrors
    nlp-arabic-fake-news-classification.ipynb (normalize + stem)."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Keep Arabic letters and whitespace only.
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(_arabic_diacritics, "", text)
    text = normalize_arabic(text)

    stop = _get_ar_stopwords()
    stemmer = _get_isri_stemmer()

    tokens = text.split()
    tokens = [t for t in tokens if t not in stop and len(t) > 1]
    tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# Unified entry point
# --------------------------------------------------------------------------- #

def clean_text(text: str, lang: str) -> str:
    if lang == "ar":
        return clean_text_ar(text)
    return clean_text_en(text)
