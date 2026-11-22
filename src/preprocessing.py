import re

import nltk

_TAG_RE = re.compile(r"<[^>]*(?:>|\Z)")
_NUMERIC_ENTITY_RE = re.compile(r"&#(\d+);")
_NON_ALNUM_RE = re.compile(r"[\W_]+")

_STOPWORDS = None
_STEMMER = None
_LEMMATIZER = None


def _ensure_resource(path, name):
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name, quiet=True)


def _ensure():
    global _STOPWORDS, _STEMMER, _LEMMATIZER
    if _STOPWORDS is None:
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer, WordNetLemmatizer

        _ensure_resource("corpora/stopwords", "stopwords")
        _ensure_resource("corpora/wordnet", "wordnet")
        _STOPWORDS = set(stopwords.words("english"))
        _STEMMER = PorterStemmer()
        _LEMMATIZER = WordNetLemmatizer()


def _decode_numeric_entities(text):
    def replace(match):
        try:
            return chr(int(match.group(1)))
        except (ValueError, OverflowError):
            return match.group(0)

    return _NUMERIC_ENTITY_RE.sub(replace, text)


def clean_tags(text):
    result = _decode_numeric_entities(_TAG_RE.sub("", text))
    for entity, ch in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'")):
        result = result.replace(entity, ch)
    return result


def normalize(text):
    return " ".join(_NON_ALNUM_RE.sub(" ", text.lower()).split())


def remove_stopwords(text):
    _ensure()
    return " ".join(word for word in text.split() if word not in _STOPWORDS)


def stem(text):
    _ensure()
    return " ".join(_STEMMER.stem(word) for word in text.split())


def lemmatize(text):
    _ensure()
    return " ".join(_LEMMATIZER.lemmatize(word) for word in text.split())


_OPS = {
    "tag_cleaning": clean_tags,
    "normalization": normalize,
    "stopwords_removal": remove_stopwords,
    "stemming": stem,
    "lemmatization": lemmatize,
}


def run_pipeline(text, steps):
    for name in steps:
        text = _OPS[name](text)
    return text


def tokenize(text):
    return text.split()
