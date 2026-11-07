import nltk

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
    out = []
    i = 0
    while i < len(text):
        if text.startswith("&#", i):
            semi = text.find(";", i + 2)
            if semi != -1:
                digits = text[i + 2:semi]
                if digits.isdigit():
                    try:
                        out.append(chr(int(digits)))
                        i = semi + 1
                        continue
                    except (ValueError, OverflowError):
                        pass
        out.append(text[i])
        i += 1
    return "".join(out)


def clean_tags(text):
    out = []
    in_tag = False
    for ch in text:
        if ch == "<":
            in_tag = True
        elif ch == ">":
            in_tag = False
        elif not in_tag:
            out.append(ch)
    result = _decode_numeric_entities("".join(out))
    for entity, ch in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'")):
        result = result.replace(entity, ch)
    return result


def normalize(text):
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    return " ".join(cleaned.split())


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
