import math
from collections import Counter

from preprocessing import run_pipeline, tokenize

MODELS = ("bow", "frequency", "tf", "tfidf")


def build_vocabulary(doc_tokens):
    vocab = sorted({term for tokens in doc_tokens for term in tokens})
    return vocab, {term: idx for idx, term in enumerate(vocab)}


def compute_weights(doc_ids, doc_tokens, model):
    counts = [Counter(tokens) for tokens in doc_tokens]
    n_docs = len(doc_ids)
    weights = {}

    if model == "bow":
        for doc_id, counter in zip(doc_ids, counts):
            for term in counter:
                weights.setdefault(term, {})[doc_id] = 1

    elif model == "frequency":
        for doc_id, counter in zip(doc_ids, counts):
            for term, freq in counter.items():
                weights.setdefault(term, {})[doc_id] = freq

    elif model == "tf":
        for doc_id, counter in zip(doc_ids, counts):
            max_freq = max(counter.values()) if counter else 0
            for term, freq in counter.items():
                weights.setdefault(term, {})[doc_id] = freq / max_freq if max_freq else 0.0

    elif model == "tfidf":
        doc_freq = {}
        for counter in counts:
            for term in counter:
                doc_freq[term] = doc_freq.get(term, 0) + 1
        idf = {term: math.log(n_docs / df) for term, df in doc_freq.items()}
        for doc_id, counter in zip(doc_ids, counts):
            max_freq = max(counter.values()) if counter else 0
            for term, freq in counter.items():
                tf = freq / max_freq if max_freq else 0.0
                weights.setdefault(term, {})[doc_id] = tf * idf[term]

    return weights


def build_all(documents, pipeline_steps):
    processed = []
    for doc in documents:
        text = run_pipeline(doc["text"], pipeline_steps)
        processed.append({"id": doc["id"], "text": doc["text"], "tokens": tokenize(text)})

    doc_ids = [doc["id"] for doc in processed]
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    doc_tokens = [doc["tokens"] for doc in processed]
    vocab, _ = build_vocabulary(doc_tokens)

    result = {
        "metadata": {
            "document_count": len(doc_ids),
            "vocabulary_size": len(vocab),
            "pipeline": pipeline_steps,
            "tf_formula": "f(t,d) / max_w f(w,d)",
            "idf_formula": "log(N / df(t))",
            "matrix_naming": "document_term = rows terms x cols docs; term_document = rows docs x cols terms",
        },
        "vocabulary": vocab,
        "documents": processed,
        "models": {},
    }

    for model in MODELS:
        weights = compute_weights(doc_ids, doc_tokens, model)
        document_term = [dict(weights.get(term, {})) for term in vocab]
        term_document = [{} for _ in doc_ids]
        for term, postings in weights.items():
            for doc_id, weight in postings.items():
                term_document[doc_id_to_idx[doc_id]][term] = weight

        result["models"][model] = {
            "inverted": weights,
            "document_term": document_term,
            "term_document": term_document,
        }

    return result
