import math

from preprocessing import run_pipeline, tokenize

MODELS = ("bow", "frequency", "tf", "tfidf")
METHODS = ("cosine", "dot")


def build_query_weights(query_tokens, model, inverted, n_docs):
    counts = {}
    for term in query_tokens:
        counts[term] = counts.get(term, 0) + 1

    weights = {}
    if model == "bow":
        for term in counts:
            weights[term] = 1
    elif model == "frequency":
        weights = dict(counts)
    elif model == "tf":
        max_freq = max(counts.values()) if counts else 0
        for term, freq in counts.items():
            weights[term] = freq / max_freq if max_freq else 0.0
    else:
        max_freq = max(counts.values()) if counts else 0
        for term, freq in counts.items():
            postings = inverted.get(term)
            if not postings:
                continue
            tf = freq / max_freq if max_freq else 0.0
            weights[term] = tf * math.log(n_docs / len(postings))
    return weights


def search(result, query, model="tfidf", method="cosine", k=10, max_query_words=None):
    model = model if model in MODELS else "tfidf"
    method = method if method in METHODS else "cosine"

    words = query.split()
    if max_query_words:
        words = words[:max_query_words]
    query_text = run_pipeline(" ".join(words), result["metadata"].get("pipeline", []))
    query_tokens = tokenize(query_text)

    block = result["models"][model]
    inverted = block["inverted"]
    n_docs = result["metadata"]["document_count"]
    query_weights = build_query_weights(query_tokens, model, inverted, n_docs)

    scores = {}
    for term, query_weight in query_weights.items():
        for doc_id, doc_weight in inverted.get(term, {}).items():
            scores[doc_id] = scores.get(doc_id, 0.0) + query_weight * doc_weight

    if method == "cosine" and scores:
        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
        doc_norms = {
            doc["id"]: math.sqrt(sum(weight * weight for weight in row.values()))
            for doc, row in zip(result["documents"], block["term_document"])
        }
        for doc_id in list(scores):
            denominator = query_norm * doc_norms.get(doc_id, 0.0)
            scores[doc_id] = scores[doc_id] / denominator if denominator else 0.0

    ranked = sorted(
        ((doc_id, score) for doc_id, score in scores.items() if score > 0),
        key=lambda item: item[1],
        reverse=True,
    )[:k]

    docs_by_id = {doc["id"]: doc for doc in result["documents"]}
    return [
        {
            "id": doc_id,
            "score": score,
            "text": docs_by_id.get(doc_id, {}).get("text", ""),
        }
        for doc_id, score in ranked
    ]
