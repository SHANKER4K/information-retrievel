import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import indexing
import preprocessing
import search_engine

DOCS = [
    {"id": "D1", "text": "oil price oil rise today"},
    {"id": "D2", "text": "oil company report profit"},
    {"id": "D3", "text": "gold price fall today"},
]


def test_vocabulary_and_orientations():
    result = indexing.build_all(DOCS, [])
    vocab = result["vocabulary"]
    assert len(vocab) == 9, vocab

    bow = result["models"]["bow"]
    # document_term: rows=terms (V), cols=docs (N)
    assert len(bow["document_term"]) == 9
    # term_document: rows=docs (N), cols=terms (V)
    assert len(bow["term_document"]) == 3

    oil_idx = vocab.index("oil")
    gold_idx = vocab.index("gold")
    assert bow["document_term"][oil_idx] == {"D1": 1, "D2": 1}
    assert bow["document_term"][gold_idx] == {"D3": 1}
    assert bow["term_document"][0]["oil"] == 1
    assert "oil" not in bow["term_document"][2]


def test_bow_values_are_binary():
    bow = indexing.compute_weights(["D1"], [["oil", "oil", "price"]], "bow")
    assert set(bow["oil"].values()) == {1}
    assert set(bow["price"].values()) == {1}


def test_frequency_counts():
    freq = indexing.compute_weights(["D1"], [["oil", "oil", "price"]], "frequency")
    assert freq["oil"] == {"D1": 2}
    assert freq["price"] == {"D1": 1}


def test_tf_matches_spec():
    tf = indexing.compute_weights(["D1"], [["oil", "oil", "price"]], "tf")
    assert tf["oil"] == {"D1": 1.0}
    assert tf["price"] == {"D1": 0.5}


def test_tfidf_matches_spec():
    result = indexing.build_all(DOCS, [])
    tfidf = result["models"]["tfidf"]
    n_docs = 3
    # oil appears in D1 and D2 -> df=2
    oil_idf = math.log(n_docs / 2)
    # gold appears in D3 only -> df=1
    gold_idf = math.log(n_docs / 1)
    assert abs(tfidf["inverted"]["oil"]["D1"] - 1.0 * oil_idf) < 1e-9
    assert abs(tfidf["inverted"]["gold"]["D3"] - 1.0 * gold_idf) < 1e-9


def test_pipeline_order_changes_output():
    raw = "<TITLE>Oil prices!</TITLE><TEXT>Prices rise.</TEXT>"
    tags_first = preprocessing.run_pipeline(raw, ["tag_cleaning", "normalization"])
    normalize_first = preprocessing.run_pipeline(raw, ["normalization", "tag_cleaning"])
    assert tags_first != normalize_first
    assert "title" not in tags_first
    assert "title" in normalize_first


def test_search_ranks_by_cosine():
    result = indexing.build_all(DOCS, [])
    hits = search_engine.search(result, "oil", model="tfidf", method="cosine", k=10)
    assert {hit["id"] for hit in hits} == {"D1", "D2"}
    assert hits[0]["score"] >= hits[1]["score"]


def test_search_reuses_document_pipeline():
    result = indexing.build_all(DOCS, ["normalization", "stopwords_removal", "stemming"])
    hits = search_engine.search(result, "OIL", model="tfidf", method="dot", k=5)
    assert {hit["id"] for hit in hits} == {"D1", "D2"}


def test_search_max_query_words():
    result = indexing.build_all(DOCS, [])
    hits = search_engine.search(result, "gold oil", model="tfidf", method="cosine", k=5, max_query_words=1)
    assert [hit["id"] for hit in hits] == ["D3"]


if __name__ == "__main__":
    test_vocabulary_and_orientations()
    test_bow_values_are_binary()
    test_frequency_counts()
    test_tf_matches_spec()
    test_tfidf_matches_spec()
    test_pipeline_order_changes_output()
    test_search_ranks_by_cosine()
    test_search_reuses_document_pipeline()
    test_search_max_query_words()
    print("all tests passed")
