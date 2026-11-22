# Reuters IR — Project Report

A from-scratch information retrieval system over the Reuters-21578 news corpus, built in
pure Python with a Streamlit front end.

**Contents**

1. [Objectives](#1-objectives)
2. [Dataset](#2-dataset)
3. [Architecture](#3-architecture)
4. [Phase 1 — Admin indexing](#4-phase-1--admin-indexing)
5. [Phase 2 — Guest search](#5-phase-2--guest-search)
6. [Algorithms and formulas](#6-algorithms-and-formulas)
7. [Data structures](#7-data-structures)
8. [Persistence format](#8-persistence-format)
9. [Results from the shipped run](#9-results-from-the-shipped-run)
10. [Verification](#10-verification)
11. [Complexity](#11-complexity)
12. [Limitations](#12-limitations)
13. [Possible extensions](#13-possible-extensions)

---

## 1. Objectives

The project is deliberately split into two phases so that the expensive work happens once,
offline, and searching is cheap:

| Phase | Actor | Work |
| --- | --- | --- |
| 1 — Admin | Builds the system | Load documents, choose and order the NLP steps, build four weighting models, build the inverted index and both sparse matrices, measure RAM, save everything to disk |
| 2 — Guest | Uses the system | Load a saved index folder, run the query through the **same** pipeline that the documents went through, score documents, return top-K |

Three properties fall out of that split and are enforced in code:

- **Pipeline symmetry.** The applied pipeline is stored in `metadata["pipeline"]` and replayed
  on the query in `search_engine.search`, so a stemmed index is queried with stemmed terms.
- **No hidden libraries.** TF-IDF is computed by hand. `sklearn` is never imported, because
  its defaults (smoothed IDF, L2-normalised rows) do not match the required formulas.
- **Sparse everywhere.** Matrices are dict-of-dicts², not dense arrays. Absent cells are
  never stored (the one exception is a TF-IDF term with `df = N`, whose weight is an
  explicit `0.0`), which is what makes a 500 × 5720 problem tractable in plain Python.

## 2. Dataset

Source: [`ucirvine/reuters21578`](https://huggingface.co/datasets/ucirvine/reuters21578).

The repository ships a legacy `datasets` loading script that `datasets` 5.x refuses to run.
`data_loader.load_hf` therefore downloads `data/reuters21578.tar.gz` directly with
`huggingface_hub`, walks the tarball, and parses the SGML itself — the same parser used by
`load_local`, so both paths behave identically.

Each `<REUTERS>` element becomes one document:

```json
{"id": "1", "text": "BAHIA COCOA REVIEW Showers continued ..."}
```

- `id` — the `NEWID` attribute, or the running document index if absent.
- `text` — `TITLE` followed by `BODY` (falling back to `TEXT`), joined by a space.
- Documents with neither `TITLE` nor a body are skipped.
- Files are decoded as `latin-1` with `errors="replace"`, which is what the original
  1987 wire copy needs.

## 3. Architecture

```
main.py  ──launch──▶  src/app.py  (Streamlit, two pages)
                          │
        ┌─────────────────┼──────────────────────┐
        ▼                 ▼                      ▼
  data_loader.py    preprocessing.py        serialization.py
        │                 │                      │
        ▼                 ▼                      ▼
   documents ──▶  indexing.py ──▶ result dict ──▶ indexes/<run>/*.json
                                        │
                                        ▼
                                  search_engine.py
                                        │
                                        ▼
                                   metrics.py
```

| Module | Responsibility | Key public names |
| --- | --- | --- |
| `main.py` | Boots Streamlit with `sys.executable -m streamlit run src/app.py` | — |
| `src/app.py` | Two-page UI; owns `st.session_state`; nothing else | `render_admin`, `render_search` |
| `src/data_loader.py` | HF tarball or local `.sgm` → list of `{id, text}` | `load_hf`, `load_local` |
| `src/preprocessing.py` | Five interchangeable text operators + dispatcher | `run_pipeline`, `tokenize`, `clean_tags`, `normalize`, `remove_stopwords`, `stem`, `lemmatize` |
| `src/indexing.py` | Vocabulary, four weighting models, inverted index, both matrices | `build_all`, `compute_weights`, `build_vocabulary` |
| `src/search_engine.py` | Query weighting, matching, cosine/dot, top-K | `search`, `build_query_weights` |
| `src/serialization.py` | JSON save/load of an index run and its search settings | `save`, `load`, `save_search_config`, `load_search_config` |
| `src/metrics.py` | Recursive object-graph size, cycle-safe | `deep_sizeof` |
| `tests/test_core.py` | Nine assert-based checks, no pytest needed | `test_*` |

`src/` is imported as flat modules (`import indexing`), which is why `tests/test_core.py`
inserts `../src` into `sys.path` and why Streamlit is launched from the repo root.

## 4. Phase 1 — Admin indexing

`render_admin()` walks the operator through five numbered sections:

1. **Load data** — `Hugging Face dataset` (with an optional document cap) or a path to a
   local `.sgm` file/folder. The result is cached in `st.session_state["docs"]`.
2. **NLP pipeline** — a checkbox and an order number for each of the five steps. Any subset,
   any order, orders must be unique. The default is
   `tag_cleaning → normalization → stopwords_removal → stemming`.
3. **Generate indexes** — calls `indexing.build_all(docs, ordered_steps)`. Disabled until
   documents exist, at least one step is selected, and no two steps share an order number.
4. **Search settings** — model (`bow`/`frequency`/`tf`/`tfidf`), matching function
   (`cosine`/`dot`), top-K, and a cap on query words. These are saved next to the index and
   become the Guest page defaults.
5. **Save indexes** — writes five JSON files plus the search settings into a run folder
   (default `indexes/run1`), then reports the total on-disk size.

`build_all` is the whole pipeline in one function:

```
documents
  └─ run_pipeline(text, steps) → tokenize → tokens per document
       └─ build_vocabulary → sorted terms + term→index map
            └─ for each of bow, frequency, tf, tfidf:
                 compute_weights → {term: {doc_id: weight}}
                 document_term = [weights[term] for term in vocab]
                 term_document = one row per doc, filled from the postings
```

The processed document list keeps **both** the original `text` (for display) and the
`tokens` (for scoring). Nothing is thrown away, so the Guest page never needs the corpus.

## 5. Phase 2 — Guest search

`render_search()` loads a folder via `serialization.load`, reads `search_config.json`
(falling back to `{"model": "tfidf", "method": "cosine", "k": 10, "max_query_words": 20}`),
and enables the query box. `search_engine.search` then does:

1. **Clamp** an unknown model to `tfidf` and an unknown method to `cosine`.
2. **Truncate first, process second.** `query.split()[:max_query_words]` operates on raw
   whitespace tokens, *before* preprocessing. This is deliberate: it bounds the work a
   maliciously long query can cause, and it means the cap counts the words the user typed,
   not the terms that survive the pipeline.
3. **Replay the stored pipeline**, then `tokenize` again to get query terms.
4. **Weight the query** with the same formula as the documents (see below).
5. **Accumulate dot products** by walking the inverted index: for each query term, for each
   posting `(doc, weight)`, add `query_weight × doc_weight` to that document's score.
   Only documents sharing at least one term are ever touched.
6. **Apply cosine**, if requested: divide by `‖q‖ · ‖d‖`, where `‖q‖` comes from the query
   weights and `‖d‖` is recomputed from the `term_document` row. A zero denominator yields a
   score of 0 instead of a `ZeroDivisionError`.
7. **Filter and rank** — drop non-positive scores, sort descending, keep the first `k`,
   and attach each document's stored text.

Two consequences are worth stating explicitly because they are easy to get wrong:

- **Query TF is normalised against the query's own maximum frequency**, not the document's.
  The same function shape is used on both sides, but on different data.
- **A query term absent from the inverted index is dropped when the model is `tfidf`**,
  because its IDF is undefined. There is no division by zero and no phantom posting list.

## 6. Algorithms and formulas

Notation: `f(t,d)` = raw count of term `t` in document `d`; `max_w f(w,d)` = count of the
most frequent term in `d`; `N` = number of documents; `df(t)` = number of documents
containing `t`. `log` is the natural logarithm throughout, as in Python's `math.log`.

| Model | Weight of term `t` in document `d` | Notes |
| --- | --- | --- |
| `bow` (binary) | `1` if `t ∈ d`, else absent | Frequency information discarded |
| `frequency` | `f(t,d)` | Raw count |
| `tf` | `f(t,d) / max_w f(w,d)` | Longest term in a document always scores 1.0 |
| `tfidf` | `TF(t,d) × IDF(t)`, `IDF(t) = log(N / df(t))` | No smoothing, no L2 row normalisation |

Query-side weights mirror this exactly, with `df` read from the loaded inverted index and
`N` from `metadata["document_count"]`.

**Scoring functions**

- Dot product — `score(q,d) = Σ_{t ∈ q∩d} w_q(t) · w_d(t)`.
- Cosine — `score(q,d) = dot(q,d) / (‖q‖ · ‖d‖)`, with `‖x‖ = sqrt(Σ w²)`.

Cosine only rescales; it never changes which documents match, only their order relative to
documents of different length. On the toy corpus, `oil price` ranks `D1 > D3 > D2` by cosine
but `D1 > D2 = D3` by dot product, because `D3` is shorter and therefore penalised less by
the length normalisation.

**NLP operators**

| Step | Implementation | Effect |
| --- | --- | --- |
| `tag_cleaning` | Character scanner dropping everything between `<` and `>`, then numeric and named HTML entity decoding | Removes SGML tags, including adjacent and nested ones, without a parser; a `>` inside an attribute value ends the tag early, and literal `<...>` in prose is dropped |
| `normalization` | Lowercase, map every non-alphanumeric character to a space, collapse whitespace | `"Oil prices!"` → `"oil prices"`; digits survive |
| `stopwords_removal` | NLTK English stopword set | Removes `the`, `of`, `and`, … |
| `stemming` | NLTK `PorterStemmer` | `prices` → `price`, `reporting` → `report` |
| `lemmatization` | NLTK `WordNetLemmatizer` | Dictionary form; defaults to noun POS |

Order matters and is not cosmetic: normalising before tag cleaning turns `<TITLE>` into the
token `title`, which then enters the vocabulary. `tests/test_core.py::test_pipeline_order_changes_output`
pins this down.

NLTK corpora (`stopwords`, `wordnet`) are fetched on first use by `_ensure()` and cached in
module-level globals, so the download happens at most once per process.

**Inverted index**

`inverted[term][doc_id] = weight` — built once per model and reused for three purposes:
scoring (Phase 2), and materialising both matrices (Phase 1). It *is* the sparse weight
matrix in term-major form; the two "matrices" are just different projections of it.

**Matrix orientation**

Naming follows the French wording given in the assignment, which is the reverse of the
usual English convention:

| Name | Rows | Columns | Python representation |
| --- | --- | --- | --- |
| `document_term` | terms (V) | documents (N) | `list[dict[doc_id, weight]]`, length V |
| `term_document` | documents (N) | terms (V) | `list[dict[term, weight]]`, length N |

Columns of `document_term` are keyed by document **id**, not by a positional index. The
representation is labelled-sparse; there is no integer column header row.

**Memory measurement**

`metrics.deep_sizeof(obj)` sums `sys.getsizeof` over the whole object graph, recursing into
`dict` keys and values and into `list`/`tuple`/`set`/`frozenset` members. A `seen` set of
`id()` values prevents double-counting shared objects and infinite recursion. The number is
CPython-specific and approximate (allocator overhead and arenas are not included), but it is
consistent enough to compare models against each other.

## 7. Data structures

```
result = {
  "metadata":   {"document_count": int, "vocabulary_size": int,
                 "pipeline": [str], "tf_formula": str,
                 "idf_formula": str, "matrix_naming": str},
  "vocabulary": [str],                       # sorted, position = term index
  "documents":  [{"id": str, "text": str, "tokens": [str]}],
  "models": {
    "<model>": {
      "inverted":       {term: {doc_id: float}},
      "document_term":  [{doc_id: float}],   # one dict per term
      "term_document":  [{term: float}],     # one dict per document
    }
  }
}
```

Why dicts rather than numpy arrays: at 500 documents and 5 720 terms a dense matrix would
hold 2.86 M cells, of which only 1.16 % are non-zero. A dict per row costs a few dozen bytes
per stored value and nothing for absent cells, keeps the code readable, and JSON-serialises
directly.

One edge case worth knowing: a TF-IDF term with `df = N` gets `idf = log(1) = 0`, so
`compute_weights` stores an explicit `0.0` posting for it. That value is kept in the index
and in both matrices, and the `score > 0` filter is what keeps it out of results.

## 8. Persistence format

`serialization.save(folder, result)` writes:

| File | Contents |
| --- | --- |
| `metadata.json` | `{"metadata": …, "vocabulary": …, "documents": …}` |
| `bow.json` | `{"inverted": …, "document_term": …, "term_document": …}` |
| `frequency.json` | same shape |
| `tf.json` | same shape |
| `tfidf.json` | same shape |
| `search_config.json` | `{"model", "method", "k", "max_query_words"}` |

All four models are written unconditionally, and `load` reads all four unconditionally, so a
run folder is only valid if it is complete. JSON is written with `ensure_ascii=False` and
`indent=2` — human-readable by design, at the cost of size (see below). If
`search_config.json` is missing, `load_search_config` returns the caller's default dict
rather than failing, which is why old run folders remain searchable.

## 9. Results from the shipped run

`indexes/run1` was produced from the first 500 Reuters-21578 documents with the default
pipeline `tag_cleaning → normalization → stopwords_removal → stemming`.

| Metric | Value |
| --- | --- |
| Documents | 500 |
| Vocabulary | 5 720 |
| Non-zero postings per model | 33 072 |
| Density of the weight matrix | 1.156 % (33 072 / 500 × 5 720) |
| On-disk size, all files | ≈ 11.0 MiB |

| File | Size |
| --- | --- |
| `metadata.json` | 1.41 MiB |
| `bow.json` | 1.75 MiB |
| `frequency.json` | 1.75 MiB |
| `tf.json` | 2.74 MiB |
| `tfidf.json` | 3.38 MiB |

In-memory size, measured by `metrics.deep_sizeof` on the loaded run (CPython 3.13):

| Model | inverted | document_term | term_document |
| --- | --- | --- | --- |
| `bow` | 2 026.9 KB | 1 625.2 KB | 1 170.0 KB |
| `frequency` | 2 027.6 KB | 1 625.9 KB | 1 170.7 KB |
| `tf` | 2 802.0 KB | 2 400.3 KB | 1 945.1 KB |
| `tfidf` | 2 802.0 KB | 2 400.3 KB | 1 945.1 KB |

Two things this table shows:

- `tf` and `tfidf` cost the same in memory: both store one Python `float` per posting.
  `tfidf.json` is larger on disk only because the printed decimals are longer.
- `frequency` is barely larger than `bow` in memory because small integer values are cached
  and counted once by `deep_sizeof`, while `bow` stores the constant `1`.

Sample queries against `run1` (TF-IDF, cosine, k = 5):

| Query | Top result | Score |
| --- | --- | --- |
| `oil prices` | 127 — DIAMOND SHAMROCK (DIA) CUTS CRUDE PRICES | 0.3914 |
| `cocoa harvest` | 1 — BAHIA COCOA REVIEW | 0.2514 |
| `interest rates dollar` | 322 — JAPAN TO PROMOTE INTEREST RATE LIBERALISATION | 0.2586 |

## 10. Verification

`python3 tests/test_core.py` runs nine assert-based checks on a three-document toy corpus
(`D1` oil/price, `D2` oil/company, `D3` gold/price) with **no preprocessing pipeline**, so
expected values can be computed by hand.

| Test | What it pins down |
| --- | --- |
| `test_vocabulary_and_orientations` | 9 terms; `document_term` has V rows, `term_document` has N rows; specific cells |
| `test_bow_values_are_binary` | Repeated term still weighs 1 |
| `test_frequency_counts` | `oil` twice → 2, `price` once → 1 |
| `test_tf_matches_spec` | `oil` → 1.0, `price` → 0.5 (max-frequency normalisation) |
| `test_tfidf_matches_spec` | `IDF = log(N/df)` with `df` 2 and 1, not smoothed |
| `test_pipeline_order_changes_output` | Ordering tag cleaning and normalisation differently changes the tokens |
| `test_search_ranks_by_cosine` | Only `D1`, `D2` match `oil`; scores are descending |
| `test_search_reuses_document_pipeline` | Uppercase query matches a stemmed index via the stored pipeline |
| `test_search_max_query_words` | Truncating `gold oil` to 1 word returns only `D3` |

Expected final line: `all tests passed`.

## 11. Complexity

Let `N` documents, `V` vocabulary, `P` non-zero postings, `Q` query terms, `K` results.

| Stage | Time | Space |
| --- | --- | --- |
| Preprocess all documents | O(total tokens) | O(total tokens) |
| Vocabulary | O(total tokens log V) for the sort | O(V) |
| Weights, one model | O(P + N) | O(P) |
| Both matrices, one model | O(P + V + N) | O(P + V + N) |
| Save / load (JSON) | O(P + V + N) | O(P + V + N) |
| Query weighting | O(Q + Q log Q) | O(Q) |
| Accumulate dot products | O(Σ_{t∈q} postings(t)) | O(matching docs) |
| Cosine | O(P_d) for documents that matched | O(matching docs) |
| Sort | O(M log M), `M` = matching docs | O(M) |

All four models are always built, so Phase 1 costs roughly four times one model. Phase 2
touches only the posting lists of the query's terms, which is why it is fast regardless of
corpus size.

## 12. Limitations

Honest list, roughly in order of practical importance:

- **No relevance evaluation.** There is no judged query set, so precision/recall, MAP or
  nDCG are not reported. Ranking quality is inspected by eye.
- **Lemmatization is noun-only.** `WordNetLemmatizer` is called without a POS tag, so
  verbs and adjectives are not lemmatised properly. This is why stemming, not lemmatization,
  is in the default pipeline — enabling both would double-stem.
- **IDF is zero for terms in every document.** `log(N/N) = 0`, so a term present everywhere
  contributes nothing to a TF-IDF score. That is the assignment formula, not a bug, but it
  means such terms are effectively invisible to the `tfidf` model.
- **Cosine recomputes document norms per query** from `term_document` rows rather than
  caching them at index time. Correct, but it makes cosine markedly slower than dot product
  on large corpora.
- **Loading always reads all four models.** Searching with `bow` still parses
  `tfidf.json`. On a larger corpus the load would dominate latency.
- **JSON is verbose.** ≈ 11 MiB for 500 documents; it scales linearly and would be painful
  at the full 21 578-document corpus. Binary formats or key compression are the obvious fix.
- **Memory figures are CPython-specific** and ignore allocator overhead, shared interning
  and the JSON parse buffers.
- **No incremental updates.** Any change to the corpus or the pipeline means a full rebuild;
  there are no document adds/deletes.
- **Pipeline safety depends on the app layer.** `run_pipeline` raises `KeyError` for an
  unknown step name; only the Streamlit UI constrains the input.
- **Single-word matching only.** No phrase, boolean or wildcard queries; `normalize` has
  already destroyed punctuation that a phrase query would need.
- **Saved-run folder size includes stale files.** The Admin page reports the sum of
  `os.listdir(folder)`, so files left over from an earlier run are counted too.

## 13. Possible extensions

Cheap and useful, in order:

1. Cache document norms at index time and store them in the model block.
2. Load lazily — only the model named in `search_config.json`.
3. Add an evaluated query set and report MAP / nDCG for the four models.
4. Store weights as arrays plus an offset table, or as a compressed format, to shrink
   `indexes/`.
5. Add POS-aware lemmatization so stemming and lemmatization can be used as alternatives.
6. Add phrase queries by keeping token positions in the postings.
7. Add relevance feedback / pseudo-relevance feedback as a fifth weighting model.
