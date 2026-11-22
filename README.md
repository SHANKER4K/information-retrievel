# Reuters IR — Indexing & Search Engine

A from-scratch Information Retrieval project built around the Reuters-21578 corpus.

- **Phase 1 (Admin):** load documents, configure a dynamic NLP pipeline, build BoW / Frequency / TF / TF-IDF indexes in two matrix orientations plus an inverted index, measure RAM, and serialize everything to JSON.
- **Phase 2 (Guest):** load a saved index from disk, process the query with the **same** NLP pipeline, compute dot-product or cosine similarity, and return the top-K documents.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Verify

```bash
.venv/bin/python tests/test_core.py
```

Expected output: `all tests passed`.

## Project structure

```
main.py                  launch Streamlit
requirements.txt         streamlit, huggingface_hub, nltk
src/
  app.py                 Streamlit UI: Admin page + Search page
  data_loader.py         Hugging Face tarball / local .sgm -> documents
  preprocessing.py       tag cleaning, normalization, stopwords, stemming, lemmatization
  indexing.py            vocabulary + 4 weighting models + 2 matrices + inverted index
  metrics.py             recursive RAM measurement
  serialization.py       save/load indexes and search configuration
  search_engine.py       query processing, matching, top-K
tests/
  test_core.py           assert-based checks on a 3-document toy corpus
```

## Documentation

| Document | Purpose |
| --- | --- |
| [docs/REPORT.md](docs/REPORT.md) | Full project report: architecture, algorithms, formulas, results, limitations |
| [docs/STEP_BY_STEP.md](docs/STEP_BY_STEP.md) | End-to-end walkthrough with exact numbers from the code |
| [docs/DIAGRAMS.md](docs/DIAGRAMS.md) | Mermaid architecture and flow diagrams |
| [docs/VIVA_QA.md](docs/VIVA_QA.md) | Teacher-defense questions, answers, and live-modification drills |

## Dataset

`ucirvine/reuters21578` on Hugging Face. The repository ships a legacy loading script that `datasets` 5.x rejects, so the loader downloads `data/reuters21578.tar.gz` with `huggingface_hub` and parses the SGML directly. Local `.sgm` files/folders are also supported.

## Notes

- Weighting formulas follow the assignment exactly: `TF = f(t,d) / max_w f(w,d)`, `IDF = log(N / df(t))`, `TF-IDF = TF × IDF`.
- Matrices are **sparse** dict-of-dicts; absent cells are never stored (TF-IDF terms with `df = N` keep an explicit `0.0`).
- Matrix orientation follows the raw French wording: `document_term` = rows terms × cols docs, `term_document` = rows docs × cols terms.
- `sklearn` is intentionally not used for TF-IDF because its defaults (smooth IDF, L2 normalization) do not match the required formulas.
