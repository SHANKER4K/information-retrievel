# Defense questions and answers

Questions a teacher or reviewer is likely to ask, with answers grounded in the actual code,
followed by live-modification drills whose expected outputs were verified against this
repository. File references use `path:line`.

---

## Part A — Foundations

**1. What does this system actually do, in one sentence?**

It turns a folder of Reuters news stories into four sparse weighting models plus an inverted
index, saves them as JSON, and answers free-text queries against the saved index by scoring
documents with a dot product or cosine similarity.

**2. Why split into Admin and Guest phases?**

Because indexing cost grows with the corpus, while query cost grows with the query. Phase 1
(`src/app.py:24`) does the expensive work once; Phase 2 (`src/app.py:101`) only reads a
saved folder. That is also the realistic deployment shape: an offline builder, an online
server.

**3. What is the single most important invariant?**

The query must be preprocessed by exactly the pipeline that produced the index. It is stored
in `metadata["pipeline"]` and replayed at `src/search_engine.py:42`. Break it and a stemmed
index is queried with unstemmed words, so `prices` never matches `price`.

**4. Why four weighting models instead of one?**

Because they answer different questions and the assignment requires comparing them. `bow`
ignores frequency, `frequency` uses raw counts, `tf` damps long documents' term repetition,
`tfidf` additionally rewards rare terms. The code builds all four in one pass over the same
counters (`src/indexing.py:75`), so the marginal cost is small.

**5. What is an inverted index, and why is it the core structure?**

A map from term to the documents containing it, with the term's weight:
`inverted[term][doc_id] = weight`. It is what makes search sublinear in corpus size: you
only touch postings for the query's terms instead of scanning every document. With 500
documents and 5 720 terms, the index holds 33 072 postings.

**6. Why do both matrices exist if the inverted index already contains the data?**

The assignment asks for both orientations. They are projections of the same postings:
`document_term` groups by term (rows = terms) and `term_document` groups by document
(rows = docs). The matrices are what you would feed to a matrix library; the inverted index
is what you use to score. They are redundant by construction, and `src/indexing.py:77-81`
builds them from the postings rather than recomputing weights.

**7. Why is the naming `document_term` = terms × docs? That seems backwards.**

It follows the assignment's French wording literally: the *document-term* matrix has
documents per row in the usual English convention, but the spec defines it as terms ×
documents. The code documents the choice in two places so no one "fixes" it later:
`tests/test_core.py:24` asserts `len(document_term) == V` and
`metadata["matrix_naming"]` records it in every saved run.

**8. Why sparse dicts rather than NumPy or SciPy?**

At 500 × 5 720 the dense matrix would be 2.86 M cells, of which 1.16 % are non-zero. Dicts
allocate nothing for absent cells, need no dependency, and serialize straight to JSON. The
trade-off is speed and memory overhead per stored value, which is acceptable at this scale
and becomes the main argument for arrays at larger scale.

**9. What is the actual density?**

33 072 postings out of 500 × 5 720 = 2 860 000 cells, i.e. 1.156 %. Identical for all four
models because they share the same support; only the values differ.

**10. Why not use `sklearn`'s `TfidfVectorizer`?**

Three defaults differ from the required formulas: sklearn applies `idf = log((1+N)/(1+df)) + 1`
(smoothed), normalises each row to unit L2 norm, and by default drops terms with `df < 2`.
This project needs `tf = f/max_f`, `idf = log(N/df)`, no smoothing, no row normalisation.
Using sklearn correctly would mean turning all of that off, which is more code than writing
it directly. The report states this, and no source file imports sklearn.

**11. Why is the dataset loaded by hand instead of with `datasets`?**

`ucirvine/reuters21578` ships a legacy loading script that `datasets` 5.x refuses to
execute. `src/data_loader.py:66` downloads `data/reuters21578.tar.gz` with
`huggingface_hub`, iterates the tarball, decodes `latin-1`, and reuses the same SGML parser
as local files — so the local and remote paths cannot drift apart.

**12. Why `latin-1` and not UTF-8?**

The corpus is 1987 wire copy. `latin-1` maps every byte to a character, so it never fails;
`errors="replace"` handles stray bytes. UTF-8 would reject the same documents.

## Part B — Formulas and numbers

**13. State the TF formula and explain the denominator.**

$TF(t,d) = f(t,d) / max_w f(w,d)$. Dividing by the most frequent term in the same document
damps documents that repeat one word, and guarantees the maximum TF in every document is
exactly 1.0. It is document-local, so it compares documents of different length on a
comparable scale.

**14. State the IDF formula. What happens when a term is in every document?**

$IDF(t) = log(N / df(t))$, natural log. If `df = N`, `IDF = 0`: the term carries no
information for distinguishing documents, so it contributes nothing to a TF-IDF score. The
posting is stored as an explicit `0.0`, and the `score > 0` filter at
`src/search_engine.py:66` keeps such documents out of results.

**15. Why log, and why natural log?**

Log compresses the range: a term in 1 of 500 documents gets 6.2, a term in 250 of 500 gets
0.69. Natural log is what `math.log` gives, and the project states the formula in
`metadata["idf_formula"]` so the base is not ambiguous. Any base only rescales all scores
by a constant and would not change rankings; but mixing bases between index and query would.

**16. Is the TF-IDF here comparable across documents of different length?**

Not absolutely — a long document accumulates more non-zero terms, hence a larger dot
product. That is exactly why the cosine option exists: dividing by `‖q‖·‖d‖` removes length
as a factor. The dot-product option deliberately keeps it.

**17. Compute the TF-IDF for `oil` in `D1` of the toy corpus.**

`D1 = "oil price oil rise today"`. `oil` occurs twice, max frequency is 2, so `TF = 1.0`.
`oil` occurs in `D1` and `D2`, so `df = 2`, `N = 3`, `IDF = log(1.5) = 0.405465`.
`TF-IDF = 0.405465`.

**18. What does cosine do to the top-5 for `oil price` on the toy corpus?**

Dot product: `D1 0.2466`, `D2 0.1644`, `D3 0.1644`. Cosine: `D1 0.5808`, `D3 0.1731`,
`D2 0.1474`. The order changes because `D3` is the shortest document (`‖d‖ = 1.6561` vs
`D2`'s 1.9456) and is penalised less by length normalisation. Cosine never adds or removes
matches; it only rescales.

**19. Why does the query get its own TF normalisation?**

`build_query_weights` (`src/search_engine.py:20-31`) divides by the query's maximum count.
If the query is `oil oil price`, `oil` gets 1.0 and `price` 0.5. Without this the query
vector would scale with repetition, which the document side deliberately does not do.
Same formula, different data.

**20. What is the memory of the TF-IDF index for 500 documents?**

`inverted` 2 802.0 KB, `document_term` 2 400.3 KB, `term_document` 1 945.1 KB, measured by
`metrics.deep_sizeof` on the shipped run. `tf` measures identically because both store one
Python float per posting; `tfidf.json` is bigger on disk only because of longer decimal
strings.

**21. Why are `bow` and `frequency` almost the same size in memory?**

`bow` stores the constant `1` and `frequency` stores small integers. CPython caches integers
−5…256, and `deep_sizeof` counts a shared object once thanks to its `seen` set, so the value
objects are not double-counted. The structure overhead dominates.

## Part C — Implementation choices

**22. Why does the query get truncated before preprocessing, not after?**

`src/search_engine.py:39-41` truncates the raw whitespace tokens, then runs the pipeline.
That bounds the input size to the pipeline itself, which is what could be abused with a
10 000-word query, and it means the user's setting counts the words they typed. A consequence
worth knowing: `the oil` with `max_query_words = 1` becomes just `the`, which stopword removal
deletes, so the search returns nothing.

**23. Why does `build_query_weights` skip terms missing from the inverted index?**

Because it needs `len(postings)` to compute IDF, and a term absent from the corpus has
`df = 0`, so `log(N/0)` is undefined. Skipping is both correct and cheap
(`src/search_engine.py:27-29`). Those terms cannot match any document anyway.

**24. The `inverted` dict and `document_term` are built from the same values. Isn't that
wasteful?**

Yes, and it is deliberate: the assignment requires both views to exist. Since they share
the same float objects, the overhead is dictionary structure, not duplicated numbers. The
real duplication is writing both to JSON.

**25. Why is tag cleaning a hand-written character scanner rather than a regex?**

`src/preprocessing.py:48-61` walks characters, toggling an `in_tag` flag. That handles
adjacent and nested elements, multi-line tags, and unbalanced input without crashing. The
known limitation is that a `>` inside an attribute value ends the tag early — the same
failure a naive `re.sub("<[^>]+>", "")` has — and any literal `<...>` in prose is removed.
It also decodes numeric entities digit by digit in the same pass and leaves malformed ones
alone.

**26. Does tag cleaning matter on the Hugging Face data?**

Only for entity decoding. The loader has already removed all `<...>` markup, so no document
in `run1` contains `<`. But 473 of the 500 contain `&#3;`, the Reuters end-of-text marker;
tag cleaning converts it to a control character, which normalization then deletes. Without
tag cleaning, `&#3;` survives normalization as the token `3`.

**27. Why is the pipeline order user-configurable at all?**

Because order provably changes the result, and the assignment asks for a dynamic pipeline.
`run_pipeline` (`src/preprocessing.py:93-96`) is a fold over a list of names, and the list is
saved with the index. The test `test_pipeline_order_changes_output` demonstrates the effect:
on raw SGML, normalising before tag cleaning inserts the tag names themselves into the
vocabulary.

**28. Why is lemmatization off by default while stemming is on?**

The app defaults (`src/app.py:20`) select stemming only. Applying both would stem lemmas
again (`prices → price → price`, fine, but `running → run → run` vs Porter's `run`), and the
WordNet lemmatizer is called without POS tags, so it only reliably handles nouns. Treat the
two as alternatives. For 500 documents: stemming gives a 5 720-term vocabulary, lemmatization
6 985, both together 5 713.

**29. How does `deep_sizeof` avoid double counting?**

It keeps a `seen` set of `id()` values (`src/metrics.py:4-10`). Any object reachable twice is
counted once, which matters because CPython interns strings and caches small integers —
without `seen`, a vocabulary of repeated strings would look enormous. Recursion covers
`dict` keys and values and list/tuple/set members, but not arbitrary attributes of custom
objects.

**30. Why is `search_config.json` separate from `metadata.json`?**

Because it is Guest-side policy, not index content. Search settings can be changed without
rebuilding, and `load_search_config` (`src/serialization.py:44-49`) returns app defaults when
the file is missing, so older run folders stay usable.

**31. Why does `load` read all four models unconditionally?**

Simplicity: one code path, and `search_engine` can switch models without a reload. It is a
known inefficiency — searching with `bow` still parses `tfidf.json`, 3.38 MiB of the 11 MiB
run.

**32. What stops a `ZeroDivisionError` in cosine?**

`src/search_engine.py:62-63` checks the denominator and assigns `0.0` instead. In practice a
zero norm means an empty weight vector, which cannot reach that code with a positive score,
but the guard makes the function total.

**33. Why filter `score > 0` before sorting?**

Two reasons: TF-IDF documents matched only through zero-IDF terms would otherwise appear with
score 0, and a document that shares no term never enters `scores` anyway. The filter also
prevents "0.0000" results from looking like weak matches.

**34. What does `st.session_state` hold, and why?**

`docs` (loaded corpus), `result` (built index), `run_dir` (last used folder), `search_index`
and `search_config` (loaded Guest state). Streamlit reruns the whole script on every widget
interaction, so anything expensive must live in session state or it would be redone on each
click.

## Part D — Limits and failure modes

**35. Where does this system break first at full corpus scale?**

Loading: all four models are parsed into memory, roughly 11 MiB per 500 documents, so the
full 21 578 documents would be on the order of 450 MiB of JSON and several GB of Python
objects. Second bottleneck is cosine, which recomputes document norms from `term_document`
on every query instead of caching them.

**36. How would you make search faster?**

Cache `‖d‖` per document at index time; load only the configured model; keep postings as
arrays with an offset table instead of dicts; and for cosine, store the pre-normalised
weights so the division happens once.

**37. How do you know the ranking is any good?**

You do not, quantitatively. There is no judged query set, so no MAP, precision, recall or
nDCG can be computed. Results were inspected by eye (for example `cocoa harvest` returning
the Bahia cocoa review first). Adding a small labelled set is the first extension in the
report.

**38. What happens if the pipeline contains an unknown step name?**

`_OPS[name]` raises `KeyError`. Only the Streamlit checkboxes constrain the names, so the
library trusts its caller. If the UI ever grew a free-text step field, this would need a
validation layer.

**39. What if two steps get the same order number?**

The Admin page refuses to build: it shows *"Two steps have the same order number."* and
disables the Generate button (`src/app.py:54-60`). Note the tie-break in `sorted()` would
otherwise be arbitrary.

**40. Why can a search return nothing for a word you can see in a document?**

Three common causes: the term was removed as a stopword; it was stemmed into a different
form; or its `df = N` so its TF-IDF weight is 0. Try the `frequency` model to distinguish the
last case.

**41. What is not covered by the tests?**

Preprocessing operators on their own (only pipeline ordering), the data loader and SGML
parser, serialization round-trips, metrics, the Streamlit UI, the Hugging Face path, and
ranking quality. The nine tests cover the weighting formulas, both matrix orientations,
pipeline symmetry and top-K behaviour.

## Part E — Live-modification drills

Each drill names the file, the change, and the verified outcome.

### Drill 1 — Make BoW store frequencies

`src/indexing.py:19-22`, change `[doc_id] = 1` to `[doc_id] = counter[term]`.

Expected: `python3 tests/test_core.py` fails at `test_bow_values_are_binary`
(`set(bow["oil"].values()) == {1}` becomes `{2}`). Lesson: BoW means presence, not count.

### Drill 2 — Add L2 row normalisation to TF-IDF

In `compute_weights`, after computing `weights`, divide each posting by the document's norm.

Expected: `test_tfidf_matches_spec` fails, because `tfidf["inverted"]["oil"]["D1"]` is no
longer `1.0 × log(3/2)`. Lesson: the assignment formula has no row normalisation; cosine at
query time is where normalisation belongs here.

### Drill 3 — Use a term present in every document

```python
import indexing, search_engine
docs = [{"id": "A", "text": "alpha x"}, {"id": "B", "text": "alpha y"}]
r = indexing.build_all(docs, [])
print(r["models"]["tfidf"]["inverted"]["alpha"])          # {'A': 0.0, 'B': 0.0}
print(search_engine.search(r, "alpha", "tfidf", "cosine", 5))   # []
print(search_engine.search(r, "alpha", "frequency", "dot", 5))  # both docs, score 1.0
```

Lesson: `log(2/2) = 0` is the formula working, not a bug; the `score > 0` filter hides it.

### Drill 4 — Swap normalization and tag cleaning

On the 500-document run, build with
`["tag_cleaning", "normalization", "stopwords_removal", "stemming"]` and then with
`["normalization", "tag_cleaning", "stopwords_removal", "stemming"]`.

Verified: total tokens 52 856 vs 53 736; vocabulary size stays 5 720 by coincidence. The
reason is the `&#3;` marker: normalising first turns it into the token `3`, tag cleaning
first turns it into a control character that normalization drops. On raw SGML the effect is
much bigger — tag names enter the vocabulary.

### Drill 5 — Add lemmatization to the default pipeline

`src/app.py:20`, add `"lemmatization"`. On the same 500 documents:

| Pipeline | Vocabulary |
| --- | --- |
| default (stemming) | 5 720 |
| lemmatization instead of stemming | 6 985 |
| stemming + lemmatization | 5 713 |

Lesson: more aggressive normalisation shrinks the vocabulary; stemming is stronger than
noun-only lemmatization.

### Drill 6 — Drop stopword removal

Build with `["tag_cleaning", "normalization", "stemming"]` instead of the default.

Verified: vocabulary grows from 5 720 to 5 845 and the token stream from 52 856 to 78 626 —
about a third of all tokens are stopwords. Query scores shift because common words now carry
weight, and IDF drops for every remaining term.

### Drill 7 — Truncate a query that starts with a stopword

```python
search_engine.search(result, "the oil", k=5, max_query_words=1)   # []
search_engine.search(result, "the oil", k=5)                      # 5 hits
```

Lesson: truncation happens before preprocessing, so the surviving word can be a stopword and
the query becomes empty after the pipeline.

### Drill 8 — Ask for zero results

`search_engine.search(result, "oil", k=0)` returns `[]` — slicing `[:0]` is legal. The UI
prevents it (`min_value=1` at `src/app.py:79`), but the library is unguarded.

### Drill 9 — Run the whole thing without Streamlit

```python
import sys; sys.path.insert(0, "src")
import data_loader, indexing, serialization, search_engine

docs = data_loader.load_hf(max_docs=100)
result = indexing.build_all(docs, ["tag_cleaning", "normalization", "stopwords_removal", "stemming"])
serialization.save("/tmp/ir_run", result)
loaded = serialization.load("/tmp/ir_run")
print(search_engine.search(loaded, "oil prices", k=3))
```

Demonstrates that `app.py` holds no logic the modules need: the UI is a thin caller.

## Part F — Numbers to have ready

| Quantity | Value |
| --- | --- |
| Documents in the shipped run | 500 |
| Vocabulary | 5 720 |
| Postings per model | 33 072 |
| Matrix density | 1.156 % |
| Dense cells if not sparse | 2 860 000 |
| On-disk size of the run | ≈ 11.0 MiB |
| In-memory TF-IDF index | 2 802.0 + 2 400.3 + 1 945.1 KB |
| `TF` denominator | max term frequency in the same document |
| `IDF` | `log(N / df)` |
| Toy corpus norms (`D1`, `D2`, `D3`) | 0.740500, 1.945572, 1.656110 |
| `oil price` top cosine | 0.5808 (`D1`) |
| Full Reuters-21578 size | 21 578 documents |
| Tests | 9, all passing |
