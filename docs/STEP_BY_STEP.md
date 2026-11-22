# Step-by-step walkthrough

Every number below was produced by this repository's code, not estimated. Follow the steps
in order and you should see the same values.

---

## Step 0 — Set up

```bash
cd information-retrievel
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` is three lines: `streamlit`, `huggingface_hub`, `nltk`.

The first time a query or an index build reaches stopword removal, stemming or lemmatization,
NLTK downloads the `stopwords` and `wordnet` corpora. That needs a network connection once;
afterwards they are cached in `~/nltk_data`.

Sanity check before launching anything:

```bash
.venv/bin/python tests/test_core.py
# all tests passed
```

## Step 1 — Launch the app

```bash
.venv/bin/python main.py
```

`main.py` does one thing: `subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app.py"])`.
Streamlit prints a URL, normally `http://localhost:8501`.

The sidebar has two pages: **Admin** (Phase 1) and **Search** (Phase 2). Start on Admin.

## Step 2 — Phase 1 in the Admin page

### 2a. Load data

Pick **Hugging Face dataset** and leave *Max documents* at `500` (0 means all 21 578).
Press **Load HF dataset**.

What happens internally: `data_loader.load_hf("ucirvine/reuters21578", 500)` downloads
`data/reuters21578.tar.gz` through `huggingface_hub`, opens it with `tarfile`, decodes each
`.sgm` member as `latin-1`, and hands the text to `_extract_docs_from_sgm`. That function
splits on `<REUTERS`, takes everything up to `</REUTERS>`, reads `TITLE` and then `BODY`
(falling back to `TEXT`), and uses the `NEWID` attribute as the id.

After the spinner you should see:

```
500 documents loaded
```

Local data works the same way: choose **Local .sgm file/dir**, type a path, press
**Load local data**. A directory is read in sorted filename order and only `*.sgm` files are
used. Both paths share the same parser, so the result is identical for the same bytes.

### 2b. Choose the pipeline

Five checkboxes, each with an order number 1–5:

| Step | Default | What it does |
| --- | --- | --- |
| Tag Cleaning | ✅ on | strips `<...>` tags and decodes entities |
| Normalization | ✅ on | lowercase, non-alphanumerics → space, collapse whitespace |
| Stopwords Removal | ✅ on | drops NLTK English stopwords |
| Stemming | ✅ on | Porter stemmer |
| Lemmatization | ⬜ off | WordNet lemmatizer |

The defaults give the order `tag_cleaning → normalization → stopwords_removal → stemming`.
Lemmatization is off by default because applying it after stemming would stem the lemmas
again; treat them as alternatives.

The app rejects duplicate order numbers with *"Two steps have the same order number."* and
disables index generation until they are fixed.

### 2c. Generate the indexes

Press **Generate indexes**. This calls `indexing.build_all(docs, ordered_steps)`, which:

1. runs the chosen pipeline over every document's `text`;
2. tokenizes the result with `str.split()` and keeps the tokens alongside the original text;
3. builds a **sorted** vocabulary — this is the row order of `document_term`;
4. for each of the four models builds `inverted`, then projects it into `document_term`
   and `term_document`.

With 500 documents and the default pipeline the panel reports (values from a fresh build):

```
Documents indexed   500
Vocabulary size     5 720
Pipeline order: tag_cleaning -> normalization -> stopwords_removal -> stemming
bow        — inverted 2 041.3 KB, document_term 1 625.2 KB, term_document 1 170.0 KB
frequency  — inverted 2 042.0 KB, document_term 1 625.9 KB, term_document 1 170.7 KB
tf         — inverted 2 816.4 KB, document_term 2 400.3 KB, term_document 1 945.1 KB
tfidf      — inverted 2 816.4 KB, document_term 2 400.3 KB, term_document 1 945.1 KB
```

The KB fluctuate a few percent between runs and Python versions. `inverted` measures slightly
smaller when the index is re-loaded from JSON than right after a build, so the report quotes
2 026.9 / 2 802.0 KB (loaded) and this walkthrough quotes 2 041.3 / 2 816.4 KB (fresh).

Vocabulary reflects wire copy, so expect names and stems: `bahia`, `cocoa`, `reuter`,
`januari`, `allevi`, `improv`.

### 2d. Search settings

Defaults: model `tfidf`, matching `cosine`, top K `10`, max query words `20`. Change them
now if you like — they are saved next to the index and become the Search page defaults.

### 2e. Save

Type a run folder (`indexes/run1` by default) and press **Save indexes + search settings**.
Six files are written:

```
indexes/run1/
  metadata.json        metadata + vocabulary + documents     1.41 MiB
  bow.json             inverted + both matrices              1.75 MiB
  frequency.json       inverted + both matrices              1.75 MiB
  tf.json              inverted + both matrices              2.74 MiB
  tfidf.json           inverted + both matrices              3.38 MiB
  search_config.json   model / method / k / max_query_words      80 B
```

The success message reports the sum of `os.listdir(folder)` sizes, ≈ 11.0 MiB.

## Step 3 — Watch one document go through the pipeline

This is the first Reuters document in `run1`, truncated for display:

```
raw : "BAHIA COCOA REVIEW Showers continued throughout the week in\nthe Bahia cocoa
       zone, alleviating the drought since early\nJanuary and improving prospects ..."
```

| Step | Output (first ~160 characters) |
| --- | --- |
| after `tag_cleaning` | unchanged for this document — no `<` survives loading, because the loader already extracted `TITLE` and `BODY` (473 of the 500 documents do contain `&#3;`, the Reuters end-of-text marker, which tag cleaning turns into a control character that normalization then drops) |
| after `normalization` | `bahia cocoa review showers continued throughout the week in the bahia cocoa zone alleviating the drought since early january and improving prospects ...` |
| after `stopwords_removal` | `bahia cocoa review showers continued throughout week bahia cocoa zone alleviating drought since early january improving prospects ...` |
| after `stemming` | `bahia cocoa review shower continu throughout week bahia cocoa zone allevi drought sinc earli januari improv prospect ...` |

Final tokens begin `['bahia', 'cocoa', 'review', 'shower', 'continu', ...]` — 373 tokens for
this document. Note that `in`, `the`, `and` vanished at the stopword step and that
`throughout` survived because it is not in NLTK's list.

This is exactly what the Search page will do to a query.

## Step 4 — Worked example with exact numbers

Three toy documents, **no pipeline**, so you can check the arithmetic by hand. This is the
corpus used by `tests/test_core.py`.

```
D1: oil price oil rise today
D2: oil company report profit
D3: gold price fall today
```

### 4a. Vocabulary

Sorted union of all terms, 9 entries:

```
['company', 'fall', 'gold', 'oil', 'price', 'profit', 'report', 'rise', 'today']
```

Term indices follow this order, so `document_term[3]` is the `oil` row.

### 4b. Document frequencies

| Term | df | IDF = log(3 / df) |
| --- | --- | --- |
| oil | 2 | 0.405465 |
| price | 2 | 0.405465 |
| today | 2 | 0.405465 |
| rise | 1 | 1.098612 |
| company | 1 | 1.098612 |
| report | 1 | 1.098612 |
| profit | 1 | 1.098612 |
| gold | 1 | 1.098612 |
| fall | 1 | 1.098612 |

### 4c. Term frequencies

`TF = f(t,d) / max_w f(w,d)`. `D1` has 5 tokens, `oil` twice → max frequency 2.

| | D1 | D2 | D3 |
| --- | --- | --- | --- |
| oil | 2/2 = 1.0 | 1/1 = 1.0 | — |
| price | 1/2 = 0.5 | — | 1/1 = 1.0 |
| rise | 1/2 = 0.5 | — | — |
| today | 1/2 = 0.5 | — | 1/1 = 1.0 |
| company / report / profit | — | 1/1 = 1.0 | — |
| gold / fall | — | — | 1/1 = 1.0 |

### 4d. TF-IDF weights

Multiply the two tables:

| Term | D1 | D2 | D3 |
| --- | --- | --- | --- |
| oil | 0.405465 | 0.405465 | — |
| price | 0.202733 | — | 0.405465 |
| rise | 0.549306 | — | — |
| today | 0.202733 | — | 0.405465 |
| company | — | 1.098612 | — |
| report | — | 1.098612 | — |
| profit | — | 1.098612 | — |
| gold | — | — | 1.098612 |
| fall | — | — | 1.098612 |

### 4e. Matrix orientation

`document_term` (rows = terms, cols = docs) for the `oil` row:

```json
{"D1": 0.405465, "D2": 0.405465}
```

`term_document` (rows = docs, cols = terms) for `D2`:

```json
{"oil": 0.405465, "company": 1.098612, "report": 1.098612, "profit": 1.098612}
```

Sum of lengths: `document_term` has 9 rows, `term_document` has 3. Both hold the same 9
non-zero values the inverted index holds.

### 4f. Document norms

`‖d‖ = sqrt(Σ w²)` for the TF-IDF row:

| Doc | Computation | Norm |
| --- | --- | --- |
| D1 | sqrt(0.405465² + 0.202733² + 0.549306² + 0.202733²) | 0.740500 |
| D2 | sqrt(0.405465² + 3 × 1.098612²) | 1.945572 |
| D3 | sqrt(2 × 0.405465² + 2 × 1.098612²) | 1.656110 |

## Step 5 — Score the query `oil price`

Query tokens: `['oil', 'price']`. Both occur once, so max frequency is 1 and both get
`TF = 1.0`. Both are in the inverted index with `df = 2`, so both get
`IDF = log(3/2) = 0.405465`.

```
w_q(oil)   = 1.0 × 0.405465 = 0.405465
w_q(price) = 1.0 × 0.405465 = 0.405465
‖q‖        = sqrt(2 × 0.405465²) = 0.573415
```

Dot products, accumulated from the posting lists:

| Doc | Terms hit | dot(q,d) |
| --- | --- | --- |
| D1 | oil 0.405465 × 0.405465, price 0.405465 × 0.202733 | 0.246603 |
| D2 | oil 0.405465 × 0.405465 | 0.164402 |
| D3 | price 0.405465 × 0.405465 | 0.164402 |

Cosine = dot / (‖q‖ · ‖d‖):

| Rank | Doc | Cosine |
| --- | --- | --- |
| 1 | D1 | 0.246603 / (0.573415 × 0.740500) = **0.5808** |
| 2 | D3 | 0.164402 / (0.573415 × 1.656110) = **0.1731** |
| 3 | D2 | 0.164402 / (0.573415 × 1.945572) = **0.1474** |

Switch the matching function to `dot` and the order changes: `D1 0.2466`, then `D2 = D3
0.1644` (a tie broken by sort stability). Cosine promoted `D3` above `D2` because `D3` is
shorter, so its norm penalises it less.

`D3` still matches on a single query term. Nothing in the ranking requires all terms — a
document sharing one term scores lower, it is not filtered out.

## Step 6 — Phase 2 in the Search page

Switch the sidebar to **Search**.

1. Set *Index folder* to `indexes/run1` (it is pre-filled after an Admin save).
2. Press **Load index**. All six files are read; the caption shows the saved settings.
3. Type a query and press **Search**.

Against the shipped `run1` (TF-IDF, cosine, k = 5):

| Query | Rank | Doc | Score | Title |
| --- | --- | --- | --- | --- |
| `oil prices` | 1 | 127 | 0.3914 | DIAMOND SHAMROCK (DIA) CUTS CRUDE PRICES |
| | 2 | 352 | 0.3409 | SAUDI ARABIA REITERATES COMMITMENT TO OPEC ACCORD |
| | 3 | 248 | 0.3268 | SAUDI ARABIA REITERATES COMMITMENT TO OPEC PACT |
| | 4 | 489 | 0.3224 | STUDY GROUP URGES INCREASED U.S. OIL RESERVES |
| | 5 | 502 | 0.3111 | STUDY GROUP URGES INCREASED U.S. OIL RESERVES |
| `cocoa harvest` | 1 | 1 | 0.2514 | BAHIA COCOA REVIEW |
| | 2 | 275 | 0.2357 | INDONESIAN TEA, COCOA EXPORTS SEEN UP, COFFEE DOWN |
| `interest rates dollar` | 1 | 322 | 0.2586 | JAPAN TO PROMOTE INTEREST RATE LIBERALISATION |
| | 2 | 225 | 0.1532 | JAPAN EXPECTED TO CUT BASE RATE FOR STATE BODIES |
| | 3 | 233 | 0.1461 | ITALIAN TREASURY CUTS INTEREST ON CERTIFICATES |

Note `oil prices` was normalised and stemmed to `oil` + `price` before matching, because
`metadata["pipeline"]` was replayed on the query. If you had built the index without
normalization, the same query would produce different rankings.

The same three queries through the code, without Streamlit:

```python
import sys; sys.path.insert(0, "src")
import serialization, search_engine

result = serialization.load("indexes/run1")
for hit in search_engine.search(result, "oil prices", model="tfidf", method="cosine", k=5):
    print(hit["id"], round(hit["score"], 4), hit["text"][:70])
```

## Step 7 — What to try next

- Build again with `lemmatization` instead of `stemming` and compare the vocabulary size.
- Move `normalization` above `tag_cleaning` and watch tag names like `title` enter the
  vocabulary — this is the effect pinned by `test_pipeline_order_changes_output`.
- Build the same 500 documents with `bow` and compare the Admin page's memory figures
  against `tfidf`: identical structure, different value types.
- Set *Max query words* to 1, search `gold oil` on the toy index, and confirm only the
  `gold` document comes back.
- Search for a term in every document. Under `tfidf` it scores 0 and disappears; under
  `frequency` it ranks normally. That is `log(N/N) = 0`, not a bug.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Could not load index: [Errno 2] ... metadata.json` | wrong folder, or a run that was never saved | save from Admin, or use `indexes/run1` |
| `Could not load index: ... tfidf.json` | run folder is incomplete — `load` reads all four models | rebuild and re-save |
| `KeyError: 'wordnet'` / download errors | NLTK corpora missing and no network | run once with network, then retry |
| Search returns nothing | every matching term had `df = N` and `idf = 0`, or the query's tokens all vanished (stopwords, punctuation only) | switch model to `frequency`, or use content words |
| Two identical titles, different ids | Reuters publishes near-duplicate stories | expected; ids differ |
| Admin memory numbers move between runs | `deep_sizeof` counts allocator-level objects and Python version differences | compare relative sizes, not absolute |
