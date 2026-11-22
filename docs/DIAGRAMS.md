# Diagrams

Mermaid sources. They render on GitHub, GitLab, and in most Markdown previewers.

---

## 1. System architecture

```mermaid
flowchart TB
    U["Operator"] --> APP["src/app.py<br/>Streamlit UI"]
    APP -->|"Phase 1: Admin"| LOADER["data_loader.py"]
    APP -->|"Phase 1: Admin"| PRE["preprocessing.py"]
    APP -->|"Phase 1: Admin"| IDX["indexing.py"]
    APP -->|"Phase 1: Admin"| SER["serialization.py"]
    APP -->|"Phase 1: Admin"| MET["metrics.py"]
    APP -->|"Phase 2: Search"| SER
    APP -->|"Phase 2: Search"| SE["search_engine.py"]

    LOADER -->|"list of {id, text}"| IDX
    PRE -->|"steps + dispatch"| IDX
    IDX -->|"result dict"| SER
    SER -->|"indexes/run1/*.json"| DISK[("Disk")]
    DISK --> SER
    SER -->|"result dict"| SE
    PRE -->|"same pipeline"| SE
    MET -->|"KB per structure"| APP
```

`app.py` is the only stateful component; every other module is a pure function library.

## 2. Phase 1 — Admin indexing flow

```mermaid
flowchart LR
    A["Load 500 docs"] --> B["Pick steps + order"]
    B --> C{"Orders unique?<br/>≥1 step on?"}
    C -- no --> B
    C -- yes --> D["build_all"]
    D --> E["Measure deep_sizeof"]
    E --> F["Choose model / method / k"]
    F --> G["Save 6 JSON files"]
    G --> H[("indexes/run1/")]
```

## 3. Phase 2 — Guest search flow

```mermaid
flowchart LR
    A["Load index folder"] --> B["Read search_config.json"]
    B --> C["Truncate query to<br/>max_query_words"]
    C --> D["Replay stored pipeline"]
    D --> E["tokenize"]
    E --> F["Weight query terms"]
    F --> G["Accumulate dot products<br/>from inverted index"]
    G --> H{"method"}
    H -- cosine --> I["Divide by ‖q‖·‖d‖"]
    H -- dot --> J["Keep scores"]
    I --> K["Drop score ≤ 0,<br/>sort desc, take k"]
    J --> K
    K --> L["Attach document text"]
```

## 4. One document through the pipeline

```mermaid
flowchart TB
    RAW["raw text<br/>BAHIA COCOA REVIEW Showers ..."] --> TC["tag_cleaning<br/>strip tags, decode entities"]
    TC --> NM["normalization<br/>lowercase, non-alnum → space"]
    NM --> SW["stopwords_removal<br/>drop the, in, and, ..."]
    SW --> ST["stemming<br/>shower, continu, allevi, ..."]
    ST --> TOK["tokenize<br/>str.split"]
    TOK --> V["vocabulary<br/>sorted union of terms"]
```

Order is data, not code: the same five operators can be applied in any permutation, and the
permutation is stored in `metadata["pipeline"]` with the index.

```mermaid
flowchart LR
    subgraph "same function, same order"
      DOC["document text"] --> PIPE["run_pipeline(steps)"]
      Q["query text"] --> PIPE
    end
    PIPE --> T1["document tokens"]
    PIPE --> T2["query tokens"]
```

## 5. Index construction, per model

```mermaid
flowchart TB
    DOCS["documents with tokens"] --> VOC["build_vocabulary<br/>sorted terms + term→index"]
    DOCS --> CNT["Counter per document"]
    CNT --> W["compute_weights"]
    W -.->|"bow: 1"| INV
    W -.->|"frequency: f(t,d)"| INV
    W -.->|"tf: f / max f"| INV
    W -.->|"tfidf: tf × log(N/df)"| INV
    INV["inverted<br/>{term: {doc_id: weight}}"] --> DT["document_term<br/>one dict per term"]
    INV --> TD["term_document<br/>one dict per document"]
```

All three structures hold exactly the same non-zero values; they differ only in how the
postings are grouped.

## 6. Data model

```mermaid
classDiagram
    class Result {
      metadata
      vocabulary
      documents
      models
    }
    class Metadata {
      document_count
      vocabulary_size
      pipeline
      tf_formula
      idf_formula
      matrix_naming
    }
    class Document {
      id
      text
      tokens
    }
    class ModelBlock {
      inverted
      document_term
      term_document
    }
    Result *-- Metadata
    Result *-- "N" Document
    Result *-- "4" ModelBlock : bow, frequency, tf, tfidf
    ModelBlock ..> Inverted : term → doc_id → weight
    ModelBlock ..> DocumentTerm : V rows, one per term
    ModelBlock ..> TermDocument : N rows, one per document
```

Sparse representation:

```
inverted["oil"]              = {"D1": 0.405465, "D2": 0.405465}
document_term[oil_index]     = {"D1": 0.405465, "D2": 0.405465}   # rows = terms
term_document[0]             = {"oil": 0.405465, "price": 0.202733, ...}  # rows = docs
```

## 7. Search sequence

```mermaid
sequenceDiagram
    participant U as User
    participant UI as app.py (Search)
    participant S as serialization
    participant E as search_engine
    participant P as preprocessing

    U->>UI: Load index folder
    UI->>S: load(folder)
    S-->>UI: result dict
    UI->>S: load_search_config(folder, defaults)
    S-->>UI: {model, method, k, max_query_words}

    U->>UI: query "oil prices"
    UI->>E: search(result, query, model, method, k, max)
    E->>E: truncate to max_query_words
    E->>P: run_pipeline(query, metadata.pipeline)
    P-->>E: "oil price"
    E->>E: build_query_weights
    E->>E: accumulate postings · weights
    alt method == cosine
        E->>E: divide by ‖q‖ · ‖d‖
    end
    E-->>UI: [{id, score, text}, ...]
    UI-->>U: ranked results
```

## 8. Persistence layout

```mermaid
flowchart LR
    subgraph MEM["result (in memory)"]
      M1["metadata"]
      M2["vocabulary"]
      M3["documents"]
      M4["models × 4"]
    end
    M1 --> F1["metadata.json"]
    M2 --> F1
    M3 --> F1
    M4 --> F2["bow.json"]
    M4 --> F3["frequency.json"]
    M4 --> F4["tf.json"]
    M4 --> F5["tfidf.json"]
    CFG["search_settings in app"] --> F6["search_config.json"]
    F1 & F2 & F3 & F4 & F5 & F6 --> R[("indexes/run1/")]
```

`load` reads `metadata.json` and all four model files. `load_search_config` tolerates a
missing `search_config.json` and returns the app's defaults.

## 9. Operator dispatch

```mermaid
flowchart LR
    NAME["step name"] --> OPS{"_OPS"}
    OPS -->|tag_cleaning| F1["clean_tags"]
    OPS -->|normalization| F2["normalize"]
    OPS -->|stopwords_removal| F3["remove_stopwords"]
    OPS -->|stemming| F4["stem"]
    OPS -->|lemmatization| F5["lemmatize"]
    F1 & F2 & F3 & F4 & F5 --> OUT["text"]
```

`run_pipeline` is a fold: `text = _OPS[name](text)` for each name in order. An unknown name
raises `KeyError`; the UI is what keeps names valid.
