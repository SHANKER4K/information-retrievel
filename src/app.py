import os

import pandas as pd
import streamlit as st

import data_loader
import indexing
import metrics
import search_engine
import serialization

st.set_page_config(page_title="Reuters IR", layout="wide")

STEPS = [
    ("tag_cleaning", "Tag Cleaning"),
    ("normalization", "Normalization"),
    ("stopwords_removal", "Stopwords Removal"),
    ("stemming", "Stemming"),
    ("lemmatization", "Lemmatization"),
]
DEFAULT_STEPS = ("tag_cleaning", "normalization", "stopwords_removal", "stemming")
DEFAULT_CONFIG = {"model": "tfidf", "method": "cosine", "k": 10, "max_query_words": 20}

MODEL_TABS = {
    "bow": "BoW",
    "frequency": "Frequency",
    "tf": "TF",
    "tfidf": "TF-IDF",
}
MODEL_FORMULAS = {
    "bow": "1 if term present in document, else 0",
    "frequency": "raw count of term in document",
    "tf": "tf = f(t,d) / max_w f(w,d)",
    "tfidf": "tf = f(t,d) / max_w f(w,d), idf = log(N / df(t)), weight = tf * idf",
}


def render_admin():
    st.title("Admin — Indexing")

    st.header("1. Load data")
    source = st.radio("Source", ["Hugging Face dataset", "Local .sgm file/dir"], horizontal=True)
    if source == "Hugging Face dataset":
        max_docs = st.number_input("Max documents (0 = all)", min_value=0, value=500, step=100)
        if st.button("Load HF dataset"):
            with st.spinner("Loading Reuters-21578 from Hugging Face..."):
                st.session_state["docs"] = data_loader.load_hf("ucirvine/reuters21578", max_docs or None)
    else:
        path = st.text_input("Path to .sgm file or folder")
        if st.button("Load local data") and path:
            with st.spinner("Parsing .sgm..."):
                st.session_state["docs"] = data_loader.load_local(path)

    if st.session_state.get("docs"):
        st.success(f"{len(st.session_state['docs'])} documents loaded")

    st.header("2. NLP pipeline")
    selected = []
    for index, (key, label) in enumerate(STEPS, start=1):
        col_check, col_order = st.columns([3, 1])
        use = col_check.checkbox(label, value=key in DEFAULT_STEPS, key=f"use_{key}")
        order = col_order.number_input("Order", min_value=1, max_value=5, value=index, key=f"order_{key}")
        if use:
            selected.append((order, key))
    selected.sort()
    ordered_steps = [key for _, key in selected]
    positions = [position for position, _ in selected]
    duplicate_order = len(positions) != len(set(positions))
    if duplicate_order:
        st.error("Two steps have the same order number.")

    st.header("3. Generate indexes")
    can_generate = bool(st.session_state.get("docs")) and bool(ordered_steps) and not duplicate_order
    if st.button("Generate indexes", disabled=not can_generate):
        with st.spinner("Indexing..."):
            st.session_state["result"] = indexing.build_all(st.session_state["docs"], ordered_steps)

    if st.session_state.get("result"):
        result = st.session_state["result"]
        meta = result["metadata"]
        st.metric("Documents indexed", meta["document_count"])
        st.metric("Vocabulary size", meta["vocabulary_size"])
        st.caption(f"Pipeline order: {' -> '.join(meta['pipeline'])}")
        for model, block in result["models"].items():
            inv = metrics.deep_sizeof(block["inverted"])
            dt = metrics.deep_sizeof(block["document_term"])
            td = metrics.deep_sizeof(block["term_document"])
            st.write(f"**{model}** — inverted {inv/1024:.1f} KB, document_term {dt/1024:.1f} KB, term_document {td/1024:.1f} KB")

    st.header("4. Search settings")
    model = st.selectbox("Model", search_engine.MODELS, index=search_engine.MODELS.index("tfidf"))
    method = st.selectbox("Matching function", search_engine.METHODS, index=0)
    k = st.number_input("Top K results", min_value=1, max_value=100, value=10)
    max_query_words = st.number_input("Max query words", min_value=1, max_value=50, value=20)
    search_config = {
        "model": model,
        "method": method,
        "k": int(k),
        "max_query_words": int(max_query_words),
    }

    st.header("5. Save indexes")
    folder = st.text_input("Run folder", st.session_state.get("run_dir", "indexes/run1"))
    if st.button("Save indexes + search settings"):
        if st.session_state.get("result"):
            serialization.save(folder, st.session_state["result"])
            serialization.save_search_config(folder, search_config)
            st.session_state["run_dir"] = folder
            total = sum(os.path.getsize(os.path.join(folder, name)) for name in os.listdir(folder))
            st.success(f"Saved to {folder} ({total/1024:.1f} KB on disk)")
        else:
            st.warning("Generate indexes first.")


def render_search():
    st.title("Search")
    folder = st.text_input("Index folder", st.session_state.get("run_dir", "indexes/run1"))
    if st.button("Load index"):
        try:
            st.session_state["search_index"] = serialization.load(folder)
            st.session_state["search_config"] = serialization.load_search_config(folder, DEFAULT_CONFIG)
            st.session_state["run_dir"] = folder
            st.success(f"Loaded index from {folder}")
        except (FileNotFoundError, ValueError) as error:
            st.error(f"Could not load index: {error}")

    result = st.session_state.get("search_index")
    if not result:
        st.info("Load a saved index folder to start searching.")
        return

    config = st.session_state.get("search_config", DEFAULT_CONFIG)
    st.caption(
        f"Model: {config.get('model')}  |  Matching: {config.get('method')}  |  "
        f"Top K: {config.get('k')}  |  Max query words: {config.get('max_query_words')}"
    )

    query = st.text_input("Your query")
    if st.button("Search") and query.strip():
        hits = search_engine.search(
            result,
            query,
            model=config.get("model", "tfidf"),
            method=config.get("method", "cosine"),
            k=config.get("k", 10),
            max_query_words=config.get("max_query_words", 20),
        )
        if not hits:
            st.info("No results.")
        for rank, hit in enumerate(hits, start=1):
            st.subheader(f"{rank}. Document {hit['id']} — score {hit['score']:.4f}")
            st.write(hit["text"][:600])


def _model_frame(result, model, max_docs, max_terms):
    term_document = result["models"][model]["term_document"]
    frame = pd.DataFrame(term_document[:max_docs], index=[
        doc["id"] for doc in result["documents"][:max_docs]
    ])
    frame = frame.reindex(columns=result["vocabulary"], fill_value=0.0).fillna(0.0)
    top_terms = frame.sum(axis=0).sort_values(ascending=False).index[:max_terms]
    return frame[list(top_terms)]


def render_visualize():
    st.title("Visualize Weight Matrices")

    folder = st.text_input("Index folder", st.session_state.get("run_dir", "indexes/run1"), key="viz_folder")
    if st.button("Load index", key="viz_load"):
        try:
            st.session_state["viz_index"] = serialization.load(folder)
            st.session_state["run_dir"] = folder
            st.success(f"Loaded index from {folder}")
        except (FileNotFoundError, ValueError) as error:
            st.error(f"Could not load index: {error}")

    result = st.session_state.get("viz_index") or st.session_state.get("result")
    if not result:
        st.info("Generate indexes in Admin or load a saved index folder.")
        return

    meta = result["metadata"]
    pipeline = " -> ".join(meta["pipeline"]) or "none"
    st.caption(
        f"{meta['document_count']} documents | {meta['vocabulary_size']} terms | pipeline: {pipeline}"
    )

    col_terms, col_docs = st.columns(2)
    max_terms = col_terms.slider("Terms shown (highest total weight)", 5, 50, 20)
    documents = meta["document_count"]
    max_docs = col_docs.slider(
        "Documents shown", 1, max(1, min(documents, 50)), min(documents, 10) or 1
    )

    tabs = st.tabs([MODEL_TABS[model] for model in search_engine.MODELS])
    for tab, model in zip(tabs, search_engine.MODELS):
        with tab:
            frame = _model_frame(result, model, max_docs, max_terms)
            non_zero = int((frame.to_numpy() != 0).sum())
            cells = frame.size
            sparsity = 100 * (1 - non_zero / cells) if cells else 0.0
            st.caption(f"Weight: {MODEL_FORMULAS[model]}")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Non-zero cells", non_zero)
            col_b.metric("Sparsity", f"{sparsity:.1f}%")
            col_c.metric("Max weight", f"{frame.to_numpy().max():.4f}" if cells else "0")

            number_format = "%.0f" if model in ("bow", "frequency") else "%.3f"
            st.dataframe(
                frame,
                width="stretch",
                column_config={
                    term: st.column_config.NumberColumn(term, format=number_format)
                    for term in frame.columns
                },
            )


role = st.sidebar.radio("Interface", ["Admin", "Search", "Visualize"])
st.sidebar.caption("Phase 1: Admin indexing. Phase 2: Guest search.")

if role == "Admin":
    render_admin()
elif role == "Search":
    render_search()
else:
    render_visualize()
