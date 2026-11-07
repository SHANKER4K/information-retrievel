import os


def _field(block, tag):
    start = block.find(f"<{tag}>")
    if start == -1:
        return ""
    content_start = block.find(">", start) + 1
    end = block.find(f"</{tag}>", content_start)
    if end == -1:
        return ""
    return block[content_start:end]


def _attr(block, name):
    key = f"{name}="
    pos = block.find(key)
    if pos == -1:
        return ""
    rest = block[pos + len(key):]
    if rest[:1] in ('"', "'"):
        quote = rest[0]
        return rest[1:rest.find(quote, 1)]
    end = len(rest)
    for i, ch in enumerate(rest):
        if ch in " >":
            end = i
            break
    return rest[:end]


def _extract_docs_from_sgm(text):
    docs = []
    for chunk in text.split("<REUTERS")[1:]:
        end = chunk.find("</REUTERS>")
        if end == -1:
            continue
        block = chunk[:end]
        title = _field(block, "TITLE")
        body = _field(block, "BODY") or _field(block, "TEXT")
        text_content = " ".join(part for part in (title, body) if part).strip()
        if not text_content:
            continue
        doc_id = _attr(block, "NEWID") or str(len(docs))
        docs.append({"id": doc_id, "text": text_content})
    return docs


def load_local(path_or_file):
    if os.path.isdir(path_or_file):
        texts = []
        for name in sorted(os.listdir(path_or_file)):
            if name.endswith(".sgm"):
                with open(os.path.join(path_or_file, name), encoding="latin-1", errors="replace") as f:
                    texts.append(f.read())
    else:
        with open(path_or_file, encoding="latin-1", errors="replace") as f:
            texts = [f.read()]

    docs = []
    for text in texts:
        docs.extend(_extract_docs_from_sgm(text))
    return docs


def load_hf(name="ucirvine/reuters21578", max_docs=None):
    # datasets 5.x rejects the repo's legacy loading script, so pull the
    # archived SGML directly and reuse the same parser as load_local.
    import tarfile

    from huggingface_hub import hf_hub_download

    archive = hf_hub_download(
        repo_id=name, filename="data/reuters21578.tar.gz", repo_type="dataset"
    )
    docs = []
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".sgm"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            raw = handle.read().decode("latin-1", errors="replace")
            docs.extend(_extract_docs_from_sgm(raw))
            if max_docs and len(docs) >= max_docs:
                return docs[:max_docs]
    return docs
