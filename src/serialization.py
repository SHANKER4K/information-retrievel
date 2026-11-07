import json
import os

MODELS = ("bow", "frequency", "tf", "tfidf")


def save(run_dir, result):
    os.makedirs(run_dir, exist_ok=True)
    metadata = {
        "metadata": result["metadata"],
        "vocabulary": result["vocabulary"],
        "documents": result["documents"],
    }
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    for model in MODELS:
        with open(os.path.join(run_dir, f"{model}.json"), "w", encoding="utf-8") as f:
            json.dump(result["models"][model], f, ensure_ascii=False, indent=2)


def load(run_dir):
    with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)

    result = {
        "metadata": metadata["metadata"],
        "vocabulary": metadata["vocabulary"],
        "documents": metadata["documents"],
        "models": {},
    }
    for model in MODELS:
        with open(os.path.join(run_dir, f"{model}.json"), encoding="utf-8") as f:
            result["models"][model] = json.load(f)
    return result


def save_search_config(run_dir, config):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "search_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_search_config(run_dir, default=None):
    path = os.path.join(run_dir, "search_config.json")
    if not os.path.exists(path):
        return dict(default or {})
    with open(path, encoding="utf-8") as f:
        return json.load(f)
