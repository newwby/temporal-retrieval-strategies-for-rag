from pathlib import Path

import chromadb
import hashlib
import json
import re
from datetime import datetime

from experiment.util.consts import CHROMA_DB_PATH, RESULTS_OUTPUT_DIR, RETRIEVER_DEBUG_MODE, REPORT_COSINE_DISTANCE, ROOT_DATA_PATH, TRACKING_FILE_PATH, UPSERT_BATCH_SIZE, VALID_EXTENSIONS



"""
sets up persistent chromadb, collections are named by model (avoids mixed embedding dims)
and by source if given (avoids mixing dev/train in one collection)
leave both out only for old manual scripts using one model and one corpus
"""
def init_chroma(model_name: str | None = None, source: str | None = None, encoding: str | None = None):
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    metadata = {"hnsw:space": "cosine"} if REPORT_COSINE_DISTANCE else {"hnsw:space": "l2"}
    collection_name = "documents"
    if model_name:
        collection_name += f"__{slugify(model_name)}"
    if source:
        collection_name += f"__{slugify(source)}"
    if encoding and encoding != "source":
        collection_name += f"__{slugify(encoding)}"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata=metadata
    )
    return client, collection


"""
loads a corpus file as a list of {id, text, metadata} dicts

corpus_path - load just this file. leave out only for old call sites (globs every
VALID_EXTENSIONS file under ROOT_DATA_PATH, warns if more than one matches)
"""
def load_data(corpus_path: Path | None = None):
    if corpus_path is not None:
        data_files = [corpus_path]
    else:
        data_files = []
        for ext in VALID_EXTENSIONS:
            data_files.extend(ROOT_DATA_PATH.glob(f"**/*{ext}"))
        if len(data_files) > 1:
            names = ", ".join(str(f) for f in sorted(data_files))
            print(
                f"Warning: load_data() called with no corpus_path - loading ALL {len(data_files)} "
                f"file(s) found under {ROOT_DATA_PATH}: {names}"
            )

    document_array = []
    for file in data_files:
        try:
            if file.suffix == ".json":
                document_array.extend(parse_json(file))
            elif file.suffix == ".jsonl":
                document_array.extend(parse_jsonl(file))
            else:
                document_array.extend(parse_text(file))
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading {file}: {e}")

    return document_array


def parse_text(file):
    file_name = file.name
    with open(file, "r", encoding="utf-8") as f:
        return [{"id": file_name, "text": f.read(), "metadata": {}}]


def parse_json(file):
    file_name = file.name
    with open(file, "r", encoding="utf-8") as f:
        if RETRIEVER_DEBUG_MODE:
            print(f"Loading JSON file: {file_name}")
        data = json.load(f)
        if isinstance(data, list):
            if RETRIEVER_DEBUG_MODE:
                print(f"Loaded {len(data)} items from {file_name}")
            return [{"id": f"{file_name}_{i}", "text": d.get('text', str(d)), "metadata": {}} for i, d in enumerate(data)]
        else:
            if RETRIEVER_DEBUG_MODE:
                print(f"Loaded single item from {file_name}")
            text = data.get('text', str(data)) if isinstance(data, dict) else str(data)
            return [{"id": file_name, "text": text, "metadata": {}}]


def parse_jsonl(file):
    file_name = file.name
    results = []
    with open(file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line {i} in {file_name}: {e}")
                continue
            original_id = d.get("id", i)
            doc_id = f"{file_name}_{original_id}"
            text = d.get("text", str(d))
            metadata = _row_metadata(d) if isinstance(d, dict) else {}
            results.append({"id": doc_id, "text": text, "metadata": metadata})
    return results


"""
hashes text with metadata, so a metadata change still forces a re-upsert
"""
def _doc_hash(doc: dict) -> str:
    metadata_json = json.dumps(doc.get("metadata", {}), sort_keys=True)
    return hashlib.md5((doc["text"] + metadata_json).encode()).hexdigest()


def _tracking_path_for(model_name: str | None = None, source: str | None = None, encoding: str | None = None) -> Path:
    tracking_path = TRACKING_FILE_PATH
    if model_name:
        tracking_path = tracking_path.with_name(f"{tracking_path.stem}__{slugify(model_name)}{tracking_path.suffix}")
    if source:
        tracking_path = tracking_path.with_name(f"{tracking_path.stem}__{slugify(source)}{tracking_path.suffix}")
    if encoding and encoding != "source":
        tracking_path = tracking_path.with_name(f"{tracking_path.stem}__{slugify(encoding)}{tracking_path.suffix}")
    return tracking_path


def _load_tracking(tracking_path: Path) -> dict:
    if not tracking_path.exists():
        return {"ids": [], "checksums": {}}
    with open(tracking_path, "r") as f:
        indexed = json.load(f)
    return indexed if indexed is not None else {"ids": [], "checksums": {}}


"""
docs that are new or changed since the tracking file, i.e. what refresh_embedding_db would encode
hashing is cheap next to encoding, so this also gives the cost estimate for the heavy run
banner (see run_guards.py)
"""
def find_changed_docs(document_array, model_name: str | None = None, source: str | None = None, encoding: str | None = None) -> list[dict]:
    indexed = _load_tracking(_tracking_path_for(model_name, source, encoding))
    return [doc for doc in document_array if doc["id"] not in indexed["ids"] or indexed["checksums"].get(doc["id"]) != _doc_hash(doc)]


"""
re-embeds new or changed docs into the vector db
 [args]
    document_array - list of {id, text} dicts
    embedding_model - sentence_transformers model
    collection - chromadb collection
    model_name (optional) - tracking file per model, same as init_chroma's collections,
        without it a new model would see every doc as already indexed
    source (optional) - tracking file per corpus source, keeps it in step with the collection
"""
def refresh_embedding_db(document_array, embedding_model, collection, model_name: str | None = None, source: str | None = None, encoding: str | None = None):
    tracking_path = _tracking_path_for(model_name, source, encoding)
    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    if not tracking_path.exists():
        with open(tracking_path, "w") as f:
            json.dump({"ids": [], "checksums": {}}, f)

    indexed = _load_tracking(tracking_path)
    changed_docs = find_changed_docs(document_array, model_name, source, encoding=encoding)

    if changed_docs:
        if RETRIEVER_DEBUG_MODE:
            print(f"Embedding {len(changed_docs)} changed documents...")
        embeddings = embedding_model.encode([doc["text"] for doc in changed_docs])

        for start in range(0, len(changed_docs), UPSERT_BATCH_SIZE):
            batch = changed_docs[start:start + UPSERT_BATCH_SIZE]
            batch_embeddings = embeddings[start:start + UPSERT_BATCH_SIZE]
            if RETRIEVER_DEBUG_MODE:
                print(f"Upserting batch {start}:{start + len(batch)}...")
            collection.upsert(
                ids=[doc["id"] for doc in batch],
                embeddings=batch_embeddings.tolist(),
                documents=[doc["text"] for doc in batch],
                metadatas=[doc.get("metadata", {}) for doc in batch]
            )

        indexed["ids"] = list(set(indexed["ids"]) | {doc["id"] for doc in changed_docs})
        for doc in changed_docs:
            indexed["checksums"][doc["id"]] = _doc_hash(doc)

        with open(tracking_path, "w") as f:
            json.dump(indexed, f)
    else:
        if RETRIEVER_DEBUG_MODE:
            print("No changed documents found")


def _row_metadata(d: dict) -> dict:
    metadata = {}
    if d.get("entity") is not None:
        metadata["entity"] = str(d["entity"])
    if d.get("link") is not None:
        metadata["link"] = str(d["link"])
    if d.get("para_index") is not None:
        metadata["para_index"] = int(d["para_index"])
    if d.get("date_start") is not None:
        metadata["date_start"] = int(d["date_start"])
    if d.get("date_end") is not None:
        metadata["date_end"] = int(d["date_end"])
    return metadata


def save_rows_to_disk(output_rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(output_rows)} query evaluation result(s) to {output_path}")


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()

