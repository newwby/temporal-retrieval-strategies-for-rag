# tracks indexed docs in a json file
import chromadb
import hashlib
import json
from pathlib import Path

from consts import CHROMA_DB_PATH, DEBUG_MODE, REPORT_COSINE_DISTANCE, TEST_RETRIEVAL_K, ROOT_DATA_PATH, TRACKING_FILE_PATH, VALID_EXTENSIONS

# TODO add chunking for large docs (and versioning for updates)

"""
set up chromadb with persistent storage
"""
def init_chroma():
    chroma_path = Path(__file__).resolve().parent / CHROMA_DB_PATH
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    # cosine by default, slower than l2
    metadata = {"hnsw:space": "cosine"} if REPORT_COSINE_DISTANCE else {"hnsw:space": "l2"}
    collection = client.get_or_create_collection(
        name="documents",
        metadata=metadata
    )
    return client, collection


"""
loads files in the data dir as a list of {id, text} dicts
"""
def load_data():
    # find all files with a valid extension
    data_files = []
    data_path = Path(__file__).resolve().parent / ROOT_DATA_PATH
    for ext in VALID_EXTENSIONS:
        data_files.extend(data_path.glob(f"**/*{ext}")) 
    
    # id is the file name, text is the file content
    document_array = []
    for file in data_files:
        file_name = file.name
        try:
            if file.suffix == ".json":
                json_file_result = parse_json(file)
                for result in json_file_result:
                    document_array.append(result)
            elif file.suffix == ".jsonl":
                jsonl_file_result = parse_jsonl(file)
                for result in jsonl_file_result:
                    document_array.append(result)
            else:
                # plain text (.txt, .md)
                for result in parse_text(file):
                    document_array.append(result)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading {file}: {e}")
    
    return document_array


def parse_text(file):
    file_name = file.name
    with open(file, "r") as f:
        return [{"id": file_name, "text": f.read()}]


def parse_json(file):
    file_name = file.name
    with open(file, "r") as f:
        if DEBUG_MODE:
            print(f"Loading JSON file: {file_name}")
        data = json.load(f)
        # one doc per item if a list, else one doc
        if isinstance(data, list):
            if DEBUG_MODE:
                print(f"Loaded {len(data)} items from {file_name}")
            return [{"id": f"{file_name}_{i}", "text": d.get('text', str(d))} for i, d in enumerate(data)]
        else:
            if DEBUG_MODE:
                print(f"Loaded single item from {file_name}")
            # use text field if there, else the whole object
            text = data.get('text', str(data)) if isinstance(data, dict) else str(data)
            return [{"id": file_name, "text": text}]


def parse_jsonl(file):
    file_name = file.name
    results = []
    with open(file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line {i} in {file_name}: {e}")
                continue
            # keep the line's own id if there, prefix file name so it's unique
            original_id = d.get("id", i)
            doc_id = f"{file_name}_{original_id}"
            text = d.get("text", str(d))
            results.append({"id": doc_id, "text": text})
    return results


"""
re-embeds new or changed docs into the vector db
 [args]
    document_array - list of {id, text} dicts
    embedding_model - sentence_transformers model
    collection - chromadb collection
"""
def refresh_embedding_db(document_array, embedding_model, collection):
    # check tracking file for indexed docs
    tracking_path = Path(__file__).resolve().parent / TRACKING_FILE_PATH
    if not tracking_path.exists():
        # create empty tracking file if missing
        with open(tracking_path, "w") as f:
            json.dump({"ids": [], "checksums": {}}, f)
    
    with open(tracking_path, "r") as f:
        indexed = json.load(f)
    if indexed is None:
        indexed = {"ids": [], "checksums": {}}
    
    changed_docs = []
    for doc in document_array:
        doc_hash = hashlib.md5(doc["text"].encode()).hexdigest()
        doc_id = doc["id"]
        if doc_id not in indexed["ids"] or indexed["checksums"].get(doc_id) != doc_hash:
            changed_docs.append(doc)
    
    if changed_docs:
        if DEBUG_MODE:
            print(f"Embedding {len(changed_docs)} changed documents...")
        embeddings = embedding_model.encode([doc["text"] for doc in changed_docs])
        collection.upsert(
            ids=[doc["id"] for doc in changed_docs],
            embeddings=embeddings.tolist(),
            documents=[doc["text"] for doc in changed_docs]
        )
        
        # update tracking
        indexed["ids"] = list(set(indexed["ids"]) | {doc["id"] for doc in changed_docs})
        for doc in changed_docs:
            indexed["checksums"][doc["id"]] = hashlib.md5(doc["text"].encode()).hexdigest()
        
        with open(tracking_path, "w") as f:
            json.dump(indexed, f)
    else:
        if DEBUG_MODE:
            print("No changed documents found")


# only used for the test run below, move to top if reused
from sentence_transformers import SentenceTransformer
if __name__ == "__main__":
    if DEBUG_MODE:
        print("Running storage.py testing behaviour")
    
    # load db & model from disk
    client, collection = init_chroma()
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models')
    
    # update embeddings
    docs = load_data()
    refresh_embedding_db(docs, embedding_model, collection)
 
    # run test query
    query_text = "financial"
    results = collection.query(
        query_texts=[query_text],
        n_results=TEST_RETRIEVAL_K,
    )

    print(f"sent query: {query_text}")
    print(f"\nChromaDB query results at k={TEST_RETRIEVAL_K}:")
    rank_idx = 0
    for doc_id, text, distance in zip(results["ids"][0], results["documents"][0], results["distances"][0]):
        rank_idx += 1
        print(f"{rank_idx}. (ID: {doc_id}) \"{text}\" [Distance: {distance:.4f}]")