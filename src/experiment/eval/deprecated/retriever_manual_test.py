import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from experiment.util.consts import RETRIEVER_DEBUG_MODE, REPORT_COSINE_DISTANCE, TEST_RETRIEVAL_K
from experiment.util.storage import init_chroma, load_data, refresh_embedding_db

# original faiss test

"""
builds a faiss index from the chromadb embeddings
"""
# deprecated, use retrievers/faiss.py, kept so older tests don't break
def build_faiss_index(collection, embedding_dim=384):
    # get docs & embeddings from chromadb
    all_data = collection.get(include=["embeddings", "documents"])
    if not all_data["ids"]:
        print("No documents in collection")
        return None, None

    # validate embeddings shape
    embeddings = np.array(all_data["embeddings"], dtype=np.float32)
    doc_ids = all_data["ids"]
    if embeddings is None or embeddings.size == 0:
        print("No embeddings found in collection")
        return None, None

    # inner product on normalised vectors (cosine) if REPORT_COSINE_DISTANCE, else l2
    if REPORT_COSINE_DISTANCE:
        index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(embeddings)
    else:
        index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)

    # output
    if RETRIEVER_DEBUG_MODE:
        print(f"Built FAISS index with {len(doc_ids)} documents")
    return index, doc_ids

"""
queries faiss, returns ranked results with metadata
"""
# deprecated, use retrievers/faiss.py, kept so older tests don't break
def query_with_faiss(query_text, index, doc_ids, collection, embedding_model, k=10):
    # embed query & search index
    query_embedding = embedding_model.encode(query_text).astype(np.float32)
    distances, indices = index.search(query_embedding.reshape(1, -1), k=k)

    # cosine similarity to distance
    if REPORT_COSINE_DISTANCE:
        distances = 1 - distances

    # map back to chromadb for full docs
    retrieved_doc_ids = [doc_ids[i] for i in indices[0]]
    results = collection.get(ids=retrieved_doc_ids, include=["documents", "metadatas"])

    # ranked with distances
    ranked = []
    for rank, (doc_id, distance) in enumerate(zip(retrieved_doc_ids, distances[0]), 1):
        idx = results["ids"].index(doc_id)
        ranked.append({
            "rank": rank,
            "id": doc_id,
            "text": results["documents"][idx],
            "distance": float(distance)
        })

    return ranked


# for testing
if __name__ == "__main__":
    if RETRIEVER_DEBUG_MODE:
        print("Running retriever_manual_test.py testing behaviour")

    # setup
    client, collection = init_chroma()
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models')
    # db update should move to storage.py
    docs = load_data()
    refresh_embedding_db(docs, embedding_model, collection)

    #TODO separate query layer
    query_text = "mid 80s"

    # comparison
    index, doc_ids = build_faiss_index(collection, embedding_dim=384)
    results = query_with_faiss(query_text, index, doc_ids, collection, embedding_model, k=TEST_RETRIEVAL_K)

    # test output
    print(f"sent query: {query_text}")
    print(f"\nFaiss results at k={TEST_RETRIEVAL_K}:")
    for r in results:
        print(f"{r['rank']}. (ID: {r['id']}) \"{r['text']}\" [Distance: {r['distance']:.4f}]")
