import faiss
import numpy as np
from experiment.util.consts import RETRIEVER_DEBUG_MODE, REPORT_COSINE_DISTANCE


"""
builds a faiss index from the chromadb embeddings
"""
def build_index(collection, embedding_dim=None, where: dict | None = None):
    all_data = collection.get(where=where, include=["embeddings", "documents"])
    if not all_data["ids"]:
        print("No documents in collection")
        return None, None

    embeddings = np.array(all_data["embeddings"], dtype=np.float32)
    doc_ids = all_data["ids"]
    if embeddings is None or embeddings.size == 0:
        print("No embeddings found in collection")
        return None, None

    embedding_dim = embeddings.shape[1]

    if REPORT_COSINE_DISTANCE:
        index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(embeddings)
    else:
        index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)

    if RETRIEVER_DEBUG_MODE:
        print(f"Built FAISS index with {len(doc_ids)} documents")
    return index, doc_ids


"""
queries faiss, returns ranked results with metadata
"""
def query(query_text, index, doc_ids, collection, embedding_model, k=10):
    query_embedding = embedding_model.encode(query_text).astype(np.float32)
    distances, indices = index.search(query_embedding.reshape(1, -1), k=k)

    if REPORT_COSINE_DISTANCE:
        distances = 1 - distances

    valid_pairs = [(i, d) for i, d in zip(indices[0], distances[0]) if i != -1]
    retrieved_doc_ids = [doc_ids[i] for i, _ in valid_pairs]
    retrieved_distances = [d for _, d in valid_pairs]
    results = collection.get(ids=retrieved_doc_ids, include=["documents", "metadatas"])

    ranked = []
    for rank, (doc_id, distance) in enumerate(zip(retrieved_doc_ids, retrieved_distances), 1):
        idx = results["ids"].index(doc_id)
        ranked.append({
            "rank": rank,
            "id": doc_id,
            "text": results["documents"][idx],
            "distance": float(distance),
            "metadata": results["metadatas"][idx],
        })

    return ranked

