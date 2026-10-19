import re
from rank_bm25 import BM25Okapi
from experiment.util.consts import RETRIEVER_DEBUG_MODE


"""
lowercase and split on non alphanumeric runs, same for build and query
"""
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


"""
builds a bm25 index from the chromadb docs
same signature as faiss build_index so either retriever can be called the same way,
embedding_dim is unused
"""
def build_index(collection, embedding_dim=None, where: dict | None = None):
    all_data = collection.get(where=where, include=["documents"])
    if not all_data["ids"]:
        print("No documents in collection")
        return None, None

    doc_ids = all_data["ids"]
    tokenized_corpus = [_tokenize(doc) for doc in all_data["documents"]]

    index = BM25Okapi(tokenized_corpus)

    if RETRIEVER_DEBUG_MODE:
        print(f"Built BM25 index with {len(doc_ids)} documents")
    return index, doc_ids


"""
queries bm25, returns ranked results with metadata (same shape as faiss.py's query)
embedding_model is unused, kept to match faiss
"""
def query(query_text, index, doc_ids, collection, embedding_model, k=10):
    scores = index.get_scores(_tokenize(query_text))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    retrieved_doc_ids = [doc_ids[i] for i in top_indices]

    results = collection.get(ids=retrieved_doc_ids, include=["documents", "metadatas"])

    ranked = []
    for rank, doc_index in enumerate(top_indices, 1):
        doc_id = doc_ids[doc_index]
        idx = results["ids"].index(doc_id)
        ranked.append({
            "rank": rank,
            "id": doc_id,
            "text": results["documents"][idx],
            "distance": float(scores[doc_index]),
            "metadata": results["metadatas"][idx],
        })

    return ranked
