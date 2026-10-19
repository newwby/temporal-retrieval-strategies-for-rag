
from sentence_transformers import SentenceTransformer
from experiment.util.consts import TEST_RETRIEVAL_K, RETRIEVER_DEBUG_MODE
from storage import init_chroma, load_data, refresh_embedding_db

if __name__ == "__main__":
    if RETRIEVER_DEBUG_MODE:
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

