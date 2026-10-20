from experiment.retrievers.faiss import build_index as faiss_build, query as faiss_query
from experiment.retrievers.bm25 import build_index as bm25_build, query as bm25_query

RETRIEVER_REGISTRY = {
    "faiss": {"build": faiss_build, "query": faiss_query},
    "bm25": {"build": bm25_build, "query": bm25_query},
}
