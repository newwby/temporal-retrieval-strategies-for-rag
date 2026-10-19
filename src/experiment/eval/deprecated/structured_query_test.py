import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from experiment.util.consts import EVAL_QUERY_K, RESULTS_PATH
from experiment.eval.metrics import hit_at_k, precision_at_k, recall_at_k, reciprocal_rank
from experiment.eval.deprecated.retriever_manual_test import build_faiss_index, query_with_faiss
from experiment.util.storage import init_chroma, load_data, refresh_embedding_db
from experiment.eval.deprecated.structured_query import load_seed_queries

"""
structured query test

runs the two seed queries (SEED_QUERY_IDS in structured_query.py) through the faiss retriever
from retriever_manual_test.py at k=EVAL_QUERY_K (30), no entity filter, against the whole corpus

writes one {"test_query", "k_results"} line per query to RESULTS_PATH and prints
hit@k / precision@k / recall@k / mrr against the gold ids (printed only, not written)

not part of the main.py pipeline, needs the ml deps and embeds live, so run it directly:
    python structured_query_test.py
"""


"""
writes one {"test_query", "k_results"} json object per line
"""
def write_structured_results(rows: list[dict], output_path: Path) -> None:
    # make sure the output dir exists so this runs on its own
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    # step 1: connect to chroma and embed new or changed paragraphs (cheap after the first run)
    client, collection = init_chroma()
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models')
    docs = load_data()
    refresh_embedding_db(docs, embedding_model, collection)

    # step 2: flat faiss index over the chroma embeddings
    index, doc_ids = build_faiss_index(collection, embedding_dim=384)

    # step 3: load the two seed queries
    structured_queries = load_seed_queries()

    output_rows = []
    for structured_query in structured_queries:
        # step 4: search with the question text at k=EVAL_QUERY_K, no entity filter
        k_results = query_with_faiss(
            structured_query.question_text,
            index,
            doc_ids,
            collection,
            embedding_model,
            k=EVAL_QUERY_K,
        )

        # step 5: keep just test_query and k_results for the output file
        output_rows.append({
            "test_query": structured_query.question_text,
            "k_results": k_results,
        })

        # step 6: score against the gold ids and print (not written)
        retrieved_ids = [result["id"] for result in k_results]
        hit = hit_at_k(retrieved_ids, structured_query.gold_doc_ids)
        precision = precision_at_k(retrieved_ids, structured_query.gold_doc_ids)
        recall = recall_at_k(retrieved_ids, structured_query.gold_doc_ids)
        mrr = reciprocal_rank(retrieved_ids, structured_query.gold_doc_ids)

        print(f"[{structured_query.relation_mode}] {structured_query.question_text}")
        print(f"  time: {structured_query.time_kind} ({structured_query.time_start} to {structured_query.time_end})")
        print(f"  gold_doc_ids: {structured_query.gold_doc_ids}")
        print(
            f"  Hit@{EVAL_QUERY_K}={hit}  "
            f"Precision@{EVAL_QUERY_K}={precision:.4f}  "
            f"Recall@{EVAL_QUERY_K}={recall:.4f}  "
            f"MRR={mrr:.4f}"
        )
        print()

    # write all rows once every query has run
    write_structured_results(output_rows, RESULTS_PATH)
    print(f"Wrote {len(output_rows)} structured query result(s) to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
