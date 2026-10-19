
__all__ = [
    "hit_at_k", "precision_at_k", "recall_at_k", "reciprocal_rank",
    "normalize_answer", "get_answer_tokens", "compute_exact", "compute_f1",
    "exact_match_score", "f1_score", "chrf_score", "bertscore_answer",
]


import re
import string
from collections import Counter

from nltk.translate.chrf_score import sentence_chrf

from experiment.classes.class_query import EvalQuery
from experiment.util.consts import EVAL_QUERY_K


def eval_metrics(query_data: EvalQuery, k: int | None = None) -> None:
    reported_k = k or EVAL_QUERY_K
    print(f"[{query_data.relation_mode}] {query_data.question_text}")
    print(f"  time: {query_data.time_start} to {query_data.time_end}")
    print(f"  gold_doc_ids: {query_data.gold_doc_ids}")
    latency_str = f"{query_data.latency:.4f}s" if query_data.latency is not None else "N/A"
    chrf_str = f"{query_data.chrf:.4f}" if query_data.chrf is not None else "N/A"
    print(
        f"  Hit@{reported_k}={query_data.hit_at_k}  "
        f"Precision@{reported_k}={query_data.precision_at_k:.4f}  "
        f"Recall@{reported_k}={query_data.recall_at_k:.4f}  "
        f"MRR@{reported_k}={query_data.mrr:.4f}  "
        f"chrF={chrf_str}  "
        f"Latency={latency_str}"
    )


def compute_metrics(
    query_data: EvalQuery,
    retrieved_ids: list[str],
    predicted_answer: str | None = None,
    latency: float | None = None,
    index_resolve_latency: float | None = None,
    mutate_latency: float | None = None,
    retrieval_latency: float | None = None,
    total_latency: float | None = None,
    index_cache_hit: bool | None = None,
) -> EvalQuery:
    query_data.hit_at_k = hit_at_k(retrieved_ids, query_data.gold_doc_ids)
    query_data.precision_at_k = precision_at_k(retrieved_ids, query_data.gold_doc_ids)
    query_data.recall_at_k = recall_at_k(retrieved_ids, query_data.gold_doc_ids)
    query_data.mrr = reciprocal_rank(retrieved_ids, query_data.gold_doc_ids)
    query_data.answer_f1 = f1_score(query_data.gold_answers, predicted_answer) if predicted_answer is not None else None
    query_data.answer_exact_match = exact_match_score(query_data.gold_answers, predicted_answer) if predicted_answer is not None else None
    query_data.chrf = chrf_score(query_data.gold_answers, predicted_answer) if predicted_answer is not None else None
    query_data.latency = latency
    query_data.index_resolve_latency = index_resolve_latency
    query_data.mutate_latency = mutate_latency
    query_data.retrieval_latency = retrieval_latency
    query_data.total_latency = total_latency
    query_data.index_cache_hit = index_cache_hit
    return query_data


def hit_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> bool:
    gold_set = set(gold_ids)
    return any(retrieved_id in gold_set for retrieved_id in retrieved_ids)


def precision_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    if not retrieved_ids:
        return 0.0

    gold_set = set(gold_ids)
    num_correct = sum(1 for retrieved_id in retrieved_ids if retrieved_id in gold_set)
    return num_correct / len(retrieved_ids)


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    if not gold_ids:
        return 0.0
    
    gold_set = set(gold_ids)
    retrieved_set = set(retrieved_ids)
    num_found = sum(1 for gold_id in gold_set if gold_id in retrieved_set)
    return num_found / len(gold_set)


def reciprocal_rank(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    gold_set = set(gold_ids)

    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in gold_set:
            return 1.0 / rank

    return 0.0


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def get_answer_tokens(text: str) -> list[str]:
    if not text:
        return []
    return normalize_answer(text).split()


def compute_exact(gold: str, prediction: str) -> float:
    return float(normalize_answer(gold) == normalize_answer(prediction))


def compute_f1(gold: str, prediction: str) -> float:
    gold_tokens = get_answer_tokens(gold)
    pred_tokens = get_answer_tokens(prediction)

    if not gold_tokens or not pred_tokens:
        return float(gold_tokens == pred_tokens)

    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(gold_answers: list[str], prediction: str) -> float:
    if not gold_answers:
        gold_answers = [""]
    return max(compute_exact(gold, prediction) for gold in gold_answers)


def f1_score(gold_answers: list[str], prediction: str) -> float:
    if not gold_answers:
        gold_answers = [""]
    return max(compute_f1(gold, prediction) for gold in gold_answers)


def chrf_score(gold_answers: list[str], candidate_text: str, beta: float = 1.0) -> float:
    references = [answer for answer in gold_answers if answer]
    if not references or not candidate_text:
        return 0.0
    return max(sentence_chrf(reference, candidate_text, beta=beta) for reference in references)


_BERTSCORE_KEYS = ("precision", "recall", "f1")


"""
bertscore of a prediction against gold answers, returns {"precision", "recall", "f1"}
like f1_score, uses the gold answer with the best f1

no non-empty gold answers -> same as compute_f1 (1.0 if the prediction is also empty, else 0.0),
bertscore of an empty reference is undefined and bert_score raises on it with transformers 5.x

bert_score is imported on call so metrics.py imports without it, the first call downloads the model
"""
def bertscore_answer(
    gold_answers: list[str],
    prediction: str,
    model_type: str | None = None,
    lang: str = "en",
) -> dict:
    from bert_score import score as _bert_score

    references = [answer for answer in gold_answers if answer]
    if not references:
        score = 1.0 if not prediction else 0.0
        return {"precision": score, "recall": score, "f1": score}

    cands = [prediction] * len(references)
    gold_answers = references
    kwargs = {"lang": lang} if model_type is None else {"model_type": model_type}
    p, r, f1 = _bert_score(cands, list(gold_answers), verbose=False, **kwargs)

    best = int(f1.argmax())
    return {
        "precision": float(p[best]),
        "recall": float(r[best]),
        "f1": float(f1[best]),
    }
