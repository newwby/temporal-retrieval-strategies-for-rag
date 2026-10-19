import os
from pathlib import Path

ANNOTATED_DEV_PATH = Path("data/timeqa/dataset/annotated_dev.json")
DEV_EASY_PATH = Path("data/timeqa/dataset/dev.easy.json")

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
RELATIONS_PATH = DATA_DIR / "meta" / "relations.json"
QUESTION_FORMS_PATH = DATA_DIR / "meta" / "question_forms.json"

CORPUS_FILENAME = "timeqa_corpus.jsonl"
CORPUS_OUTPUT_DIR = DATA_DIR / "corpus"
CORPUS_OUTPUT_PATH = CORPUS_OUTPUT_DIR / CORPUS_FILENAME

QUERIES_OUTPUT_DIR = DATA_DIR / "queries"
QUERIES_OUTPUT_PATH = QUERIES_OUTPUT_DIR / "timeqa_queries.jsonl"

EVAL_QUERY_K = 30
RESULTS_OUTPUT_DIR = DATA_DIR / "results"
RESULTS_PATH = QUERIES_OUTPUT_DIR / "timeqa_structured_test_output.jsonl"

ROOT_DATA_PATH = CORPUS_OUTPUT_DIR
TRACKING_FILE_PATH = DATA_DIR / "indexing" / "indexed_docs.json"
CHROMA_DB_PATH = EXPERIMENT_DIR / "chroma_db"
VALID_EXTENSIONS = [".json", ".jsonl", ".txt", ".md"]


def _resolve_device() -> str:
    override = os.environ.get("EXPERIMENT_DEVICE")
    if override:
        return override
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = _resolve_device()

TEST_RETRIEVAL_K = 5
REPORT_COSINE_DISTANCE = True

UPSERT_BATCH_SIZE = 4000

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

EXPECTED_TOTAL_QUESTIONS = 3021
EXPECTED_WITH_EVIDENCE = 2674
EXPECTED_WITHOUT_EVIDENCE = 347
EXPECTED_MULTI_EVIDENCE = 225

DEBUG_MODE = False
RETRIEVER_DEBUG_MODE = True

LLM_BACKEND = "openai"
LLM_BASE_URL = "http://127.0.0.1:8080/v1"
LLM_MODEL_NAME = "llama3.1:8b"
LLM_BATCH_SIZE = 32
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

PHRASING_CACHE_DIR = DATA_DIR / "phrasing"
PHRASING_COVERAGE_MIN = 0.95
QUESTION_FORM_COVERAGE_MIN = 0.95
CORPUS_ENCODING_COVERAGE_MIN = 0.95

EMBED_DOCS_PER_SECOND = {"cuda": 800, "cpu": 60}

SMALL_DEFAULT_MODEL = "all-MiniLM-L6-v2"
HEAVY_RUN_QUERY_THRESHOLD = 5000
HEAVY_RUN_REEMBED_THRESHOLD = 1000
HEAVY_RUN_PAUSE_SECONDS = 5
