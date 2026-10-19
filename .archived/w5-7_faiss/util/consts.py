from pathlib import Path

# dense retriever constants
CHROMA_DB_PATH = Path("./chroma_db")
ROOT_DATA_PATH = Path("./data/")
TRACKING_FILE_PATH = Path("./meta/indexed_docs.json")
VALID_EXTENSIONS = [".json", ".jsonl", ".txt", ".md"]

TEST_RETRIEVAL_K = 5
REPORT_COSINE_DISTANCE = True  # cosine distance if true, else l2

# enable debug logging (noisy)
DEBUG_MODE = True