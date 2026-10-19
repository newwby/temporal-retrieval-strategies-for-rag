
RAW_DATA_DIRECTORY = "raw_data"
INVALID_RAW_DATA_DIRECTORY = "no_explicit_range"
PROCESSED_DATA_PATH = "processed_data"
EXTRACTED_DATA_FILE_NAME = "range_data.jsonl"
SIMPLIFIED_DATA_FILE_NAME = "range_data_simplified.jsonl"
EMBEDDING_DATA_FILE_NAME = "range_embeddings.jsonl"
COMPARISON_DATA_FILE_NAME = "range_embedding_scores.csv"
HISTOGRAM_OUTPUT_DIRECTORY = "similarity_distributions"

# global toggle to allow ollama calls - if disabled prevents locking up on slower machines, defaults to False
OLLAMA_ENABLED = True
UNDEFINED_KEY = "UNDEFINED"

# enable to log messy errors for inspecting data shapes
# mostly relevant in strip_range_data.py
DEBUG_MODE = True

# used to convert month names to numbers for fuzzy range extraction
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