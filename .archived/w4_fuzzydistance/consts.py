
RAW_DATA_DIRECTORY = "raw_data"
INVALID_RAW_DATA_DIRECTORY = "no_explicit_range"
PROCESSED_DATA_PATH = "processed_data"
EXTRACTED_DATA_FILE_NAME = "range_data.jsonl"
SIMPLIFIED_DATA_FILE_NAME = "range_data_simplified.jsonl"
EMBEDDING_DATA_FILE_NAME = "range_embeddings.jsonl"
COMPARISON_DATA_FILE_NAME = "range_embedding_scores.csv"
HISTOGRAM_OUTPUT_DIRECTORY = "similarity_distributions"

# toggle ollama calls, disable on slow machines to avoid lock ups
OLLAMA_ENABLED = True
UNDEFINED_KEY = "UNDEFINED"

# log extra errors to check data shapes
# mostly used in strip_range_data.py
DEBUG_MODE = True

# month name to number lookup
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
