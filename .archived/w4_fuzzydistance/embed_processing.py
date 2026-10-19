import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from consts import DEBUG_MODE, EMBEDDING_DATA_FILE_NAME, EXTRACTED_DATA_FILE_NAME, PROCESSED_DATA_PATH

"""
embed_processing.py
This module handles the embedding of range data using the SentenceTransformer model.
"""

# enable to check basic functionality with simple text data (faster than embedding the full range data)
TEST_MODE = False


def write_embeddings(output_path: Path):
    has_language_embeddings = None
    records = []
    texts = []

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    range_data_path = Path(__file__).resolve().parent / PROCESSED_DATA_PATH / EXTRACTED_DATA_FILE_NAME

    # load data & extract the range_numeric and range_fuzzy for embedding whilst keying by output_id to match back
    with range_data_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            obj = json.loads(line)

            # validate
            if obj.get("output_id") is None:
                if DEBUG_MODE:
                    print(f"Missing output_id in the JSON entry at line {line_number}. Skipping embedding for this entry.")
                continue

            if "range_numeric" not in obj or "range_fuzzy" not in obj:
                if DEBUG_MODE:
                    print(f"Missing required fields in the JSON entry at line {line_number}. Skipping embedding for this entry.")
                continue
        
            records.append(obj["output_id"])

            # verify shape
            row_has_language = obj.get("range_language") is not None
            if has_language_embeddings is None:
                has_language_embeddings = row_has_language
            elif has_language_embeddings != row_has_language:
                raise ValueError(
                    f"Inconsistent range_language presence at line {line_number}."
                )

            if row_has_language == True:
                texts.extend([obj["range_numeric"], obj["range_fuzzy"], obj["range_language"]])
            else:
                texts.extend([obj["range_numeric"], obj["range_fuzzy"]])

    # preserve order of records and texts, so that embeddings can be matched back to the correct output_id
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    # if range_language not present there are 2 embeddings per output_id, else 3 embeddings per output_id
    if has_language_embeddings:
        # handling for range_language present:
        # indices of embeddings are 3x the position of the output_id in records, since each output_id has 3 embeddings (range_numeric, range_fuzzy, and range_language)
        with output_path.open("w", encoding="utf-8") as f:
            for position, output_id in enumerate(records):
                embedding_position = position * 3

                row = {
                    str(output_id): {
                        "range_numeric": embeddings[embedding_position].tolist(),
                        "range_fuzzy": embeddings[embedding_position + 1].tolist(),
                        "range_language": embeddings[embedding_position + 2].tolist(),
                    }
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        # handling for no range_language:
        # indices of embeddings are 2x the position of the output_id in records, since each output_id has 2 embeddings (range_numeric and range_fuzzy)
        with output_path.open("w", encoding="utf-8") as f:
            for position, output_id in enumerate(records):
                embedding_position = position * 2

                row = {
                    str(output_id): {
                        "range_numeric": embeddings[embedding_position].tolist(),
                        "range_fuzzy": embeddings[embedding_position + 1].tolist(),
                    }
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")



"""
Test embedding functionality.
"""
def test_embeddings(output_path: Path):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [
        "Nov 1935 to Sep 1937",
        "2005 to 2007",
        "early 1940s",
    ]

    embeddings = model.encode(texts, normalize_embeddings=True)
    # convert to dict and write to jsonl (key is the text converted to embedding, value is the embedding vector)
    #TODO - later will need to make this dict into range text vs fuzzy text keys/embeddings, or could use this embeddings jsonl as a lookup
    with output_path.open("w", encoding="utf-8") as f:
        for text, embedding in zip(texts, embeddings):
            row = {
                text: embedding.tolist()
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    #TODO add timestamps for own review
    if DEBUG_MODE:
        print(f"Embeddings written to: {output_path}")
        print(f"Embeddings shape: {embeddings.shape}")
    print(f"Total embeddings written: {len(embeddings)}")


def main():
    base_path = Path(__file__).resolve().parent
    output_dir = base_path / PROCESSED_DATA_PATH
    output_path = output_dir / EMBEDDING_DATA_FILE_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    if TEST_MODE:
        test_embeddings(output_path)
    else:
        write_embeddings(output_path)


if __name__ == "__main__":
    main()

