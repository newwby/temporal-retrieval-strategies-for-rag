import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from consts import DEBUG_MODE, EMBEDDING_DATA_FILE_NAME, EXTRACTED_DATA_FILE_NAME, PROCESSED_DATA_PATH, UNDEFINED_KEY

"""
embeds range data with a sentence transformer model
"""

# enable to test with a few sample texts (faster than full range data)
TEST_MODE = False


def write_embeddings(output_path: Path):
    has_language_embeddings = None
    records = []
    texts = []

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    range_data_path = Path(__file__).resolve().parent / PROCESSED_DATA_PATH / EXTRACTED_DATA_FILE_NAME

    # load range_numeric & range_fuzzy texts, keep output_id to match back
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
            row_has_language = obj.get("range_language") not in (None, UNDEFINED_KEY)
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

    # order kept so embeddings match back to output_id
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    # 3 embeddings per output_id with range_language, else 2
    if has_language_embeddings:
        # with range_language: numeric, fuzzy, language at position * 3
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
        # without range_language: numeric, fuzzy at position * 2
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
test embedding with sample texts
"""
def test_embeddings(output_path: Path):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [
        "Nov 1935 to Sep 1937",
        "2005 to 2007",
        "early 1940s",
    ]

    embeddings = model.encode(texts, normalize_embeddings=True)
    # write each text and its vector as a jsonl row
    #TODO split into range vs fuzzy keys, or use this file as a lookup
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

