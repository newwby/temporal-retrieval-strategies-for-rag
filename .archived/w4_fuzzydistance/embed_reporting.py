import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from consts import COMPARISON_DATA_FILE_NAME, EMBEDDING_DATA_FILE_NAME, EXTRACTED_DATA_FILE_NAME, HISTOGRAM_OUTPUT_DIRECTORY, PROCESSED_DATA_PATH

"""
compares embedding pairs, reports and plots similarity scores
"""


def _load_embeddings(path: Path):
    output_ids = []
    range_embeddings = []
    fuzzy_embeddings = []
    language_embeddings = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            output_id, embeddings = next(iter(row.items()))
            output_ids.append(int(output_id))
            range_embeddings.append(embeddings["range_numeric"])
            fuzzy_embeddings.append(embeddings["range_fuzzy"])
            language_embeddings.append(embeddings.get("range_language"))

    range_embeddings = np.asarray(range_embeddings, dtype=np.float32)
    fuzzy_embeddings = np.asarray(fuzzy_embeddings, dtype=np.float32)

    # nan rows where no language embedding, keeps rows aligned
    language_array = np.full(range_embeddings.shape, np.nan, dtype=np.float32)
    for i, embedding in enumerate(language_embeddings):
        if embedding is not None:
            language_array[i] = np.asarray(embedding, dtype=np.float32)

    return output_ids, range_embeddings, fuzzy_embeddings, language_array


def _compare_pairs(input_path: Path, output_path: Path, report: bool = False):
    output_ids, range_embeddings, fuzzy_embeddings, language_embeddings = _load_embeddings(input_path)

    # vectors already normalised so dot product = cosine similarity
    numeric_fuzzy = np.einsum("ij,ij->i", range_embeddings, fuzzy_embeddings)
    numeric_language = np.full(len(output_ids), np.nan, dtype=np.float32)
    fuzzy_language = np.full(len(output_ids), np.nan, dtype=np.float32)

    # only compare rows with a language embedding
    has_language = ~np.isnan(language_embeddings).any(axis=1)
    if has_language.any():
        numeric_language[has_language] = np.einsum("ij,ij->i", range_embeddings[has_language], language_embeddings[has_language])
        fuzzy_language[has_language] = np.einsum("ij,ij->i", fuzzy_embeddings[has_language], language_embeddings[has_language])

    results = pd.DataFrame({
        "output_id": output_ids,
        "cosine_similarity": numeric_fuzzy,
        "cosine_distance": 1.0 - numeric_fuzzy,
        "numeric_language_cosine_similarity": numeric_language,
        "numeric_language_cosine_distance": 1.0 - numeric_language,
        "fuzzy_language_cosine_similarity": fuzzy_language,
        "fuzzy_language_cosine_distance": 1.0 - fuzzy_language,
    })

    # add to the authorative data file
    _update_range_data(results)
    results.to_csv(output_path, index=False)

    if report:
        print(f"Results saved to: {output_path}")
        comparisons = [
            ("Numeric vs fuzzy", "cosine_similarity"),
            ("Numeric vs language", "numeric_language_cosine_similarity"),
            ("Fuzzy vs language", "fuzzy_language_cosine_similarity"),
        ]
        for name, column in comparisons:
            similarities = results[column].dropna()
            if not similarities.empty:
                print(f"\n{name}:")
                print(similarities.describe())
        _display_data(results, output_path.parent)


"""
human readable label for a cosine similarity (for the authorative data file)
bands are relative only
"""
def _describe_similarity(similarity: float) -> str:
    if similarity >= 0.90:
        return "very strong"
    if similarity >= 0.75:
        return "strong"
    if similarity >= 0.60:
        return "moderate"
    if similarity >= 0.45:
        return "weak"
    if similarity >= 0.30:
        return "very weak"
    if similarity >= 0.00:
        return "near nil"
    return "NOT SIMILAR"


"""
histograms of similarity scores, dashed lines mark the bands used in _describe_similarity
"""
def _display_data(results: pd.DataFrame, output_dir: Path | None = None):
    comparisons = [
        ("cosine_similarity", "Numeric vs Fuzzy", "numeric_fuzzy"),
        ("numeric_language_cosine_similarity", "Numeric vs Language", "numeric_language"),
        ("fuzzy_language_cosine_similarity", "Fuzzy vs Language", "fuzzy_language"),
    ]

    histogram_dir = None
    timestamp = None
    if output_dir is not None:
        histogram_dir = output_dir / HISTOGRAM_OUTPUT_DIRECTORY
        histogram_dir.mkdir(parents=True, exist_ok=True)
        # timestamped names so old reports aren't overwritten
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    else:
        print("No output directory provided, histograms will not be saved to disk.")

    for column, comparison_name, filename in comparisons:
        similarities = results[column].dropna()
        if similarities.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(similarities, bins=20)
        for boundary in [0.30, 0.45, 0.60, 0.75, 0.90]:
            ax.axvline(boundary, linestyle="--", linewidth=1, label=f"{boundary:.2f}")
        ax.set_title(f"Distribution of {comparison_name} Cosine Similarity Scores")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Number of pairs")
        ax.set_xlim(-1.0, 1.0)
        ax.legend(title="Band boundaries")
        fig.tight_layout()

        if histogram_dir is not None:
            fig.savefig(histogram_dir / f"{timestamp}_{filename}.png", dpi=300)

    plt.show()


"""
adds similarity scores to the authorative data file
"""
def _update_range_data(cosine_results: pd.DataFrame):
    range_data_path = Path(__file__).resolve().parent / PROCESSED_DATA_PATH / EXTRACTED_DATA_FILE_NAME
    jsonl_range_data = {}

    # load entries keyed by output_id
    with range_data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            jsonl_range_data[int(obj["output_id"])] = obj

    optional_comparisons = [
        ("numeric_language", "numeric_language_cosine_similarity", "numeric_language_cosine_distance"),
        ("fuzzy_language", "fuzzy_language_cosine_similarity", "fuzzy_language_cosine_distance"),
    ]

    # match by output_id and add scores
    for _, row in cosine_results.iterrows():
        output_id = int(row["output_id"])
        if output_id not in jsonl_range_data:
            continue

        entry = jsonl_range_data[output_id]
        entry["cosine_similarity"] = float(row["cosine_similarity"])
        entry["cosine_distance"] = float(row["cosine_distance"])
        entry["similarity"] = _describe_similarity(float(row["cosine_similarity"]))

        for prefix, similarity_column, distance_column in optional_comparisons:
            similarity = row[similarity_column]
            if pd.notna(similarity):
                entry[similarity_column] = float(similarity)
                entry[distance_column] = float(row[distance_column])
                entry[f"{prefix}_similarity"] = _describe_similarity(float(similarity))
            else:
                entry.pop(similarity_column, None)
                entry.pop(distance_column, None)
                entry.pop(f"{prefix}_similarity", None)

    # overwrite file with updated entries
    with range_data_path.open("w", encoding="utf-8") as f:
        for obj in jsonl_range_data.values():
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    base_path = Path(__file__).resolve().parent
    input_path = base_path / PROCESSED_DATA_PATH / EMBEDDING_DATA_FILE_NAME
    output_path = base_path / PROCESSED_DATA_PATH / COMPARISON_DATA_FILE_NAME
    _compare_pairs(input_path, output_path, report=True)


if __name__ == "__main__":
    main()
