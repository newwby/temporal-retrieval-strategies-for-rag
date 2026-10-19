import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from consts import COMPARISON_DATA_FILE_NAME, EMBEDDING_DATA_FILE_NAME, EXTRACTED_DATA_FILE_NAME, HISTOGRAM_OUTPUT_DIRECTORY, PROCESSED_DATA_PATH

"""
embed_reporting.py
This module handles the reporting of embedding similarities.
"""

def _load_embeddings(path: Path):
    output_ids = []
    range_embeddings = []
    fuzzy_embeddings = []
    language_embeddings = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
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

    # Preserve row alignment while marking missing language embeddings
    language_embedding_array = np.full(
        range_embeddings.shape,
        np.nan,
        dtype=np.float32,
    )

    for index, embedding in enumerate(language_embeddings):
        if embedding is not None:
            language_embedding_array[index] = np.asarray(
                embedding,
                dtype=np.float32,
            )

    return (
        output_ids,
        range_embeddings,
        fuzzy_embeddings,
        language_embedding_array,
    )


def _compare_pairs(input_path_path: Path, output_path_path: Path, report: bool = False):
    output_path_ids, range_embeddings, fuzzy_embeddings, language_embeddings = _load_embeddings(input_path_path)

    # Dot product equals cosine similarity because vectors are already normalised
    numeric_fuzzy_similarities = np.einsum(
        "ij,ij->i",
        range_embeddings,
        fuzzy_embeddings,
    )

    numeric_language_similarities = np.full(
        len(output_path_ids),
        np.nan,
        dtype=np.float32,
    )

    fuzzy_language_similarities = np.full(
        len(output_path_ids),
        np.nan,
        dtype=np.float32,
    )

    # Only compare rows where a language embedding is present
    language_mask = ~np.isnan(language_embeddings).any(axis=1)

    if language_mask.any():
        numeric_language_similarities[language_mask] = np.einsum(
            "ij,ij->i",
            range_embeddings[language_mask],
            language_embeddings[language_mask],
        )

        fuzzy_language_similarities[language_mask] = np.einsum(
            "ij,ij->i",
            fuzzy_embeddings[language_mask],
            language_embeddings[language_mask],
        )

    results = pd.DataFrame({
        "output_id": output_path_ids,
        "cosine_similarity": numeric_fuzzy_similarities,
        "cosine_distance": 1.0 - numeric_fuzzy_similarities,
        "numeric_language_cosine_similarity": numeric_language_similarities,
        "numeric_language_cosine_distance": 1.0 - numeric_language_similarities,
        "fuzzy_language_cosine_similarity": fuzzy_language_similarities,
        "fuzzy_language_cosine_distance": 1.0 - fuzzy_language_similarities,
    })

    # add to the authorative data file
    _update_range_data(results)

    results.to_csv(output_path_path, index=False)

    if report:
        print(f"Results saved to: {output_path_path}")

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

        _display_data(results, output_path_path.parent)


"""
Adds human readadble descriptor of cosine similarity (for the authorative data file)
These are only relative similarity bands
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
Visualises data as histograms, using bounds to show similarity bands that correspond with the human readable descriptors
"""
def _display_data(results: pd.DataFrame, output_dir: Path | None = None):
    comparisons = [
        ("cosine_similarity", "Numeric vs Fuzzy", "numeric_fuzzy"),
        (
            "numeric_language_cosine_similarity",
            "Numeric vs Language",
            "numeric_language",
        ),
        (
            "fuzzy_language_cosine_similarity",
            "Fuzzy vs Language",
            "fuzzy_language",
        ),
    ]

    if output_dir is not None:
        histogram_dir = output_dir / HISTOGRAM_OUTPUT_DIRECTORY
        histogram_dir.mkdir(parents=True, exist_ok=True)

        # save under timestamped filenames to avoid overwriting previous reports
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    else:
        histogram_dir = None
        timestamp = None
        print("No output directory provided, histograms will not be saved to disk.")

    for column, comparison_name, filename in comparisons:
        similarities = results[column].dropna()

        if similarities.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(similarities, bins=20)

        for boundary in [0.30, 0.45, 0.60, 0.75, 0.90]:
            ax.axvline(
                boundary,
                linestyle="--",
                linewidth=1,
                label=f"{boundary:.2f}",
            )

        ax.set_title(
            f"Distribution of {comparison_name} Cosine Similarity Scores"
        )
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Number of pairs")
        ax.set_xlim(-1.0, 1.0)
        ax.legend(title="Band boundaries")
        fig.tight_layout()

        if histogram_dir is not None:
            fig.savefig(
                histogram_dir / f"{timestamp}_{filename}.png",
                dpi=300,
            )

    plt.show()


"""
Applies similarities and distance to the authorative data file
"""
def _update_range_data(cosine_results: pd.DataFrame):
    range_data_path = Path(__file__).resolve().parent / PROCESSED_DATA_PATH / EXTRACTED_DATA_FILE_NAME

    jsonl_range_data = {}

    # load data & extract the range_numeric and range_fuzzy for embedding whilst keying by output_path_id to match back
    with range_data_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            obj = json.loads(line)
            jsonl_range_data[int(obj["output_id"])] = obj

    # matching by source ID, apply the cosine similarities and distances to the authorative data file
    for _, row in cosine_results.iterrows():
        output_id = int(row["output_id"])

        if output_id not in jsonl_range_data:
            continue

        entry = jsonl_range_data[output_id]

        entry["cosine_similarity"] = float(row["cosine_similarity"])
        entry["cosine_distance"] = float(row["cosine_distance"])
        entry["similarity"] = _describe_similarity(
            float(row["cosine_similarity"])
        )

        optional_comparisons = [
            (
                "numeric_language",
                "numeric_language_cosine_similarity",
                "numeric_language_cosine_distance",
            ),
            (
                "fuzzy_language",
                "fuzzy_language_cosine_similarity",
                "fuzzy_language_cosine_distance",
            ),
        ]

        for prefix, similarity_column, distance_column in optional_comparisons:
            similarity = row[similarity_column]

            if pd.notna(similarity):
                entry[similarity_column] = float(similarity)
                entry[distance_column] = float(row[distance_column])
                entry[f"{prefix}_similarity"] = _describe_similarity(
                    float(similarity)
                )
            else:
                entry.pop(similarity_column, None)
                entry.pop(distance_column, None)
                entry.pop(f"{prefix}_similarity", None)

    # write back to disk, overwriting the original file with the updated data
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