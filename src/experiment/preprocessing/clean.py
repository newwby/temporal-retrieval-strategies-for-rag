from pathlib import Path
import shutil

from experiment.util.consts import CHROMA_DB_PATH, CORPUS_OUTPUT_DIR, QUERIES_OUTPUT_DIR, TRACKING_FILE_PATH

"""
wipes built files (corpus, queries, chroma_db/, data/indexing/) for a clean rebuild
clears everything rather than per config, avoids a mismatched corpus/queries pair
leaves data/timeqa/ (source datasets) alone
"""

__all__ = ["clean_built_data"]


def _clear_directory_contents(path: Path) -> list[Path]:
    removed = []
    if path.exists():
        for child in sorted(path.iterdir()):
            removed.append(child)
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    return removed


def clean_built_data() -> None:
    indexing_dir = TRACKING_FILE_PATH.parent

    targets = [CORPUS_OUTPUT_DIR, QUERIES_OUTPUT_DIR, CHROMA_DB_PATH, indexing_dir]

    print("Cleaning previously built data (source datasets under data/timeqa/ are never touched). Targets:")
    for target in targets:
        print(f"  - {target}")

    total_removed = 0
    for target in targets:
        removed = _clear_directory_contents(target)
        total_removed += len(removed)
        target.mkdir(parents=True, exist_ok=True)

    print(f"Clean complete - removed {total_removed} item(s) across {len(targets)} director{'y' if len(targets) == 1 else 'ies'}.")


if __name__ == "__main__":
    clean_built_data()
