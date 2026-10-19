import json
from pathlib import Path

"""
loads build_manifest.json, which sets what gets built (which raw timeqa files make one
corpus + queries build). eval/manifest.json sets what gets evaluated

each top level key is a config name (e.g. "dev", "train") with either a "sources" list of
{"annotated_path", "easy_path"} pairs, or a "directory" to scan by file name

don't add annotated_test.json/test_easy.json/test_hard.json, that's the held out test split
human_annotated_* files use a different schema (the validation check rejects them)
"""

BUILD_MANIFEST_PATH = Path(__file__).parent / "build_manifest.json"

DEFAULT_BUILD_CONFIG = "dev"

__all__ = ["BUILD_MANIFEST_PATH", "DEFAULT_BUILD_CONFIG", "load_build_manifest", "resolve_build_sources"]


def load_build_manifest(path: Path = BUILD_MANIFEST_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest is None:
        raise ValueError(f"Failed to load build manifest from {path}, cannot proceed with the build pipeline")
    return manifest


"""
for each annotated_<split>.json, finds the matching <split>.easy.json or <split>.easy.json.gzip
fits dev/train naming, timeqa isn't consistent (e.g. test_easy.json) so use "sources" for those
skips with a warning if there's no match, rather than failing the whole config
"""
def _scan_directory_for_sources(directory: Path) -> list[dict]:
    sources = []
    for annotated_path in sorted(directory.glob("annotated_*.json")):
        split = annotated_path.stem[len("annotated_"):]
        easy_json = directory / f"{split}.easy.json"
        easy_gzip = directory / f"{split}.easy.json.gzip"
        if easy_json.exists():
            easy_path = easy_json
        elif easy_gzip.exists():
            easy_path = easy_gzip
        else:
            print(
                f"Warning: no companion easy-question file found for {annotated_path.name} "
                f"(looked for {easy_json.name} or {easy_gzip.name} in {directory}) - skipping this source"
            )
            continue
        sources.append({"annotated_path": str(annotated_path), "easy_path": str(easy_path)})
    return sources


def resolve_build_sources(config_name: str, manifest: dict) -> list[dict]:
    if config_name not in manifest:
        available = sorted(manifest.keys())
        raise KeyError(f"Unknown build config {config_name!r} in {BUILD_MANIFEST_PATH}. Available configs: {available}")

    entry = manifest[config_name]
    if "sources" in entry:
        sources = entry["sources"]
    elif "directory" in entry:
        sources = _scan_directory_for_sources(Path(entry["directory"]))
    else:
        raise ValueError(f"Build config {config_name!r} in {BUILD_MANIFEST_PATH} must define either 'sources' or 'directory'")

    if not sources:
        raise ValueError(
            f"Build config {config_name!r} resolved to zero usable sources - check its 'sources'/'directory' "
            f"entry in {BUILD_MANIFEST_PATH}"
        )

    return sources
