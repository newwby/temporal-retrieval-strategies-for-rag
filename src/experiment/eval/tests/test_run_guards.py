import sys

import pytest

from experiment.classes.class_query import ExperimentRunMetadata
from experiment.util import consts
from experiment.util.consts import HEAVY_RUN_QUERY_THRESHOLD, HEAVY_RUN_REEMBED_THRESHOLD, SMALL_DEFAULT_MODEL
from experiment.util.run_guards import assess_suite, warn_if_heavy

LARGE_MODEL = "all-mpnet-base-v2"


def _docs(n: int) -> list[dict]:
    return [{"id": f"doc{i}", "text": f"text {i}", "metadata": {}} for i in range(n)]


def _metadata(model_name: str = SMALL_DEFAULT_MODEL, phrasing: str = "source") -> ExperimentRunMetadata:
    return ExperimentRunMetadata(datestamp="d", timestamp="t", model_name=model_name, file_stem="stem", phrasing=phrasing)


@pytest.fixture(autouse=True)
def isolated_tracking_file(monkeypatch, tmp_path):
    monkeypatch.setattr("experiment.util.storage.TRACKING_FILE_PATH", tmp_path / "indexed_docs.json")


def test_baseline_does_not_fire():
    run_metadata_list = [_metadata()]
    summary = assess_suite("baseline", run_metadata_list, query_set=[None] * 10, docs=_docs(5), query_source="dev")
    assert summary is None


def test_fires_for_non_default_model():
    run_metadata_list = [_metadata(), _metadata(model_name=LARGE_MODEL)]
    summary = assess_suite("manifest_test", run_metadata_list, query_set=[None] * 10, docs=_docs(5), query_source="dev")
    assert summary is not None
    assert LARGE_MODEL in summary["models"]


def test_fires_for_run_count_times_query_count():
    run_metadata_list = [_metadata() for _ in range(4)]
    num_queries = (HEAVY_RUN_QUERY_THRESHOLD // 4) + 10
    summary = assess_suite("sweep", run_metadata_list, query_set=[None] * num_queries, docs=_docs(5), query_source="dev")
    assert summary is not None


def test_fires_for_large_reembed():
    run_metadata_list = [_metadata()]
    summary = assess_suite("baseline", run_metadata_list, query_set=[None] * 10, docs=_docs(HEAVY_RUN_REEMBED_THRESHOLD + 1), query_source="dev")
    assert summary is not None
    assert summary["docs_to_embed"] == HEAVY_RUN_REEMBED_THRESHOLD + 1


def test_fires_for_language_phrasing():
    run_metadata_list = [_metadata(phrasing="language")]
    summary = assess_suite("phrasing_test", run_metadata_list, query_set=[None] * 10, docs=_docs(5), query_source="dev")
    assert summary is not None
    assert summary["needs_llm"] is True


def test_yes_flag_skips_pause(monkeypatch):
    slept = []
    monkeypatch.setattr("experiment.util.run_guards.time.sleep", lambda seconds: slept.append(seconds))
    run_metadata_list = [_metadata(model_name=LARGE_MODEL)]

    warn_if_heavy("manifest_test", run_metadata_list, query_set=[None] * 10, docs=_docs(5), query_source="dev", skip_pause=True)
    assert slept == []

    warn_if_heavy("manifest_test", run_metadata_list, query_set=[None] * 10, docs=_docs(5), query_source="dev", skip_pause=False)
    assert slept == [consts.HEAVY_RUN_PAUSE_SECONDS]


def test_device_resolution_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.delenv("EXPERIMENT_DEVICE", raising=False)
    assert consts._resolve_device() == "cpu"


def test_device_resolution_respects_env_override(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_DEVICE", "cuda")
    assert consts._resolve_device() == "cuda"


def test_device_resolution_does_not_raise_when_torch_absent(monkeypatch):
    monkeypatch.delenv("EXPERIMENT_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert consts._resolve_device() == "cpu"
