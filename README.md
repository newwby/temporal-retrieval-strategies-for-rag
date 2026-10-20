# Bridging Human and Machine Time: Temporal Strategies for Natural Language Retrieval

How can fuzzy human time queries such as 'a few weeks ago' be retrieved effectively? This project looks at the mismatch between relative time language (human) and timestamp storage (computer).

Undergraduate Research Opportunity Scheme 2026 (Semester C) project by Daniel Newby, supervised by Heriberto Cuayahuitl Portilla.

## Goals

Compare temporal retrieval strategies to see which best supports:
- Accurate retrieval
- Ambiguity handling
- Latency (time cost)

## Setup

```
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pulls the CUDA 13.0 build of torch, change the `cu130` suffix to match your driver.

The experiment uses the [TimeQA](https://github.com/wenhuchen/Time-Sensitive-QA) dataset (BSD-3-Clause). It isn't included here, clone it into `data/timeqa/` so the files sit at `data/timeqa/dataset/`.

## Running the experiment

Run from the repo root.

```
python -m experiment.main -b dev        # build the corpus and queries, then verify
python -m experiment.main -s <suite>    # run a suite from manifest.json (baseline if left out)
python -m experiment.main -c            # clear built data for a clean rebuild
```

`run_final_suites.py` reruns the suites used for the final results and copies their outputs into `results/`.

## Experiment suites

Suites are defined in `src/experiment/manifest.json`. These fields can be a single value or a list, every combination of the lists is run:

| field | values | what it varies |
|---|---|---|
| `model_name` | any sentence-transformers name | the encoder |
| `k` | int | retrieval depth |
| `retriever` | `faiss`, `bm25` | retrieval method |
| `entity_filter` | bool | oracle entity restriction, an upper bound not a comparable arm |
| `temporal_filter` | `none`, `exclude`, `rerank` | how the query's time window affects retrieval |
| `corpus_encoding` | `source`, `numeric`, `fuzzy`, `language`, `delete` | how the corpus writes its dates before embedding |
| `question_form` | `timeqa_source`, `timeqa_alt`, `declarative`, `keyword`, `time_first`, `no_relation` | how the question is structured |
| `phrasing` | `source`, `numeric`, `fuzzy`, `language`, `delete` | how the query's date is worded |

Other suite fields: `model_aliases` (maps a model name to a file safe name), `query_source`, `metrics`, `gold_filter`.

## Results

Final figures, CSVs and tables are in [`results/`](results/README.md).

## Licence

MIT, see [LICENSE](LICENSE).
