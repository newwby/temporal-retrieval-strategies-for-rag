# Final experiment results

Made by `run_final_suites.py`. Each suite folder has the figures, a CSV for each figure and a
markdown table. Rebuild with `python run_final_suites.py -s <suite>`, or `--analysis-only` for the
post-hoc scripts.

Corpus is TimeQA `dev` (32,678 paragraphs, 2,674 scored queries) and the retriever is FAISS over
`all-MiniLM-L6-v2`, unless noted. Every control arm gets 71.4% Hit@30 / 0.193 MRR.

## Reading the figures

- Chart titles use the manifest suite name, plus the model in brackets when every arm shares one.
  Bar labels then drop both and only show what changes between arms (`k3_temporalmode-none`).
  `model_sweep` keeps the model in the label since that is what varies.
- Metrics shown are Hit@K and MRR. Precision@K and Recall@K are only shown for `multi_gold_train`,
  where the multi-gold subset makes them useful. EM / F1 / chrF1 aren't plotted or in the per-suite
  CSVs, as they compare a whole passage to a short answer and barely change between arms. They are
  still in `_metadata.jsonl` and in the TimeQA comparison files.
- The k-sweep x axis stops at the deepest k the run retrieved, so a k=1/3/5 suite has three points.
- The latency figure only has an index build panel when a build cost was paid, and that cost is
  charged to every arm that used the index. A filtered index is built once per session and cached,
  so the first arm takes the whole cost and the rest would show zero. Inherited costs are hatched
  and lighter, and appear in the CSV as `index_build_effective_seconds` with
  `index_build_attributed=True`, next to the measured `index_build_mean_seconds`. `rerank` arms are
  never charged, they use the full index and only reorder. Suites that never filter get a single
  panel noting no build cost.

| file | what it is |
|---|---|
| `<suite>.png` / `.csv` | metric bars per arm |
| `<suite>_latency.png` / `.csv` | retrieval time per query, plus index build cost (measured and inherited) |
| `<suite>_k_sweep.png` / `.csv` | metrics across k, by cutting each run's stored ranked list |
| `<suite>_table.md` / `_table.png` | the same numbers as a table |
| `_metadata.jsonl` | one row per run, every axis value plus the aggregate metrics |

## Suites

### `phrasing/` - query date phrasing, k=1/3/5
How the date is written in the query. `source` is the unchanged control.

| arm (k=5) | Hit@5 | MRR@5 |
|---|---|---|
| source | 40.8% | 0.161 |
| numeric | 39.7% | 0.158 |
| fuzzy | 37.7% | 0.145 |
| delete | 34.4% | 0.121 |
| language | 32.7% | 0.118 |

Writing the date out in words (`language`) costs more than removing it (`delete`).

Also here: `chrf_divergence.csv` / `.json` (character F1 of the top 1 passage against gold, mean
chrF 0.0799, hit rate 20.7%, false-close rate 15.4%; the numeric/textual split doesn't work on this
dataset as TimeQA gold answers are entity names, never bare numbers) and
`phrasing_test_timeqa_comparison.md` (EM/F1 against TimeQA's published easy split figures).

### `temporal_filtering_k1-3-5/` - exclude vs rerank at the same k
| k | none | exclude | rerank |
|---|---|---|---|
| 1 | 2.9% / 0.029 | 24.8% / 0.248 | 2.9% / 0.029 |
| 3 | 26.3% / 0.127 | 50.5% / 0.364 | 26.3% / 0.217 |
| 5 | 40.8% / 0.161 | 57.5% / 0.380 | 40.8% / 0.297 |

`exclude` filters the index before retrieval, so it is a real k=1/3/5 result. `rerank` reorders
what was retrieved, so at low k it has little to work with. At k=1 reordering one item does
nothing, so it matches `none`.

### `temporal_filtering_k30/` - the same three arms at k=30
| mode | Hit@30 | MRR@30 |
|---|---|---|
| none | 71.4% | 0.193 |
| exclude | 67.7% | 0.390 |
| rerank | 71.4% | 0.393 |

Here rerank overtakes exclude: the same MRR gain (+104% over `none`) with no recall loss, while
`exclude` loses 3.7pp of Hit@30.

Also here, EM / F1 / BERTScore (`*_bertscore.csv` / `.md`, roberta-large, n=2,674 per arm):

| mode | EM | Answer F1 | BERTScore P | R | F1 |
|---|---|---|---|---|---|
| none | 0.0034 | 0.0274 | 0.8181 | 0.8485 | 0.8321 |
| exclude | 0.0000 | 0.0533 | 0.7736 | 0.8539 | 0.8111 |
| rerank | 0.0000 | 0.0533 | 0.7737 | 0.8539 | 0.8112 |

`predicted_answer` is the whole top 1 passage, not a generated span, so EM is about 0 and F1 is
lower than TimeQA's own numbers. That comes from how it's measured, not worse retrieval.

### `model_sweep/` and `bm25/` - retrieval model comparison, k=1/3/5
| retriever (k=5) | Hit@5 | MRR@5 | retrieval s/query |
|---|---|---|---|
| BGE-base | 53.3% | 0.207 | 0.0196 |
| MPNet | 47.8% | 0.237 | 0.0210 |
| MiniLM | 40.8% | 0.161 | 0.0120 |
| BM25 | 36.2% | 0.154 | 0.0562 |

Both larger dense models beat MiniLM and BM25 at every k. BGE ran without its query instruction
prefix, so it isn't tuned. BM25 is its own suite because the harness builds a session per model
name and BM25 doesn't use one.

### `entity_upper_bound/` - oracle, k=30
| arm | Hit@30 | MRR@30 |
|---|---|---|
| entity filter on | 93.0% | 0.222 |
| entity filter off | 71.4% | 0.193 |

This is an upper bound, not a retrieval condition, as it already knows the gold article. It
shouldn't be compared with the other arms or combined with the temporal filter.

### `multi_gold_train/` - precision/recall on the multi-gold subset, k=1/3/5
TimeQA `train` (149,771 paragraphs, 12,532 scored queries), cut down to the 1,029 queries with more
than one gold paragraph.

| k | Hit@K | Precision@K | Recall@K | MRR@K |
|---|---|---|---|---|
| 1 | 3.6% | 3.6% | 2.7% | 0.036 |
| 3 | 28.2% | 9.8% | 21.2% | 0.140 |
| 5 | 43.1% | 9.6% | 33.1% | 0.174 |

The figure, CSV and table use the multi-gold numbers (n=1,029). `_metadata.jsonl` has the
unfiltered totals over all 12,532 scored queries (3.0% / 24.5% / 37.7%, MRR 0.030 / 0.120 / 0.150),
since the filter is applied when plotting, so the two won't match.

There is no k-sweep figure for this suite, it would use the unfiltered set and contradict the table.

## Not run

`encoding_grid` and `question_form_test` weren't part of this pass. They are still in
`manifest.json` at `k: 30` with their 2026_08_25 runs, so their k differs from everything above.
