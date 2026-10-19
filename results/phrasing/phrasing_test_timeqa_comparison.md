# Phrasing Test - EM / F1 vs TimeQA (easy split)

| System | EM | F1 |
| --- | --- | --- |
| phrasing_test_k1_all-minilm-l6-v2_phrasing-source | 0.3 | 2.7 |
| phrasing_test_k1_all-minilm-l6-v2_phrasing-numeric | 0.3 | 2.7 |
| phrasing_test_k1_all-minilm-l6-v2_phrasing-fuzzy | 0.3 | 2.5 |
| phrasing_test_k1_all-minilm-l6-v2_phrasing-language | 0.3 | 2.2 |
| phrasing_test_k1_all-minilm-l6-v2_phrasing-delete | 0.4 | 2.3 |
| phrasing_test_k3_all-minilm-l6-v2_phrasing-source | 0.3 | 2.7 |
| phrasing_test_k3_all-minilm-l6-v2_phrasing-numeric | 0.3 | 2.7 |
| phrasing_test_k3_all-minilm-l6-v2_phrasing-fuzzy | 0.3 | 2.5 |
| phrasing_test_k3_all-minilm-l6-v2_phrasing-language | 0.3 | 2.2 |
| phrasing_test_k3_all-minilm-l6-v2_phrasing-delete | 0.4 | 2.3 |
| phrasing_test_k5_all-minilm-l6-v2_phrasing-source | 0.3 | 2.7 |
| phrasing_test_k5_all-minilm-l6-v2_phrasing-numeric | 0.3 | 2.7 |
| phrasing_test_k5_all-minilm-l6-v2_phrasing-fuzzy | 0.3 | 2.5 |
| phrasing_test_k5_all-minilm-l6-v2_phrasing-language | 0.3 | 2.2 |
| phrasing_test_k5_all-minilm-l6-v2_phrasing-delete | 0.4 | 2.3 |
| TimeQA FiD baseline (easy) | 60.5 | 67.9 |
| TimeQA human performance (easy) | 89.0 | 93.3 |

predicted_answer here is the full top-1 retrieved passage text, not a generated short answer. Precision, and therefore F1 and exact match, are structurally suppressed relative to TimeQA's own answer-generation pipeline, which scores a short generated string against the gold answer. The gap between this project and the TimeQA reference rows is a measurement-methodology difference, not evidence of worse retrieval.
