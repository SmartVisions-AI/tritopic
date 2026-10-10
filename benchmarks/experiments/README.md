# Experiments behind TriTopic 2.4 (archived as run)

These scripts document how the 2.4 design decisions were made (October 2026). They were run from a
temporary working directory, so paths inside them (`h2h.py`, `h2h_cache/`, absolute Dropbox paths) must be
adjusted before re-running. The maintained, reusable tools are one level up:
`benchmarks/compare_bertopic.py`, `summarize.py`, `results_to_markdown.py`, `llm_eval.py`.
Raw results live in `benchmark_outputs/2.4/` (CSV files are not in git).

Rule used throughout: **decide on the dev splits** (20NG test, BBC test, AG News train sample, arXiv
validation), **evaluate once on the paper splits** (the datasets/sizes of `run_benchmark.py`).

| Script | Question | Result | Decision |
|---|---|---|---|
| `stages.py`, `variants.py` | Where does fit time go? Which resolution? (2.2 code, 20 Newsgroups) | UMAP.transform per iteration, dense consensus, repeated lexical graph dominated; resolution 1.0 gave 60 topics for 6 classes | Refine in reduced space, edge consensus, cache lexical graph, resolution 0.3 |
| `kw_exp.py` | Keyword scoring variants on fixed clusterings | tf*IDF 0.210 strict NPMI, BERTopic formula 0.193, coverage*IDF 0.370 (boilerplate), geometric mean 0.305 | Coverage-weighted c-TF-IDF |
| `auto_exp.py`, `auto_exp2.py` | How to pick the resolution in auto mode | Coherence argmax best but collapses to 2 topics occasionally; guard "largest topic <= 50%" + 5% tolerance: NMI 0.594 / ARI 0.474 vs. 0.565 / 0.415 fixed | `auto_resolution` with `auto_resolution_max_share=0.5`, tolerance 0.05 |
| dev ablation (`compare_bertopic.py --datasets *_dev --cfg ...`) | Graph/UMAP/refinement variants | All within seed noise except no lexical view (-0.055 NMI) and n_neighbors=10 | Keep defaults |
| `assign_exp.py` | Why is LLM assignment worse than centroids? | The "other" option caused 35% abstentions; without it + 2 examples: 0.631 vs. 0.620 | `assign_documents(allow_other=False, n_examples=2)`; `reduce_outliers` keeps "other" |
| `merge_exp.py`, `merge_exp2.py` | Is the LLM merge judgement informative? Prompt variants | AUC 0.63-0.75 vs. cosine 0.61-0.65; very conservative; prompt "winner" on paper splits did not replicate on dev | Keep strict prompt; document as safe clean-up, not a quality lever |
| `reduce_exp.py` | reduce_topics(2k -> k) methods | Current (size penalty) 0.492 NMI dev / 0.542 paper; no penalty 0.632 / 0.591; graph agglomeration worse | `reduce_topics(size_penalty=0.0)` default |
| `make_report.py`, `gen_bench_md.py` | Word reports and README tables | | `benchmark_outputs/2.4/*.docx`, README "Benchmarks" |
| `interp_exp.py`, `interp_heldout.py` | Which LLM interprets topics best; does `refine()` help? | gpt-6-luna found 9 of 15 mixed topics, refine NMI 0.614 -> 0.654 (dev); first version hurt fine-grained BBC on held-out data | Default model gpt-6-luna; add safeguards |
| `refine_diag.py` | Why did refine reject the BBC politics/business split? | Farthest-point sampling gave 1 politics vs. 6 business examples; NPMI rejected the correct split | Proportional (k-means) example sampling |
| `accept_exp.py` | Which acceptance rule for splits (same LLM verdicts)? | none +0.044 (worst -0.017), consistency +0.038 (worst -0.001), coherence +0.033 (worst -0.021) | `accept="consistency"` |
| `testrun2.py` | BBC demo with interpreter | Politics/business and football/rugby split; NMI 0.742 -> 0.781 | `examples/bbc_demo/index.html` |

