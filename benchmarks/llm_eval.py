"""
LLM-judged evaluation with the OpenAI Decisions API.

1. Interpretability: word intrusion + 0-3 rating, TriTopic vs. BERTopic (default/tuned)
2. Assignment: held-out documents -> topics, Decisions API vs. centroid transform
3. Merging: over-segmented TriTopic (2x the class count) + suggest_merges/apply_merges

python benchmarks/llm_eval.py --pkg . --out llm_results.csv            (needs OPENAI_API_KEY)
python benchmarks/llm_eval.py --pkg . --out llm_results.csv --dry-run  (cost estimate, no API calls)
"""
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--datasets", default="20newsgroups,bbc_news,agnews,arxiv")
ap.add_argument("--seeds", default="42")
ap.add_argument("--parts", default="interpretability,assignment,merging")
ap.add_argument("--n_heldout", type=int, default=200)
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
sys.path.insert(0, args.pkg)

import numpy as np, pandas as pd
from sklearn.metrics import normalized_mutual_info_score as NMI

HERE = os.path.dirname(os.path.abspath(__file__))
# reuse loaders/embeddings/model runners of the head-to-head harness
_src = open(os.path.join(HERE, "compare_bertopic.py"), encoding="utf-8").read()
_argv, sys.argv = sys.argv, ["compare_bertopic.py", "--pkg", args.pkg, "--out", "unused.csv"]
_own_args = args
exec(_src.split("# ---------------- main loop")[0])
sys.argv, args = _argv, _own_args

from tritopic import TriTopic, TriTopicConfig
from tritopic.integrations import decisions as dec

MID_K = {"20newsgroups": 20, "bbc_news": 5, "agnews": 4, "arxiv": 10}  # = number of classes (arXiv: 11 -> 10)


class DryRunClient:
    """Counts requests and approximate input tokens instead of calling the API."""
    def __init__(self):
        self.n_requests, self.chars = 0, 0
    def decide_many(self, reqs):
        return [self.decide(i, q) for i, q in reqs]
    def decide(self, input, questions):
        import json
        self.n_requests += 1
        self.chars += len(input if isinstance(input, str) else json.dumps(input)) + len(json.dumps(questions))
        q = questions[0]
        if q["type"] == "choice":
            return {q["name"]: {"type": "choice", "name": q["name"], "choice": q["choices"][0]["value"], "confidence": 1.0}}
        if q["type"] == "score":
            return {q["name"]: {"type": "score", "name": q["name"], "score": 0.0}}
        return {q["name"]: {"type": "predicate", "name": q["name"], "probability": 0.0}}


client = DryRunClient() if args.dry_run else dec.DecisionsClient()


def save(row):
    # one file per part: the parts have different columns
    root, ext = os.path.splitext(args.out)
    path = f"{root}_{row['part']}{ext or '.csv'}"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


def bertopic_fit(docs, emb, k, seed, tuned):
    from bertopic import BERTopic
    from umap import UMAP
    kw = {}
    if tuned:
        from bertopic.vectorizers import ClassTfidfTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        kw = dict(vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2),
                  ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True))
    m = BERTopic(umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed),
                 nr_topics=k, calculate_probabilities=False, verbose=False, **kw)
    labels, _ = m.fit_transform(docs, embeddings=emb)
    tids = sorted(set(labels) - {-1})
    words = [[w for w, _ in (m.get_topic(t) or [])][:10] for t in tids]
    reps = [(m.get_representative_docs(t) or [])[:2] for t in tids]
    return words, reps


def tritopic_fit(docs, emb, k, seed):
    m = TriTopic(config=TriTopicConfig(verbose=False, random_state=seed), n_topics=k if k else "auto")
    m.fit(docs, embeddings=emb)
    return m


def majority_accuracy(train_labels, y_train, pred, y_true):
    """Map each topic to its majority class on the training docs, score predictions."""
    mapping = pd.Series(y_train).groupby(train_labels).agg(lambda s: s.value_counts().idxmax()).to_dict()
    mapped = np.array([mapping.get(p, -999) for p in pred])
    return float((mapped == y_true).mean())


parts = set(args.parts.split(","))
for ds in args.datasets.split(","):
    docs, y = load(ds, 0); emb = embed(ds, docs)
    k = MID_K[ds]
    for seed in map(int, args.seeds.split(",")):
        # ---------------- 1. interpretability ----------------
        if "interpretability" in parts:
            m = tritopic_fit(docs, emb, k, seed)
            topics = [t for t in m.topics_ if t.topic_id != -1]
            cands = {"TriTopic 2.4": ([t.keywords[:10] for t in topics],
                                      [[docs[i] for i in t.representative_docs[:2]] for t in topics])}
            for name, tuned in [("BERTopic (default)", False), ("BERTopic (tuned)", True)]:
                cands[name] = bertopic_fit(docs, emb, k, seed, tuned)
            for name, (words, reps) in cands.items():
                wi = dec.word_intrusion(words, client, random_state=seed)
                rating = dec.rate_topics(words, client)
                rating_docs = dec.rate_topics(words, client, representative_docs=reps)
                save(dict(part="interpretability", dataset=ds, seed=seed, model=name, k=k, n_topics=len(words),
                          intrusion_acc=wi.accuracy, intrusion_prob=wi.intruder_probability,
                          rating=float(np.nanmean(rating)), rating_with_docs=float(np.nanmean(rating_docs))))

        # ---------------- 2. assignment ----------------
        if "assignment" in parts:
            rng = np.random.default_rng(seed)
            held = rng.choice(len(docs), args.n_heldout, replace=False)
            train = np.setdiff1d(np.arange(len(docs)), held)
            m = tritopic_fit([docs[i] for i in train], emb[train], k, seed)
            # centroid transform without re-encoding
            from sklearn.metrics.pairwise import cosine_similarity
            tids = np.array([t.topic_id for t in m.topics_ if t.topic_id != -1])
            centroid_pred = tids[cosine_similarity(emb[held], m.topic_embeddings_).argmax(axis=1)]
            t0 = time.perf_counter()
            llm_pred, conf = dec.assign_documents(m, [docs[i] for i in held], client, embeddings=emb[held])
            dt = time.perf_counter() - t0
            abstain_pred, _ = dec.assign_documents(m, [docs[i] for i in held], client, embeddings=emb[held],
                                                   allow_other=True)
            for method, pred in [("centroid", centroid_pred), ("decisions", llm_pred),
                                 ("decisions (may abstain)", abstain_pred)]:
                save(dict(part="assignment", dataset=ds, seed=seed, model=method, k=k, n=len(held),
                          accuracy=majority_accuracy(m.labels_, y[train], pred, y[held]),
                          unassigned=float((pred == -1).mean()), time=dt if method == "decisions" else 0.0))

        # ---------------- 3. merging ----------------
        if "merging" in parts:
            m = tritopic_fit(docs, emb, 2 * k, seed)
            before_n, before = len(set(m.labels_) - {-1}), NMI(y, m.labels_)
            merges = dec.suggest_merges(m, client, n_pairs=3 * k)
            dec.apply_merges(m, merges)
            save(dict(part="merging", dataset=ds, seed=seed, model="TriTopic 2.4", k=2 * k,
                      n_topics_before=before_n, n_topics_after=len(set(m.labels_) - {-1}),
                      nmi_before=before, nmi_after=NMI(y, m.labels_), merges=len(merges)))

if args.dry_run:
    tokens = client.chars / 4
    print(f"\nDRY RUN: {client.n_requests} requests, ~{tokens / 1e6:.2f}M input tokens, "
          f"~${tokens / 1e6 * 0.10:.3f} at $0.10 / 1M input tokens")
