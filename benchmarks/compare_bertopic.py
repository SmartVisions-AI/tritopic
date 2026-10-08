"""
Head-to-head benchmark TriTopic vs BERTopic (same embeddings for both).

Datasets/sizes/topic counts follow run_benchmark.py (paper splits); "*_dev"
datasets are disjoint splits used for tuning.  Results are appended to --out
(re-running resumes).

python benchmarks/compare_bertopic.py --pkg . --models tritopic,bertopic --out results.csv
              [--datasets 20newsgroups,bbc_news,agnews,arxiv] [--seeds 42,123,456]
              [--ks paper|auto|paper+auto] [--cfg key=val,...] [--tag name]
"""
import argparse, os, sys, time, math, re, random, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True)
ap.add_argument("--models", default="tritopic,bertopic")
ap.add_argument("--out", required=True)
ap.add_argument("--datasets", default="20newsgroups,bbc_news,agnews,arxiv")
ap.add_argument("--seeds", default="42,123,456")
ap.add_argument("--ks", default="paper+auto")
ap.add_argument("--cfg", default="")
ap.add_argument("--tag", default="")
ap.add_argument("--max_docs", type=int, default=0, help="override dataset size (0 = paper sizes)")
args = ap.parse_args()
sys.path.insert(0, args.pkg)

import numpy as np, pandas as pd
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")  # embeddings (*.npy, git-ignored); os.makedirs(CACHE, exist_ok=True)

# ---------------- datasets (same as run_benchmark.py) ----------------
def load(name, max_docs):
    if name == "20newsgroups":
        from sklearn.datasets import fetch_20newsgroups
        d = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
        n = max_docs or 2000
        return [x.replace("\n", " ").strip() for x in d.data][:n], np.array(d.target[:n])
    if name == "20newsgroups_full":
        from sklearn.datasets import fetch_20newsgroups
        d = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        keep = [i for i, x in enumerate(d.data) if len(x.strip()) >= 50]
        return [d.data[i].replace("\n", " ").strip() for i in keep], np.array(d.target)[keep]
    from datasets import load_dataset
    if name == "bbc_news":
        ds = load_dataset("SetFit/bbc-news", split="train"); n = max_docs or 1225
        return [x.replace("\n", " ").strip() for x in ds["text"]][:n], np.array(ds["label"][:n])
    if name == "agnews":
        ds = load_dataset("ag_news", split="test"); n = max_docs or 2000
        idx = list(range(len(ds))); random.seed(42); random.shuffle(idx); idx = idx[:n]
        return [ds[i]["text"].replace("\n", " ").strip() for i in idx], np.array([ds[i]["label"] for i in idx])
    if name == "arxiv":
        ds = load_dataset("ccdv/arxiv-classification", split="test"); n = max_docs or 2000
        return [x.replace("\n", " ").strip() for x in ds["text"]][:n], np.array(ds["label"][:n])
    # ---- dev splits (disjoint from the paper evaluation splits) ----
    if name == "20newsgroups_dev":
        from sklearn.datasets import fetch_20newsgroups
        d = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
        idx = [i for i, x in enumerate(d.data) if len(x.strip()) >= 20][:2000]
        return [d.data[i].replace("\n", " ").strip() for i in idx], np.array(d.target)[idx]
    if name == "bbc_news_dev":
        ds = load_dataset("SetFit/bbc-news", split="test")
        return [x.replace("\n", " ").strip() for x in ds["text"]], np.array(ds["label"])
    if name == "agnews_dev":
        ds = load_dataset("ag_news", split="train")
        rng = np.random.default_rng(7); idx = rng.choice(len(ds), 2000, replace=False)
        return [ds[int(i)]["text"].replace("\n", " ").strip() for i in idx], np.array([ds[int(i)]["label"] for i in idx])
    if name == "arxiv_dev":
        ds = load_dataset("ccdv/arxiv-classification", split="validation")
        return [x.replace("\n", " ").strip() for x in ds["text"][:2000]], np.array(ds["label"][:2000])
    raise ValueError(name)

PAPER_KS = {"20newsgroups": [10, 20, 30, 40, 50], "bbc_news": [3, 5, 10, 15, 20],
            "agnews": [3, 4, 8, 15, 20], "arxiv": [5, 10, 15, 20, 25], "20newsgroups_full": [20]}
PAPER_KS.update({f"{k}_dev": v for k, v in list(PAPER_KS.items())})

def embed(name, docs):
    path = os.path.join(CACHE, f"emb_{name}_{len(docs)}.npy")
    if os.path.exists(path):
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    e = SentenceTransformer("all-MiniLM-L6-v2").encode(docs, batch_size=64, normalize_embeddings=True,
                                                       show_progress_bar=False, convert_to_numpy=True)
    np.save(path, e); return e

# ---------------- metrics ----------------
def paper_tokenize(docs):
    return [set(re.findall(r"[a-zA-Z]{3,}", d.lower())) for d in docs]

def paper_npmi(topic_words, doc_sets):
    """Coherence exactly as in run_benchmark.py (pairs that never co-occur are skipped)."""
    n = len(doc_sets); df = {}
    for s in doc_sets:
        for w in s: df[w] = df.get(w, 0) + 1
    scores = []
    for words in topic_words:
        words = [w.lower() for w in words[:10] if df.get(w.lower(), 0) > 0]
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                c12 = sum((words[i] in s) and (words[j] in s) for s in doc_sets)
                if c12 == 0: continue
                p12 = c12 / n; p1 = df[words[i]] / n; p2 = df[words[j]] / n
                scores.append(math.log(p12 / (p1 * p2) + 1e-12) / (-math.log(p12 + 1e-12)))
    return float(np.mean(scores)) if scores else 0.0

def strict_npmi(topic_words, docs):
    """NPMI over the whole corpus, never-co-occurring pairs = -1 (Bouma 2009), bigram-aware.
    Self-contained so it is identical for every package version under test."""
    from itertools import combinations
    from sklearn.feature_extraction.text import CountVectorizer
    topics = [[w.lower() for w in t[:10]] for t in topic_words]
    vocab = sorted({w for t in topics for w in t})
    if not vocab:
        return 0.0
    vec = CountVectorizer(vocabulary=vocab, ngram_range=(1, max(len(v.split()) for v in vocab)),
                          stop_words="english", binary=True, token_pattern=r"(?u)\b[^\W\d_]{2,}[^\W_]*\b")
    X = vec.transform(docs).tocsc().astype(float)
    n = X.shape[0]; df = np.asarray(X.sum(axis=0)).ravel(); co = (X.T @ X).toarray()
    ix = {w: i for i, w in enumerate(vocab)}
    per_topic = []
    for t in topics:
        sc = []
        for a_, b_ in combinations([ix[w] for w in t], 2):
            if a_ == b_ or df[a_] == 0 or df[b_] == 0: continue
            if co[a_, b_] == 0: sc.append(-1.0); continue
            p = co[a_, b_] / n
            sc.append(1.0 if p >= 1 else math.log(p / (df[a_] / n * df[b_] / n)) / -math.log(p))
        per_topic.append(np.mean(sc) if sc else 0.0)
    return float(np.mean(per_topic)) if per_topic else 0.0


def diversity(topic_words):
    words = [w for t in topic_words for w in t[:10]]
    return len(set(words)) / len(words) if words else 0.0

# ---------------- models ----------------
def parse_cfg(s):
    out = {}
    for item in filter(None, s.split(",")):
        k, v = item.split("=")
        try: v = int(v)
        except ValueError:
            try: v = float(v)
            except ValueError: v = {"True": True, "False": False}.get(v, v)
        out[k] = v
    return out

def run_tritopic(docs, emb, k, seed, cfg):
    from tritopic import TriTopic, TriTopicConfig
    c = TriTopicConfig(verbose=False, random_state=seed)
    for key, val in cfg.items(): setattr(c, key, val)
    m = TriTopic(config=c, n_topics=k if k else "auto")
    m.fit(docs, embeddings=emb)
    words = [t.keywords[:10] for t in m.topics_ if t.topic_id != -1]
    return np.asarray(m.labels_), words

def run_bertopic(docs, emb, k, seed, cfg):
    from bertopic import BERTopic
    from umap import UMAP
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed)
    m = BERTopic(umap_model=umap_model, nr_topics=k if k else None, calculate_probabilities=False, verbose=False)
    labels, _ = m.fit_transform(docs, embeddings=emb)
    labels = np.asarray(labels)
    words = [[w for w, _ in (m.get_topic(t) or [])][:10] for t in sorted(set(labels) - {-1})]
    return labels, words

def run_bertopic_tuned(docs, emb, k, seed, cfg):
    """BERTopic with its documented best practices for keyword quality."""
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed)
    m = BERTopic(umap_model=umap_model, nr_topics=k if k else None, calculate_probabilities=False, verbose=False,
                 vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2),
                 ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True))
    labels, _ = m.fit_transform(docs, embeddings=emb)
    labels = np.asarray(labels)
    words = [[w for w, _ in (m.get_topic(t) or [])][:10] for t in sorted(set(labels) - {-1})]
    return labels, words

RUNNERS = {"tritopic": run_tritopic, "bertopic": run_bertopic, "bertopic_tuned": run_bertopic_tuned}

# ---------------- main loop ----------------
import umap  # JIT warm-up so the first run is not penalised
umap.UMAP(n_components=5, min_dist=0.0).fit_transform(np.random.rand(300, 20))

cfg = parse_cfg(args.cfg)
done = set()
if os.path.exists(args.out):
    prev = pd.read_csv(args.out)
    done = {(r.dataset, r.model, int(r.k_target), int(r.seed)) for r in prev.itertuples()}

for ds in args.datasets.split(","):
    docs, y = load(ds, args.max_docs)
    emb = embed(ds, docs)
    doc_sets = paper_tokenize(docs)
    ks = ([0] if "auto" in args.ks else []) + (PAPER_KS[ds] if "paper" in args.ks else [])
    for model in args.models.split(","):
        name = model + (f"[{args.tag}]" if args.tag and model == "tritopic" else "")
        for k in ks:
            for seed in map(int, args.seeds.split(",")):
                if (ds, name, k, seed) in done: continue
                random.seed(seed); np.random.seed(seed)
                t = time.perf_counter()
                try:
                    labels, words = RUNNERS[model](docs, emb, k, seed, cfg if model == "tritopic" else {})
                except Exception as e:
                    print(f"FAILED {ds} {name} k={k} seed={seed}: {e!r}", flush=True); continue
                dt = time.perf_counter() - t
                inl = labels >= 0
                row = dict(dataset=ds, model=name, k_target=k, seed=seed, n_docs=len(docs),
                           k_actual=len(set(labels[inl])), outliers=float(1 - inl.mean()),
                           nmi=NMI(y, labels), ari=ARI(y, labels),
                           nmi_inliers=NMI(y[inl], labels[inl]) if inl.sum() > 1 else np.nan,
                           coh_paper=paper_npmi(words, doc_sets), coh_strict=strict_npmi(words, docs),
                           diversity=diversity(words), time=dt)
                pd.DataFrame([row]).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
                print(f"{ds:13s} {name:22s} k={k:3d} s={seed} k_act={row['k_actual']:3d} out={row['outliers']:.2f} "
                      f"NMI={row['nmi']:.3f} ARI={row['ari']:.3f} cohP={row['coh_paper']:.3f} cohS={row['coh_strict']:.3f} "
                      f"div={row['diversity']:.2f} t={dt:.1f}s", flush=True)
