"""
13 - Research charts: sixteen views of a fitted model, saved as interactive HTML files.

Run:  python examples/13_research_charts.py
      (charts 14 and 15 need OPENAI_API_KEY and are skipped without it)

Every chart is a Plotly figure from tritopic.visualization.charts; the numbers behind it come from
tritopic.research, so you can also use them directly in tables or statistics.
"""
import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from tritopic import TopicInterpreter, TriTopic, TriTopicConfig
from tritopic.research import topic_evolution, topic_reliability
from tritopic.visualization import charts

cats = ["sci.space", "rec.autos", "sci.med", "comp.graphics", "talk.politics.guns"]
data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), categories=cats)
keep = [i for i, d in enumerate(data.data) if len(d.strip()) > 50]
documents = [data.data[i] for i in keep]
newsgroup = np.array(data.target_names)[data.target[keep]]

model = TriTopic(config=TriTopicConfig(verbose=False, random_state=42)).fit(documents)
rel = topic_reliability(model)            # needed for charts 1, 4 and 9
os.makedirs("charts", exist_ok=True)


def save(fig, name):
    fig.write_html(f"charts/{name}.html")
    print("saved", f"charts/{name}.html")


save(charts.plot_topic_table(model, reliability=rel), "01_topic_table")           # overview tiles
save(charts.plot_triview(model), "02_triview")                                    # meaning / wording / both
save(charts.plot_resolution_ladder(model), "03_zoom_ladder")                      # how topics split
save(charts.plot_topic_onions(model), "04_topic_onions")                          # core and edge
save(charts.plot_keyword_barcode(model, int(rel.topic.iloc[0])), "05_keyword_barcode")
save(charts.plot_constellation(model), "06_constellation")                        # topic network
save(charts.plot_coassignment(model, n_refits=5), "07_fuzzy_borders")             # stability under refits
lengths = np.array([len(d.split()) for d in documents])
save(charts.plot_group_tilt(model, np.where(lengths > np.median(lengths), "long", "short")), "08_group_tilt")
save(charts.plot_trust(model, rel), "09_trust_quadrant")
save(charts.plot_composition(model, newsgroup, reference_name="newsgroup"), "10_composition")

# 11 - birth timeline (20 Newsgroups has no dates: graphics posts are simulated to start in 2024;
#      real dated news: examples/12_topic_evolution_news.py)
rng = np.random.default_rng(0)
dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, len(documents)), unit="D")
late = newsgroup == "comp.graphics"
dates = dates.where(~late, pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, late.sum()), unit="D"))
save(charts.plot_birth_timeline(topic_evolution(model, dates, freq="Y"), highlight={"graphics": "graphics|image"}), "11_birth_timeline")

# 12 - codebook coverage: one model with all themes seeded, one with two seeds
seeds = {"Space": "Space flight, NASA, rockets, orbits and planets", "Cars": "Cars, engines, driving and dealers"}
seeded = TriTopic(config=TriTopicConfig(verbose=False, random_state=42)).fit(documents, seeds=seeds)
save(charts.plot_codebook_coverage({"2 seeds": seeded}), "12_codebook_coverage")

save(charts.plot_quote_wall(model), "13_quote_wall")                              # one sentence per topic
save(charts.plot_topic_discovery(model), "16_topic_discovery")                    # saturation per topic

if os.environ.get("OPENAI_API_KEY"):
    from tritopic.integrations.decisions import DecisionsClient, intercoder_reliability

    interp = TopicInterpreter(domain_hint="newsgroup posts")
    results = interp.interpret(model)
    log = interp.refine(model, results=results)
    save(charts.plot_verdict_board(results, refine_log=log), "14_verdict_board")
    ic = intercoder_reliability(model, DecisionsClient(), sample_size=100)
    save(charts.plot_coder_confusion(ic, model), "15_coder_confusion")
else:
    print("charts 14 and 15 skipped (set OPENAI_API_KEY)")
