"""
11 - From topics to a coding scheme, checked by an LLM second coder.

Run:  OPENAI_API_KEY=sk-... python examples/11_codebook_and_second_coder.py
Cost: a few cents (one request per topic for the codebook, ~one per document for the coder).

1. TopicInterpreter.codebook() writes a category per topic in the style of qualitative
   content analysis (Mayring): definition, inclusion and exclusion criteria, coding notes.
   The anchor examples are real sentences from the documents, not generated text.
2. intercoder_reliability() lets an LLM assign a random sample to the topics, knowing only
   the codebook information (label, description, keywords), and reports Cohen's kappa and
   precision / recall / F1 per topic.
"""
from sklearn.datasets import fetch_20newsgroups

from tritopic import TopicInterpreter, TriTopic, TriTopicConfig
from tritopic.integrations.decisions import DecisionsClient, intercoder_reliability

data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                          categories=["sci.space", "rec.autos", "sci.med", "comp.graphics"])
documents = [d for d in data.data if len(d.strip()) > 50]
model = TriTopic(config=TriTopicConfig(verbose=False, random_state=42)).fit(documents)

# Labels and descriptions first: the second coder works from them.
interp = TopicInterpreter(domain_hint="newsgroup posts")
interp.interpret(model)

# --- 1. Codebook ------------------------------------------------------------------------------
codebook = interp.codebook(model, n_quotes=2)            # also stored as model.codebook_
for _, c in codebook.iterrows():
    print(f"\n## {c['name']} ({c['size']} posts)\n{c['definition']}")
    print("Include:", "; ".join(c["inclusion"]))
    print("Exclude:", "; ".join(c["exclusion"]))
    print("Anchor:", c["anchor_examples"][0] if c["anchor_examples"] else "-")
# codebook.to_excel("codebook.xlsx")  /  codebook.to_markdown()

# --- 2. LLM as second coder ---------------------------------------------------------------------
ic = intercoder_reliability(model, DecisionsClient(), sample_size=100)
print(f"\nCohen's kappa {ic.kappa:.2f}, agreement {ic.agreement:.0%} on {ic.n} posts")
print(ic.per_topic.round(2).to_string(index=False))
print(ic.confusion)                                       # rows: TriTopic, columns: LLM
