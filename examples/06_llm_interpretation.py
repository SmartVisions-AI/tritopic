"""
06 - Let an LLM interpret the topics: labels, descriptions, mixed-topic detection, overview.

Run:  OPENAI_API_KEY=sk-... python examples/06_llm_interpretation.py
Cost: about 1,500 input tokens per topic (a few cents for this example).
"""
import os

from sklearn.datasets import fetch_20newsgroups

from tritopic import SimpleLabeler, TopicInterpreter, TriTopic

data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                          categories=["sci.space", "rec.autos", "rec.motorcycles", "sci.med", "talk.politics.guns"])
documents = [d for d in data.data if len(d.strip()) > 50]

# A deliberately coarse model: some topics will mix two newsgroups
model = TriTopic(n_topics=3, verbose=False).fit(documents)

if not os.environ.get("OPENAI_API_KEY"):
    # Without an API key: keyword-based labels
    model.generate_labels(SimpleLabeler(n_words=3))
    print(model.get_topic_info()[["Topic", "Size", "Label"]].to_string(index=False))
    raise SystemExit("Set OPENAI_API_KEY to run the LLM part.")

# --- 1. Interpret every topic ------------------------------------------------------------
interpreter = TopicInterpreter(domain_hint="Usenet newsgroup posts")   # default model: gpt-6-luna
results = interpreter.interpret(model)          # also sets topic.label / topic.description
for topic_id, r in results.items():
    print(f"\nTopic {topic_id} ({r.size} docs): {r.label}  [{r.verdict}, confidence {r.confidence:.2f}]")
    print("  ", r.description)
    print("   aspects:", ", ".join(r.aspects))
    if r.verdict == "mixed":
        print("   sub-themes:", " | ".join(s["name"] for s in r.sub_themes))
    print("   evidence:", r.evidence)

# --- 2. Split topics the LLM judges as mixed ------------------------------------------------
log = interpreter.refine(model, results=results)   # reuse the interpretations from step 1
for entry in log:
    status = "split" if entry["kept"] else "kept together"
    print(f"\n{entry['label']}: {status} -> {entry['sub_themes']}")
print(model.get_topic_info()[["Topic", "Size", "Label"]].to_string(index=False))

# --- 3. Overview of the whole corpus ---------------------------------------------------------
print("\n" + interpreter.summarize(model))
for group in model.overview_["groups"]:
    print(f"- {group['name']}: {', '.join(group['topics'])}")

print(f"\n{interpreter.n_requests} requests, {interpreter.usage['input_tokens']} input tokens")

# --- Alternative: labels only, Anthropic or OpenAI SDK ---------------------------------------
# from tritopic import LLMLabeler
# model.generate_labels(LLMLabeler(provider="anthropic", api_key="...", language="german"))
