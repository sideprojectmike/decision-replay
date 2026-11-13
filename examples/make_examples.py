"""Regenerate the SYNTHETIC example files. No randomness: the output is always the same.

Run from the repository root:  python3 examples/make_examples.py
The numbers are invented to exercise the report; they say nothing about any real model.
"""
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
decisions, predictions = [], []

# model_route: 40 cases, no truth labels. The 32 high-confidence cases all agree with the
# LLM; of the 8 low-confidence ones, half disagree.
for i in range(40):
    llm = "large" if i % 3 == 0 else "small"
    if i < 32:
        confidence, answer = 0.80 + (i % 10) * 0.02, llm
    else:
        confidence = 0.45 + (i - 32) * 0.04
        answer = llm if i % 2 == 0 else ("small" if llm == "large" else "large")
    decisions.append({"id": f"syn-route-{i:02d}", "question": "model_route", "llm_answer": llm,
                      "llm_latency_ms": 700 + (i * 37) % 400,
                      "llm_cost_usd": round(0.0021 + (i % 5) * 0.0002, 4), "synthetic": True})
    predictions.append({"id": f"syn-route-{i:02d}", "answer": answer, "confidence": round(confidence, 2),
                        "latency_ms": 30 + i % 9, "cost_usd": 0.00002, "synthetic": True})

# is_risky: 24 yes/no cases with truth labels. Too few to calibrate at the default --min-n.
for i in range(24):
    truth = i % 4 == 0
    llm = (not truth) if i in (5, 17) else truth
    answer = (not truth) if i in (3, 11, 19) else truth
    decisions.append({"id": f"syn-risky-{i:02d}", "question": "is_risky", "llm_answer": llm,
                      "truth": truth, "llm_latency_ms": 500 + (i * 53) % 300,
                      "llm_cost_usd": 0.0012, "synthetic": True})
    predictions.append({"id": f"syn-risky-{i:02d}", "answer": answer,
                        "confidence": round(0.55 + (i * 7 % 24) * 0.018, 3),
                        "latency_ms": 28 + i % 5, "cost_usd": 0.00002, "synthetic": True})

for name, rows in (("decisions.jsonl", decisions), ("predictions.jsonl", predictions)):
    with open(os.path.join(here, name), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
