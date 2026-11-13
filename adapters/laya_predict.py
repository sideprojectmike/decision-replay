#!/usr/bin/env python3
"""Produce a PREDICTIONS.jsonl for replay.py by running the open-source Laya decision model locally.

NOT RUN AGAINST LAYA. This script was written from Laya's published documentation only;
Laya could not be installed where it was written, so it has never been executed against
the real library. Check the output of your first few rows before trusting it.

Documentation it was written from:
  https://pypi.org/project/laya/               (laya.load, predict, the answer fields)
  https://huggingface.co/convaiinnovations/laya  (model card, same quickstart)

What those pages document, and all this script relies on:
  agent = laya.load("convaiinnovations/laya")
  result = agent.predict(state, {name: {"type": "choice", "instructions": ..., "criteria": {label: description}}})
  result["answers"][name]["choice"]             the chosen label
  result["answers"][name]["answer_confidence"]  "the probability of the reported answer"
  result["answers"][name]["probabilities"]      {label: probability}

Assumption not confirmed by the docs: a list of options is passed as {label: label},
because the documented "choice" examples give criteria as a dict of label -> description.

Usage: python adapters/laya_predict.py INPUT.jsonl PREDICTIONS.jsonl
Each input line: {"id": ..., "state": "text to judge", "question": "Which ...?", "options": [...]}
"options" may also be a dict of label -> description. Answers come back as label strings.
Requires: pip install laya  (Python 3.10 or newer, per the docs).
"""
import json
import sys
import time


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: laya_predict.py INPUT.jsonl PREDICTIONS.jsonl", file=sys.stderr)
        return 2
    import laya

    agent = laya.load("convaiinnovations/laya")
    with open(args[0], encoding="utf-8") as src, open(args[1], "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            options = row["options"]
            criteria = options if isinstance(options, dict) else {str(o): str(o) for o in options}
            question = {"type": "choice", "instructions": row["question"], "criteria": criteria}
            start = time.perf_counter()
            result = agent.predict(row["state"], {"decision": question})
            latency_ms = (time.perf_counter() - start) * 1000
            answer = result["answers"]["decision"]
            confidence = answer.get("answer_confidence")
            if confidence is None:
                confidence = answer["probabilities"][answer["choice"]]
            dst.write(json.dumps({"id": row["id"], "answer": answer["choice"],
                                  "confidence": confidence, "latency_ms": round(latency_ms, 2)}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
