# decision-replay

`decision-replay` is a command-line tool that compares decisions your LLM already made with what a decision model would have answered for the same cases, and finds the confidence threshold above which the decision model can be trusted on its own. It is for teams thinking about replacing the small decisions inside an agent loop (which tool, which model tier, is this risky) with a decision model such as Jev, Laya, Kev or Nimble.

**Status: tested on synthetic data only.** It has not yet been run on a real production log.

## The problem

Decision models answer typed questions (pick one, yes/no, a score) with a confidence, in one fast pass, and are much cheaper than asking an LLM. The guides for adopting them say to replay your own logged LLM decisions through the model and fit confidence thresholds on your own data before switching, but leave that work to you. This tool is that analysis: one file, Python standard library only.

## Quick start

No install step. You need Python 3 (tested on 3.13).

```
python3 replay.py examples/decisions.jsonl examples/predictions.jsonl --target 0.85
```

The example files are synthetic (made by `examples/make_examples.py`). This is what the command prints:

```
== is_risky ==
  matched cases         24
  agreement with LLM    79.2% (19/24)
  accuracy vs truth     LLM 91.7%, decision model 87.5% (24 labelled)
  calibrated against    the LLM's answer (24 cases)
  threshold             not enough data (24 cases; a 85.0% target needs at least 30)

== model_route ==
  matched cases         40
  agreement with LLM    90.0% (36/40)
  calibrated against    the LLM's answer (40 cases)
  threshold             0.69 (Wilson 95% lower bound 85.1% >= target 85.0%)
  coverage              85.0% of cases at or above the threshold
  agreement above it    97.1% (33/34)
  latency per decision  881.5 ms -> 160.2 ms (-81.8%)
  cost per decision     $0.00250 -> $0.00040 (-84.2%)

OVERALL: matched cases 64; agreement with LLM 85.9% (55/64); accuracy vs truth LLM 91.7%, decision model 87.5% (24 labelled); calibrated against the LLM's answer (64 cases); threshold none reaches the 85.0% target: keep using the LLM for this decision
unmatched ids: 0 only in decisions, 0 only in predictions
```

The example uses `--target 0.85` because 64 rows are too few for the default. With the default target of 0.95 a group needs at least 73 cases above the threshold even if every one of them agrees, so both groups would report "not enough data".

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--target` | 0.95 | The agreement rate the threshold must guarantee |
| `--min-n` | 30 | Minimum number of cases at or above a threshold |
| `--json PATH` | off | Also write everything to a JSON file, including the full threshold sweep per group (each candidate threshold with `n`, `coverage`, `agreement`, `wilson_lower`) so you can plot it |

Run the tests with `python3 -m unittest`.

## Input files

Both files are JSONL: one JSON object per line. Rows are matched by `id`. Extra fields are ignored.

**DECISIONS.jsonl**: what the LLM already decided.

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Unique string |
| `question` | yes | A name for the kind of decision, e.g. `"model_route"`. Results are grouped by it |
| `llm_answer` | yes | What the LLM answered (string, boolean or number) |
| `truth` | no | The known-correct answer, if you have labels |
| `llm_latency_ms` | no | How long the LLM took |
| `llm_cost_usd` | no | What the LLM call cost |

**PREDICTIONS.jsonl**: what the decision model answers for the same ids.

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Same id as in the decisions file |
| `answer` | yes | The decision model's answer |
| `confidence` | yes | A number from 0 to 1 |
| `latency_ms` | no | How long the decision model took |
| `cost_usd` | no | What the call cost (use 0 for a local model) |

Answers are compared after trimming whitespace and ignoring case for strings; booleans and numbers compare by value. The string `"true"` and the boolean `true` are not the same answer, so log both sides in the same form.

You produce PREDICTIONS.jsonl yourself, with whatever decision model you use. The tool never calls a model, which is why it works with any of them.

### Optional: a Laya adapter (not run)

`adapters/laya_predict.py` writes a PREDICTIONS.jsonl using the open-source Laya model. **It was written from Laya's documentation and has never been run against the real library**, because Laya could not be installed where it was written. Its test uses a fake stand-in shaped like the documented output. Check its output on a few rows before relying on it. Docs it was written from: https://pypi.org/project/laya/ and https://huggingface.co/convaiinnovations/laya

```
pip install laya
python3 adapters/laya_predict.py INPUT.jsonl predictions.jsonl
```

Each input line needs `id`, `state` (the text to judge), `question` (the question in words) and `options` (a list of labels, or an object of label to description).

## How to read the report

- **Agreement**: out of the matched cases, how often the decision model gave the same answer as the LLM. If you supplied `truth`, you also get how often each side was right on the labelled cases.
- **Threshold**: the lowest confidence at which it is safe to stop asking the LLM. "Safe" means that among the cases at or above it, the Wilson lower bound on agreement reaches your target, with at least `--min-n` such cases. Cases with confidence exactly equal to the threshold count as above it.
- **Coverage**: the share of all cases at or above the threshold. This is the share of decisions the decision model could handle alone.
- **Savings**: the average latency and cost per decision if you ask the decision model first and also ask the LLM only when confidence is below the threshold, compared with asking the LLM every time. Low-confidence cases pay for both calls, so a poor threshold can make things slower. Only cases that have both the LLM field and the decision model field are used; if none do, that line says "not available (no latency/cost fields)".
- **No threshold**: "not enough data" means the log is too small to prove the target however well the model did. "Keep using the LLM for this decision" means there was enough data and the model was not good enough at any confidence level.
- **Calibrated against**: if a group has truth labels for at least `--min-n` cases, the threshold is fitted on whether the decision model matched the truth, using only the labelled cases. Otherwise it is fitted on whether it matched the LLM. Coverage and savings always use every matched case.

The per-question thresholds are the ones to use. The overall line mixes different kinds of decision and is there as a summary.

**Wilson lower bound**: given the agreement seen in a sample, it is a cautious estimate of the lowest the true agreement rate plausibly is (the bottom of a 95% confidence interval). It is used so that a small lucky sample does not look safe: 10 out of 10 agreeing is 100% observed but only 72% by this bound, while 200 out of 200 is 98%.

## Limits

- Agreement with the LLM is not correctness. If the LLM was wrong, a model that agrees with it is wrong too. Only truth labels tell you who is right.
- A threshold fitted on a small or unrepresentative log will not hold up. Use a log that looks like your real traffic, and remember the threshold was chosen on the same data it is reported on, so expect slightly worse in production.
- Re-check after changing prompts, the LLM, the decision model or its version. The confidences will shift.
- It does not call any model itself. You bring both files.
- The savings projection is simple arithmetic on the numbers you logged. It assumes the two calls run one after the other and ignores retries, batching and caching.

## Exit codes

0 on success. 2, with a one-line message on stderr, for a missing file, invalid JSON (with file and line number), a missing required field, a duplicate id or a confidence outside 0 to 1. Ids found in only one file are skipped, counted in the report and warned about on stderr.

## License

MIT. See `LICENSE`.
