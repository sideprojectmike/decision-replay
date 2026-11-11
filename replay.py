#!/usr/bin/env python3
"""decision-replay: compare logged LLM decisions with a decision model's answers.

Usage: python replay.py DECISIONS.jsonl PREDICTIONS.jsonl [--target 0.95] [--min-n 30] [--json PATH]
"""
import argparse
import bisect
import json
import math
import sys

Z = 1.959963984540054  # two-sided 95% normal quantile


class InputError(Exception):
    """A problem with an input file; shown as a one-line error, exit code 2."""


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def load_jsonl(path, required, strings=(), numbers=()):
    """Read a JSONL file into {id: row}, validating as it goes."""
    rows = {}
    try:
        f = open(path, encoding="utf-8")
    except OSError as e:
        raise InputError(f"{path}: cannot read file ({e.strerror})")
    with f:
        try:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                where = f"{path}:{n}"
                try:
                    row = json.loads(line)
                except ValueError as e:
                    raise InputError(f"{where}: invalid JSON ({e})")
                if not isinstance(row, dict):
                    raise InputError(f"{where}: expected a JSON object")
                for key in required:
                    if row.get(key) is None:
                        raise InputError(f"{where}: missing required field '{key}'")
                for key in strings:
                    if not isinstance(row[key], str):
                        raise InputError(f"{where}: field '{key}' must be a string")
                for key in numbers:
                    v = row.get(key)
                    if v is not None and not (is_number(v) and v >= 0):
                        raise InputError(f"{where}: field '{key}' must be a non-negative number")
                if "confidence" in required and not 0 <= row["confidence"] <= 1:
                    raise InputError(f"{where}: confidence {row['confidence']} is outside 0 to 1")
                if row["id"] in rows:
                    raise InputError(f"{where}: duplicate id '{row['id']}'")
                rows[row["id"]] = row
        except UnicodeDecodeError:
            raise InputError(f"{path}: not valid UTF-8 text")
    return rows


def load_decisions(path):
    return load_jsonl(path, ("id", "question", "llm_answer"), strings=("id", "question"),
                      numbers=("llm_latency_ms", "llm_cost_usd"))


def load_predictions(path):
    return load_jsonl(path, ("id", "answer", "confidence"), strings=("id",),
                      numbers=("confidence", "latency_ms", "cost_usd"))


def same(a, b):
    """Answers match after trimming and case-folding strings; other values by value."""
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().casefold() == b.strip().casefold()
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b  # keep true distinct from 1
    return a == b


def build_cases(decisions, predictions):
    """One case per id present in both files."""
    cases = []
    for i, d in decisions.items():
        p = predictions.get(i)
        if p is None:
            continue
        has_truth = d.get("truth") is not None
        cases.append({
            "question": d["question"],
            "conf": p["confidence"],
            "agree": same(p["answer"], d["llm_answer"]),
            "has_truth": has_truth,
            "llm_ok": has_truth and same(d["llm_answer"], d["truth"]),
            "model_ok": has_truth and same(p["answer"], d["truth"]),
            "llm_latency_ms": d.get("llm_latency_ms"),
            "llm_cost_usd": d.get("llm_cost_usd"),
            "latency_ms": p.get("latency_ms"),
            "cost_usd": p.get("cost_usd"),
        })
    return cases


def rate(k, n):
    return k / n if n else None


def wilson_lower(k, n):
    """Lower end of the Wilson 95% score interval for k successes out of n."""
    if n == 0:
        return 0.0
    p = k / n
    centre = p + Z * Z / (2 * n)
    margin = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return max(0.0, (centre - margin) / (1 + Z * Z / n))


def threshold_sweep(cal, all_confs):
    """Every candidate threshold, ascending.

    cal is [(confidence, correct)] for the cases used to calibrate; all_confs are the
    confidences of every matched case in the group, used for coverage.
    """
    cal = sorted(cal, key=lambda x: x[0], reverse=True)
    all_confs = sorted(all_confs)
    out, k = [], 0
    for i, (conf, ok) in enumerate(cal):
        k += ok
        if i + 1 < len(cal) and cal[i + 1][0] == conf:
            continue  # ties: every case with confidence equal to t counts as at-or-above t
        n = i + 1
        covered = len(all_confs) - bisect.bisect_left(all_confs, conf)
        out.append({"threshold": conf, "n": n, "coverage": covered / len(all_confs),
                    "agreement": k / n, "wilson_lower": wilson_lower(k, n)})
    return out[::-1]


def calibrate(cases, target, min_n):
    """Lowest threshold whose Wilson lower bound on agreement reaches the target."""
    labelled = [c for c in cases if c["has_truth"]]
    if len(labelled) >= min_n:
        basis, cal = "truth", [(c["conf"], c["model_ok"]) for c in labelled]
    else:
        basis, cal = "llm_answer", [(c["conf"], c["agree"]) for c in cases]
    sweep = threshold_sweep(cal, [c["conf"] for c in cases])
    # even with perfect agreement the bound is n / (n + Z^2), so fewer cases than this cannot pass
    needed = max(min_n, math.ceil(target * Z * Z / (1 - target)))
    pick = next((s for s in sweep if s["n"] >= min_n and s["wilson_lower"] >= target), None)
    if pick:
        status = "ok"
    elif len(cal) < needed:
        status = "not_enough_data"
    else:
        status = "no_threshold"
    return {"basis": basis, "n_calibration": len(cal), "n_needed": needed, "status": status,
            "threshold": pick, "sweep": sweep}


def analyse(cases, target, min_n):
    n = len(cases)
    agree = sum(c["agree"] for c in cases)
    labelled = [c for c in cases if c["has_truth"]]
    return {
        "n": n,
        "agree_count": agree,
        "agreement": rate(agree, n),
        "n_truth": len(labelled),
        "llm_accuracy": rate(sum(c["llm_ok"] for c in labelled), len(labelled)),
        "model_accuracy": rate(sum(c["model_ok"] for c in labelled), len(labelled)),
        "calibration": calibrate(cases, target, min_n),
    }


def replay(decisions, predictions, target=0.95, min_n=30):
    cases = build_cases(decisions, predictions)
    groups = {}
    for c in cases:
        groups.setdefault(c["question"], []).append(c)
    only_dec = [d["question"] for i, d in decisions.items() if i not in predictions]
    return {
        "target": target,
        "min_n": min_n,
        "unmatched": {
            "only_in_decisions": len(only_dec),
            "only_in_predictions": sum(i not in decisions for i in predictions),
            "only_in_decisions_by_question": {q: only_dec.count(q) for q in sorted(set(only_dec))},
        },
        "groups": {q: analyse(groups[q], target, min_n) for q in sorted(groups)},
        "overall": analyse(cases, target, min_n),
    }


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def group_lines(r, target):
    """(label, value) rows for one group."""
    rows = [("matched cases", str(r["n"])),
            ("agreement with LLM", f"{pct(r['agreement'])} ({r['agree_count']}/{r['n']})")]
    if r["n_truth"]:
        rows.append(("accuracy vs truth", f"LLM {pct(r['llm_accuracy'])}, decision model "
                     f"{pct(r['model_accuracy'])} ({r['n_truth']} labelled)"))
    c = r["calibration"]
    against = "truth labels" if c["basis"] == "truth" else "the LLM's answer"
    rows.append(("calibrated against", f"{against} ({c['n_calibration']} cases)"))
    t = c["threshold"]
    if t:
        rows += [("threshold", f"{t['threshold']:g} (Wilson 95% lower bound {pct(t['wilson_lower'])} "
                  f">= target {pct(target)})"),
                 ("coverage", f"{pct(t['coverage'])} of cases at or above the threshold"),
                 ("agreement above it", f"{pct(t['agreement'])} ({round(t['agreement'] * t['n'])}/{t['n']})")]
    elif c["status"] == "not_enough_data":
        rows.append(("threshold", f"not enough data ({c['n_calibration']} cases; a {pct(target)} target "
                     f"needs at least {c['n_needed']})"))
    else:
        rows.append(("threshold", f"none reaches the {pct(target)} target: "
                     "keep using the LLM for this decision"))
    return rows


def format_report(result):
    out = []
    for q, r in result["groups"].items():
        out.append(f"== {q} ==")
        out += [f"  {label:<22}{value}" for label, value in group_lines(r, result['target'])]
        missing = result["unmatched"]["only_in_decisions_by_question"].get(q)
        if missing:
            out.append(f"  {'no prediction for':<22}{missing} decisions")
        out.append("")
    o, u = result["overall"], result["unmatched"]
    if not o["n"]:
        out.append("No matched cases: nothing to compare.")
    else:
        out.append("OVERALL: " + "; ".join(f"{label} {value}" for label, value in group_lines(o, result['target'])))
    out.append(f"unmatched ids: {u['only_in_decisions']} only in decisions, "
               f"{u['only_in_predictions']} only in predictions")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare logged LLM decisions with a decision model's answers.")
    parser.add_argument("decisions", help="JSONL of decisions the LLM already made")
    parser.add_argument("predictions", help="JSONL of the decision model's answers for the same ids")
    parser.add_argument("--target", type=float, default=0.95,
                        help="agreement rate the Wilson lower bound must reach (default 0.95)")
    parser.add_argument("--min-n", type=int, default=30,
                        help="minimum cases at or above a threshold (default 30)")
    parser.add_argument("--json", metavar="PATH",
                        help="also write full results, including the threshold sweep, as JSON")
    args = parser.parse_args(argv)
    if not 0 < args.target < 1:
        parser.error("--target must be between 0 and 1 (exclusive)")
    if args.min_n < 1:
        parser.error("--min-n must be at least 1")
    try:
        result = replay(load_decisions(args.decisions), load_predictions(args.predictions),
                        args.target, args.min_n)
        if args.json:
            try:
                with open(args.json, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
            except OSError as e:
                raise InputError(f"{args.json}: cannot write file ({e.strerror})")
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(format_report(result))
    u = result["unmatched"]
    if u["only_in_decisions"] or u["only_in_predictions"]:
        print(f"warning: {u['only_in_decisions']} ids only in {args.decisions}, "
              f"{u['only_in_predictions']} only in {args.predictions}; they were skipped",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
