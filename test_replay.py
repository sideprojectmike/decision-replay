import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest

import replay


def run_main(argv):
    """Run replay.main, returning (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = replay.main(argv)
    return code, out.getvalue(), err.getvalue()


class FileCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def write(self, name, rows):
        """Write rows (dicts, or raw strings for broken lines) as JSONL; return the path."""
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write((row if isinstance(row, str) else json.dumps(row)) + "\n")
        return path


def dec(i, answer, question="q", **extra):
    return {"id": i, "question": question, "llm_answer": answer, **extra}


def pred(i, answer, confidence=0.9, **extra):
    return {"id": i, "answer": answer, "confidence": confidence, **extra}


class TestSame(unittest.TestCase):
    def test_strings_trimmed_and_casefolded(self):
        self.assertTrue(replay.same("  Large ", "large"))
        self.assertFalse(replay.same("large", "small"))

    def test_non_strings_compare_by_value(self):
        self.assertTrue(replay.same(True, True))
        self.assertFalse(replay.same(True, False))
        self.assertTrue(replay.same(3, 3.0))
        self.assertFalse(replay.same(3, 4))
        self.assertFalse(replay.same(True, 1))
        self.assertFalse(replay.same("1", 1))
        self.assertTrue(replay.same(["a", 1], ["a", 1]))


class TestAgreement(unittest.TestCase):
    def result(self, decisions, predictions):
        return replay.replay({d["id"]: d for d in decisions}, {p["id"]: p for p in predictions})

    def test_agreement_per_question_and_overall(self):
        r = self.result(
            [dec("1", "a", "x"), dec("2", "b", "x"), dec("3", True, "y"), dec("4", 2, "y")],
            [pred("1", "A "), pred("2", "c"), pred("3", True), pred("4", 2.0)])
        self.assertEqual(r["groups"]["x"]["agreement"], 0.5)
        self.assertEqual(r["groups"]["y"]["agreement"], 1.0)
        self.assertEqual(r["overall"]["n"], 4)
        self.assertEqual(r["overall"]["agreement"], 0.75)

    def test_accuracy_against_truth_uses_only_labelled_cases(self):
        r = self.result(
            [dec("1", "a", truth="a"), dec("2", "a", truth="b"), dec("3", "a")],
            [pred("1", "a"), pred("2", "b"), pred("3", "z")])["groups"]["q"]
        self.assertEqual(r["n_truth"], 2)
        self.assertEqual(r["llm_accuracy"], 0.5)
        self.assertEqual(r["model_accuracy"], 1.0)

    def test_unmatched_ids_are_counted_and_skipped(self):
        r = self.result([dec("1", "a"), dec("2", "a"), dec("3", "a", "other")],
                        [pred("1", "a"), pred("8", "a"), pred("9", "a")])
        self.assertEqual(r["overall"]["n"], 1)
        self.assertEqual(r["unmatched"]["only_in_decisions"], 2)
        self.assertEqual(r["unmatched"]["only_in_predictions"], 2)
        self.assertEqual(r["unmatched"]["only_in_decisions_by_question"], {"q": 1, "other": 1})


def cases(rows):
    """Cases from (confidence, agree) or (confidence, agree, model_ok) tuples; 3-tuples carry truth."""
    return [{"conf": r[0], "agree": r[1], "has_truth": len(r) > 2, "model_ok": len(r) > 2 and r[2],
             "llm_ok": len(r) > 2} for r in rows]


class TestWilson(unittest.TestCase):
    def test_hand_computed_values(self):
        # 30/30: bound is n / (n + z^2) = 30 / 33.8415 = 0.88649
        self.assertAlmostEqual(replay.wilson_lower(30, 30), 0.88649, places=4)
        # 45/50: (0.9 + 0.0384146 - 1.959964 * sqrt(0.0018 + 0.00038415)) / 1.0768292 = 0.78640
        self.assertAlmostEqual(replay.wilson_lower(45, 50), 0.78640, places=4)

    def test_edges(self):
        self.assertEqual(replay.wilson_lower(0, 0), 0.0)
        self.assertEqual(replay.wilson_lower(0, 20), 0.0)


class TestThreshold(unittest.TestCase):
    # With target 0.80: 20/20 has bound 20/23.84 = 0.839 (passes), 20/21 has 0.773 (fails).

    def test_lowest_passing_threshold(self):
        # 20 agreeing cases at 0.9; below them only disagreements, so 0.9 is the answer.
        c = replay.calibrate(cases([(0.9, True)] * 20 + [(0.5, False)] * 20), 0.80, 10)
        self.assertEqual(c["status"], "ok")
        self.assertEqual(c["basis"], "llm_answer")
        self.assertEqual(c["threshold"]["threshold"], 0.9)
        self.assertEqual(c["threshold"]["n"], 20)
        self.assertEqual(c["threshold"]["coverage"], 0.5)
        self.assertEqual(c["threshold"]["agreement"], 1.0)

    def test_goes_lower_while_the_bound_holds(self):
        # 40 agreeing cases spread over 0.7..0.9, then junk at 0.3: lowest passing t is 0.7.
        rows = [(0.9, True)] * 20 + [(0.8, True)] * 10 + [(0.7, True)] * 10 + [(0.3, False)] * 10
        c = replay.calibrate(cases(rows), 0.80, 10)
        self.assertEqual(c["threshold"]["threshold"], 0.7)
        self.assertEqual(c["threshold"]["coverage"], 0.8)
        self.assertEqual([s["threshold"] for s in c["sweep"]], [0.3, 0.7, 0.8, 0.9])
        self.assertEqual([s["n"] for s in c["sweep"]], [50, 40, 30, 20])

    def test_ties_at_the_threshold_are_all_included(self):
        # 19 agree at 0.9; at 0.8 one agrees and one disagrees. t=0.8 must take both tied
        # cases (20/21 = 0.773, fails), not just the agreeing one (20/20 would pass).
        rows = [(0.9, True)] * 19 + [(0.8, True), (0.8, False)]
        c = replay.calibrate(cases(rows), 0.80, 10)
        self.assertEqual(c["sweep"][0], {"threshold": 0.8, "n": 21, "coverage": 1.0,
                                         "agreement": 20 / 21,
                                         "wilson_lower": replay.wilson_lower(20, 21)})
        self.assertEqual(c["threshold"]["threshold"], 0.9)  # 19/19 has bound 0.832
        # Requiring 20 cases rules out 0.9, and the tie rules out 0.8.
        c = replay.calibrate(cases(rows), 0.80, 20)
        self.assertEqual(c["status"], "no_threshold")
        self.assertIsNone(c["threshold"])

    def test_no_threshold_meets_the_bar(self):
        # 100 cases, half disagree at every confidence level: plenty of data, never good enough.
        rows = [(conf, ok) for conf in (0.6, 0.7, 0.8, 0.9) for ok in (True, False)] * 13
        c = replay.calibrate(cases(rows), 0.95, 30)
        self.assertEqual(c["status"], "no_threshold")
        self.assertIsNone(c["threshold"])

    def test_not_enough_data(self):
        # 29 perfect cases < min_n; and 40 perfect cases cannot reach 0.95 (needs 73).
        self.assertEqual(replay.calibrate(cases([(0.9, True)] * 29), 0.80, 30)["status"], "not_enough_data")
        c = replay.calibrate(cases([(0.9, True)] * 40), 0.95, 30)
        self.assertEqual((c["status"], c["n_needed"]), ("not_enough_data", 73))
        self.assertEqual(replay.calibrate([], 0.95, 30)["status"], "not_enough_data")

    def test_identical_confidences_and_perfect_agreement(self):
        c = replay.calibrate(cases([(0.5, True)] * 80), 0.95, 30)
        self.assertEqual(len(c["sweep"]), 1)
        self.assertEqual(c["threshold"]["threshold"], 0.5)
        self.assertEqual(c["threshold"]["coverage"], 1.0)
        self.assertEqual(c["threshold"]["agreement"], 1.0)

    def test_truth_based_calibration(self):
        # The model disagrees with the LLM everywhere, but is right against truth at 0.9.
        rows = [(0.9, False, True)] * 20 + [(0.4, True, False)] * 20
        c = replay.calibrate(cases(rows), 0.80, 10)
        self.assertEqual(c["basis"], "truth")
        self.assertEqual(c["threshold"]["threshold"], 0.9)

    def test_too_few_truth_labels_falls_back_to_llm_answer(self):
        rows = [(0.9, True, False)] * 5 + [(0.9, True)] * 20
        c = replay.calibrate(cases(rows), 0.80, 10)
        self.assertEqual(c["basis"], "llm_answer")
        self.assertEqual(c["threshold"]["n"], 25)

    def test_coverage_counts_unlabelled_cases_too(self):
        # 20 labelled cases at 0.9, 20 unlabelled at 0.95 and 40 unlabelled at 0.1.
        rows = [(0.9, True, True)] * 20 + [(0.95, True)] * 20 + [(0.1, True)] * 40
        c = replay.calibrate(cases(rows), 0.80, 10)
        self.assertEqual(c["threshold"]["n"], 20)
        self.assertEqual(c["threshold"]["coverage"], 0.5)


class TestSavings(unittest.TestCase):
    CASES = [  # two confident cases skip the LLM; the two below 0.8 pay for both
        {"conf": 0.9, "llm_latency_ms": 1000, "latency_ms": 50, "llm_cost_usd": 0.01, "cost_usd": 0.001},
        {"conf": 0.8, "llm_latency_ms": 1000, "latency_ms": 50, "llm_cost_usd": 0.01, "cost_usd": 0.001},
        {"conf": 0.7, "llm_latency_ms": 1000, "latency_ms": 50, "llm_cost_usd": 0.01, "cost_usd": 0.001},
        {"conf": 0.1, "llm_latency_ms": 2000, "latency_ms": 50, "llm_cost_usd": 0.02, "cost_usd": None},
    ]

    def test_latency_by_hand(self):
        # before = (1000 * 3 + 2000) / 4 = 1250; after = (50 + 50 + 1050 + 2050) / 4 = 800
        s = replay.savings(self.CASES, 0.8, "llm_latency_ms", "latency_ms")
        self.assertEqual((s["n"], s["before"], s["after"]), (4, 1250, 800))
        self.assertAlmostEqual(s["change_pct"], -36.0)

    def test_cost_by_hand_skips_cases_missing_a_field(self):
        # three cases have both fields: before = 0.01; after = (0.001 + 0.001 + 0.011) / 3
        s = replay.savings(self.CASES, 0.8, "llm_cost_usd", "cost_usd")
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["before"], 0.01)
        self.assertAlmostEqual(s["after"], 0.013 / 3)
        self.assertAlmostEqual(s["change_pct"], -56.6666667)

    def test_slower_when_nothing_is_confident(self):
        s = replay.savings(self.CASES, 0.95, "llm_latency_ms", "latency_ms")
        self.assertEqual((s["before"], s["after"]), (1250, 1300))
        self.assertAlmostEqual(s["change_pct"], 4.0)

    def test_missing_fields(self):
        rows = [{"conf": 0.9, "llm_latency_ms": 5, "latency_ms": None}]
        self.assertIsNone(replay.savings(rows, 0.5, "llm_latency_ms", "latency_ms"))


class TestValidation(FileCase):
    def assert_error(self, decisions, predictions, *expected):
        code, out, err = run_main([decisions, predictions])
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertEqual(len(err.strip().splitlines()), 1, err)
        self.assertNotIn("Traceback", err)
        for text in expected:
            self.assertIn(text, err)

    def test_missing_file(self):
        missing = os.path.join(self.dir, "nope.jsonl")
        self.assert_error(missing, self.write("p.jsonl", []), missing)

    def test_invalid_json_names_file_and_line(self):
        d = self.write("d.jsonl", [dec("1", "a"), "{not json"])
        self.assert_error(d, self.write("p.jsonl", []), f"{d}:2", "invalid JSON")

    def test_line_that_is_not_an_object(self):
        d = self.write("d.jsonl", ["[1, 2]"])
        self.assert_error(d, self.write("p.jsonl", []), f"{d}:1", "JSON object")

    def test_missing_required_field(self):
        d = self.write("d.jsonl", [dec("1", "a"), dec("2", "a"), {"id": "3", "question": "q"}])
        self.assert_error(d, self.write("p.jsonl", []), f"{d}:3", "llm_answer")
        p = self.write("p.jsonl", [{"id": "1", "answer": "a"}])
        self.assert_error(self.write("d2.jsonl", []), p, f"{p}:1", "confidence")

    def test_duplicate_id(self):
        d = self.write("d.jsonl", [dec("1", "a"), dec("1", "b")])
        self.assert_error(d, self.write("p.jsonl", []), f"{d}:2", "duplicate id '1'")

    def test_confidence_out_of_range_or_not_a_number(self):
        d = self.write("d.jsonl", [])
        for bad in (1.5, -0.1, "0.9", True):
            p = self.write("p.jsonl", [pred("1", "a"), pred("2", "a", bad)])
            self.assert_error(d, p, f"{p}:2", "confidence")

    def test_non_numeric_latency(self):
        d = self.write("d.jsonl", [dec("1", "a", llm_latency_ms="fast")])
        self.assert_error(d, self.write("p.jsonl", []), f"{d}:1", "llm_latency_ms")


class TestEndToEnd(FileCase):
    def test_report_and_exit_code(self):
        d = self.write("d.jsonl", [dec("1", "big", "route", truth="big"), dec("2", "small", "route"),
                                   dec("3", False, "risky"), dec("4", True, "risky")])
        p = self.write("p.jsonl", [pred("1", "BIG"), pred("2", "big"), pred("3", False), pred("5", True)])
        code, out, err = run_main([d, p])
        self.assertEqual(code, 0)
        self.assertIn("== route ==", out)
        self.assertIn("50.0% (1/2)", out)
        self.assertIn("== risky ==", out)
        self.assertIn("OVERALL", out)
        self.assertIn("1 only in decisions, 1 only in predictions", out)
        self.assertIn("not enough data", out)
        self.assertIn("warning", err)

    def test_threshold_report_and_json_sweep(self):
        d = self.write("d.jsonl", [dec(str(i), "a") for i in range(40)])
        p = self.write("p.jsonl", [pred(str(i), "a" if i < 20 else "b", 0.9 if i < 20 else 0.5)
                                   for i in range(40)])
        out_path = os.path.join(self.dir, "out.json")
        code, out, err = run_main([d, p, "--target", "0.8", "--min-n", "10", "--json", out_path])
        self.assertEqual((code, err), (0, ""))
        self.assertIn("threshold             0.9 ", out)
        self.assertIn("50.0% of cases at or above", out)
        with open(out_path) as f:
            data = json.load(f)
        self.assertEqual(data["target"], 0.8)
        sweep = data["groups"]["q"]["calibration"]["sweep"]
        self.assertEqual([(s["threshold"], s["n"]) for s in sweep], [(0.5, 40), (0.9, 20)])
        self.assertEqual(set(sweep[0]), {"threshold", "n", "coverage", "agreement", "wilson_lower"})

    def test_savings_in_report(self):
        d = self.write("d.jsonl", [dec(str(i), "a", llm_latency_ms=1000) for i in range(40)])
        p = self.write("p.jsonl", [pred(str(i), "a" if i < 20 else "b", 0.9 if i < 20 else 0.5,
                                        latency_ms=100) for i in range(40)])
        code, out, _ = run_main([d, p, "--target", "0.8", "--min-n", "10"])
        self.assertEqual(code, 0)
        self.assertIn("1000.0 ms -> 600.0 ms (-40.0%)", out)  # half the cases still pay 1000
        self.assertIn("cost per decision     not available (no latency/cost fields)", out)

    def test_keep_using_the_llm_message(self):
        d = self.write("d.jsonl", [dec(str(i), "a") for i in range(100)])
        p = self.write("p.jsonl", [pred(str(i), "a" if i % 2 else "b") for i in range(100)])
        code, out, _ = run_main([d, p])
        self.assertEqual(code, 0)
        self.assertIn("keep using the LLM for this decision", out)

    def test_bad_options_exit_2(self):
        d, p = self.write("d.jsonl", []), self.write("p.jsonl", [])
        for extra in (["--target", "1.5"], ["--min-n", "0"]):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                replay.main([d, p] + extra)
            self.assertEqual(cm.exception.code, 2)
        code, _, err = run_main([d, p, "--json", os.path.join(self.dir, "no", "dir.json")])
        self.assertEqual(code, 2)
        self.assertIn("cannot write", err)

    def test_empty_files(self):
        code, out, err = run_main([self.write("d.jsonl", []), self.write("p.jsonl", [])])
        self.assertEqual(code, 0)
        self.assertIn("No matched cases", out)
        self.assertEqual(err, "")


class TestExamples(unittest.TestCase):
    def test_example_files_run_clean(self):
        here = os.path.dirname(os.path.abspath(__file__))
        code, out, err = run_main([os.path.join(here, "examples", "decisions.jsonl"),
                                   os.path.join(here, "examples", "predictions.jsonl"),
                                   "--target", "0.85"])
        self.assertEqual((code, err), (0, ""))
        self.assertIn("== model_route ==", out)
        self.assertIn("== is_risky ==", out)


class TestLayaAdapter(FileCase):
    """Runs the adapter against a FAKE laya module shaped like Laya's documented output.
    This checks the adapter's own plumbing only; the real library has never been run here."""

    def test_writes_predictions_from_documented_result_shape(self):
        calls = []

        class FakeAgent:
            def predict(self, state, questions):
                calls.append((state, questions))
                (name, q), = questions.items()
                first = next(iter(q["criteria"]))
                return {"answers": {name: {"choice": first, "answer_confidence": 0.91,
                                           "probabilities": {first: 0.91}}}}

        fake = types.ModuleType("laya")
        fake.load = lambda repo: FakeAgent()
        sys.modules["laya"] = fake
        self.addCleanup(sys.modules.pop, "laya")
        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            "laya_predict", os.path.join(here, "adapters", "laya_predict.py"))
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)

        src = self.write("in.jsonl", [
            {"id": "1", "state": "billed twice", "question": "Which team?", "options": ["billing", "tech"]},
            {"id": "2", "state": "app crashes", "question": "Which team?",
             "options": {"tech": "bugs and outages", "billing": "invoices"}}])
        dst = os.path.join(self.dir, "predictions.jsonl")
        self.assertEqual(adapter.main([src, dst]), 0)
        self.assertEqual(calls[0][1]["decision"], {"type": "choice", "instructions": "Which team?",
                                                   "criteria": {"billing": "billing", "tech": "tech"}})
        rows = replay.load_predictions(dst)  # the output must be valid input for replay.py
        self.assertEqual((rows["1"]["answer"], rows["1"]["confidence"]), ("billing", 0.91))
        self.assertEqual(rows["2"]["answer"], "tech")
        self.assertIn("latency_ms", rows["1"])


if __name__ == "__main__":
    unittest.main()
