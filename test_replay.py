import contextlib
import io
import json
import os
import tempfile
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
        self.assertIn("warning", err)

    def test_empty_files(self):
        code, out, err = run_main([self.write("d.jsonl", []), self.write("p.jsonl", [])])
        self.assertEqual(code, 0)
        self.assertIn("No matched cases", out)
        self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
