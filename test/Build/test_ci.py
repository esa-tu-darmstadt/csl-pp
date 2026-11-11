"""Exercise CI log retention and failure handling with real subprocesses."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).resolve().parents[2] / "scripts/ci-run.py"


class BuildLogTests(unittest.TestCase):
    """Ensure noisy builds retain their output and report their actual failure."""

    def run_command(self, source, timeout=10):
        """Run a Python fixture through the same wrapper used for CI builds."""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "logs/build.log"
            result = subprocess.run(
                [sys.executable, str(RUNNER), "--log", str(log),
                 "--timeout", str(timeout), "--", sys.executable, "-u", "-c", source],
                capture_output=True, text=True, timeout=timeout + 15)
            return result, log.read_text()

    def test_large_failure_log_is_retained_and_exit_code_preserved(self):
        """A log exceeding the runner's 4 MB cap still exposes the final error."""
        result, log = self.run_command(
            'print("warning\\n" * 700000); print("FINAL LINK ERROR"); raise SystemExit(7)')
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertGreater(len(log), 4 * 1024 * 1024)
        self.assertIn("FINAL LINK ERROR", result.stdout)
        self.assertLess(len(result.stdout), 15000)

    def test_success_preserves_test_summary(self):
        """A successful test command remains successful and shows its summary."""
        result, log = self.run_command('print("140 tests passed")')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("140 tests passed", result.stdout)
        self.assertIn("140 tests passed", log)

    def test_timeout_preserves_output_and_returns_failure(self):
        """A hung command exits early so the runner can upload caches and logs."""
        result, log = self.run_command(
            'import time; print("compile started"); time.sleep(60)', timeout=0.2)
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assertIn("compile started", log)
        self.assertIn("timed out", result.stdout)


if __name__ == "__main__":
    unittest.main()
