"""Custom LIT test format for lexer roundtrip tests.

Discovers .csl files from sibling test directories (Comptime, Roundtrip) and
creates a test for each that verifies byte-identical reconstruction through
the Parser3 lexer.
"""

import os
import subprocess

import lit.formats
import lit.Test


# Directories (relative to the test source root) to scan for .csl files.
_SCAN_DIRS = [
    "Comptime",
    "Roundtrip",
]


class LexerRoundtripTest(lit.formats.TestFormat):
    """Discovers .csl files and runs lexer roundtrip checks."""

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig, localConfig):
        # Only discover from the Lexer/ directory itself, not subdirectories.
        if path_in_suite != ("Lexer",):
            return

        source_root = testSuite.getSourcePath(())

        for scan_dir in _SCAN_DIRS:
            scan_path = os.path.join(source_root, scan_dir)
            if not os.path.isdir(scan_path):
                continue

            for dirpath, dirnames, filenames in os.walk(scan_path):
                dirnames.sort()
                for filename in sorted(filenames):
                    if not filename.endswith(".csl"):
                        continue

                    # Build a test path like Lexer/Comptime/arithmetic.csl
                    rel = os.path.relpath(
                        os.path.join(dirpath, filename), source_root
                    )
                    parts = tuple(rel.replace(os.sep, "/").split("/"))
                    test_path = ("Lexer",) + parts

                    yield lit.Test.Test(testSuite, test_path, localConfig)

    def execute(self, test, litConfig):
        config = test.config

        # Recover the original .csl file path from the test path.
        # test.path_in_suite is ("Lexer", "Comptime", "arithmetic.csl") etc.
        parts = test.path_in_suite[1:]  # strip "Lexer" prefix
        source_root = test.suite.getSourcePath(())
        csl_path = os.path.join(source_root, *parts)

        if not os.path.isfile(csl_path):
            return lit.Test.Result(
                lit.Test.UNRESOLVED, f"File not found: {csl_path}"
            )

        # Find csl-lex-roundtrip on PATH.
        env = dict(config.environment)

        try:
            proc = subprocess.run(
                ["csl-lex-roundtrip", csl_path],
                capture_output=True,
                env=env,
                timeout=10,
            )
        except FileNotFoundError:
            return lit.Test.Result(
                lit.Test.UNRESOLVED,
                "csl-lex-roundtrip not found on PATH",
            )
        except subprocess.TimeoutExpired:
            return lit.Test.Result(lit.Test.FAIL, "Timeout (10s)")

        if proc.returncode != 0:
            return lit.Test.Result(
                lit.Test.FAIL,
                f"csl-lex-roundtrip failed (exit {proc.returncode}):\n"
                + proc.stderr.decode("utf-8", errors="replace"),
            )

        # Read original file and compare.
        with open(csl_path, "rb") as f:
            original = f.read()

        if proc.stdout != original:
            # Find first difference for diagnostics.
            min_len = min(len(original), len(proc.stdout))
            diff_pos = next(
                (i for i in range(min_len) if original[i] != proc.stdout[i]),
                min_len,
            )
            return lit.Test.Result(
                lit.Test.FAIL,
                f"Roundtrip mismatch at byte {diff_pos}\n"
                f"  original length: {len(original)}\n"
                f"  reconstructed length: {len(proc.stdout)}",
            )

        return lit.Test.Result(lit.Test.PASS, "")
