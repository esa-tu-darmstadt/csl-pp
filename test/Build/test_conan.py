"""Check gitlink-driven dependency changes and installed-SDK rejection."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from conan_support import llvm_pin


class PinTests(unittest.TestCase):
    """Exercise real Git indexes instead of mocking source pin discovery."""

    def checkout(self, root, origin):
        """Create a fixture repository with an uninitialized LLVM submodule."""
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "remote", "add", "origin", origin], check=True)
        (root / ".gitmodules").write_text(
            '[submodule "llvm"]\npath = llvm\nurl = ../llvm-fork.git\n')
        subprocess.run(["git", "-C", str(root), "update-index", "--add", "--cacheinfo",
                        "160000," + "a" * 40 + ",llvm"], check=True)

    def test_gitlink_update_changes_requirement(self):
        """A gitlink update changes the dependency version without another pin."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.checkout(root, "https://example.test/cerebras/csl-mlir.git")
            self.assertEqual(llvm_pin(root), {
                "commit": "a" * 40, "url": "https://example.test/cerebras/llvm-fork.git"})
            subprocess.run(["git", "-C", str(root), "update-index", "--cacheinfo",
                            "160000," + "b" * 40 + ",llvm"], check=True)
            self.assertEqual(llvm_pin(root)["commit"], "b" * 40)

    def test_ssh_and_credential_free_urls(self):
        """Resolve relative URLs correctly and exclude HTTP credentials."""
        for origin, expected in (
                ("git@example.test:cerebras/csl-mlir.git",
                 "git@example.test:cerebras/llvm-fork.git"),
                ("https://ci:private@example.test/cerebras/csl-mlir.git",
                 "https://example.test/cerebras/llvm-fork.git")):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.checkout(root, origin)
                self.assertEqual(llvm_pin(root)["url"], expected)

    def test_exported_manifest_has_no_checkout_dependency(self):
        """An exported recipe retains the pin without a Git checkout."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {"commit": "c" * 40, "url": "https://example.test/llvm.git"}
            (root / "llvm-pin.json").write_text(json.dumps(expected))
            self.assertEqual(llvm_pin(root), expected)

    def test_unresolved_gitlink_is_rejected(self):
        """An unresolved submodule merge cannot choose a compiler version."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.checkout(root, "https://example.test/cerebras/csl-mlir.git")
            entries = ("0 " + "0" * 40 + "\tllvm\n" +
                       "160000 " + "a" * 40 + " 1\tllvm\n" +
                       "160000 " + "b" * 40 + " 2\tllvm\n")
            subprocess.run(["git", "-C", str(root), "update-index", "--index-info"],
                           input=entries, text=True, check=True)
            with self.assertRaisesRegex(ValueError, "resolved"):
                llvm_pin(root)


class ProvenanceTests(unittest.TestCase):
    """Run the actual CMake SDK validator against isolated package fixtures."""

    def validate(self, record):
        """Return CMake's diagnostic and exit status for a fixture SDK."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "share/llvm-sdk/provenance.json"
            if record is not None:
                manifest.parent.mkdir(parents=True)
                manifest.write_text(json.dumps(record))
            script = root / "check.cmake"
            script.write_text(
                'cmake_minimum_required(VERSION 3.25)\n'
                f'set(LLVM_INSTALL_PREFIX "{root}")\n'
                f'include("{ROOT}/cmake/VerifyLLVM.cmake")\n'
                f'csl_verify_llvm_revision("{"d" * 40}")\n')
            return subprocess.run(["cmake", "-P", str(script)], text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def test_accept_exact_pin_and_reject_other_sdks(self):
        """Reject a stale pin, unsupported schema, malformed or absent provenance."""
        result = self.validate({"schema": 1, "revision": "d" * 40})
        self.assertEqual(result.returncode, 0, result.stdout)
        for record in (None, {}, {"schema": 2, "revision": "d" * 40},
                       {"schema": 1, "revision": "e" * 40}):
            result = self.validate(record)
            self.assertNotEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
