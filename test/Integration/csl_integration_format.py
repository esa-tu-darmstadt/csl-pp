"""Custom LIT test format for CSL SDK integration tests.

Discovers benchmark and tutorial examples from csl-extras and runs their
commands_wse{2,3}.sh scripts through a cslc wrapper that transparently
invokes csl-pp.
"""

import fnmatch
import os
import signal
import shutil
import subprocess
import tempfile
import threading

import lit.formats
import lit.Test
import lit.util

# Directories and files to skip when copying examples to temp dirs.
# These are build/simulation artifacts that can be very large (tens of GB).
_IGNORE_PATTERNS = [
    "simfab_traces",
    "out",
    "wio_flows_tmpdir*",
    "sim.log",
    "sim_stats.json",
]

# File tracking active process groups spawned by integration tests.
_PGID_FILE = os.path.join(tempfile.gettempdir(), "csl-int-pgids")
_pgid_lock = threading.Lock()


def _docs_examples_dir(config):
    """The examples of the documentation website."""
    return os.path.join(
        config.test_source_root, os.pardir, os.pardir, "website", "examples"
    )


def _ignore_artifacts(directory, contents):
    ignored = set()
    for pattern in _IGNORE_PATTERNS:
        ignored.update(fnmatch.filter(contents, pattern))
    return ignored


def _record_pgid(pgid):
    """Append a process group ID to the tracking file."""
    with _pgid_lock:
        with open(_PGID_FILE, "a") as f:
            f.write(f"{pgid}\n")


def _remove_pgid(pgid):
    """Remove a process group ID from the tracking file."""
    with _pgid_lock:
        try:
            with open(_PGID_FILE, "r") as f:
                pgids = f.read().splitlines()
            pgids = [p for p in pgids if p != str(pgid)]
            with open(_PGID_FILE, "w") as f:
                f.write("\n".join(pgids) + "\n" if pgids else "")
        except FileNotFoundError:
            pass


def _cleanup_stale_processes_and_dirs():
    """Kill leftover process groups and remove temp dirs from previous runs."""
    # Kill tracked process groups.
    try:
        with open(_PGID_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    os.killpg(int(line), signal.SIGKILL)
                except (OSError, ValueError):
                    pass
        os.remove(_PGID_FILE)
    except FileNotFoundError:
        pass

    # Remove leftover temp dirs.
    tmp = tempfile.gettempdir()
    for name in os.listdir(tmp):
        if name.startswith("csl-int-"):
            path = os.path.join(tmp, name)
            try:
                shutil.rmtree(path)
            except OSError:
                pass


class CslIntegrationTest(lit.formats.TestFormat):
    """Discovers and runs CSL SDK example command scripts."""

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig, localConfig):
        # Only discover tests from the suite root; lit recurses into
        # subdirectories (e.g. __pycache__, bin) and calls us again - skip those.
        if path_in_suite:
            return

        # Clean up processes and temp dirs left behind by previous interrupted runs.
        _cleanup_stale_processes_and_dirs()

        # --- SDK examples (benchmarks/tutorials from csl-extras) ---
        extras_dir = localConfig.csl_extras_dir
        examples_dir = os.path.join(extras_dir, "examples")

        for category in ("benchmarks", "tutorials"):
            category_dir = os.path.join(examples_dir, category)
            if not os.path.isdir(category_dir):
                continue

            for example_name in sorted(os.listdir(category_dir)):
                example_dir = os.path.join(category_dir, example_name)
                if not os.path.isdir(example_dir):
                    continue

                for wse_ver in ("2", "3"):
                    cmd_file = os.path.join(example_dir, f"commands_wse{wse_ver}.sh")
                    if not os.path.isfile(cmd_file):
                        continue

                    yield lit.Test.Test(
                        testSuite,
                        (category, example_name, f"wse{wse_ver}"),
                        localConfig,
                    )

        # --- Coroutine benchmarks (in-tree under coro-benchmarks/) ---
        # Each coro test is run twice: once with --coro-mode=trampoline,
        # once with --coro-mode=fiber.
        coro_dir = os.path.join(localConfig.test_source_root, "coro-benchmarks")
        if os.path.isdir(coro_dir):
            for example_name in sorted(os.listdir(coro_dir)):
                example_dir = os.path.join(coro_dir, example_name)
                if not os.path.isdir(example_dir):
                    continue

                for wse_ver in ("2", "3"):
                    cmd_file = os.path.join(
                        example_dir, f"commands_wse{wse_ver}.sh"
                    )
                    if not os.path.isfile(cmd_file):
                        continue

                    for coro_mode in ("trampoline", "fiber"):
                        yield lit.Test.Test(
                            testSuite,
                            (
                                "coro-benchmarks",
                                example_name,
                                f"wse{wse_ver}",
                                coro_mode,
                            ),
                            localConfig,
                        )

        # --- Non-coroutine benchmarks (in-tree under non-coro-benchmarks/) ---
        # Mirrors the coro-benchmarks layout but without coro_mode iteration.
        # Imports system modules (e.g. <collectives_2d/pe>) from the SDK,
        # so no symlink to benchmark-libs is needed.
        non_coro_dir = os.path.join(
            localConfig.test_source_root, "non-coro-benchmarks"
        )
        if os.path.isdir(non_coro_dir):
            for example_name in sorted(os.listdir(non_coro_dir)):
                example_dir = os.path.join(non_coro_dir, example_name)
                if not os.path.isdir(example_dir):
                    continue

                for wse_ver in ("2", "3"):
                    cmd_file = os.path.join(
                        example_dir, f"commands_wse{wse_ver}.sh"
                    )
                    if not os.path.isfile(cmd_file):
                        continue

                    yield lit.Test.Test(
                        testSuite,
                        (
                            "non-coro-benchmarks",
                            example_name,
                            f"wse{wse_ver}",
                        ),
                        localConfig,
                    )

        # --- Documentation examples (website/examples/<group>/<name>/) ---
        # Each example with commands_wse{2,3}.sh scripts is a test.
        docs_dir = _docs_examples_dir(localConfig)
        if os.path.isdir(docs_dir):
            for group in sorted(os.listdir(docs_dir)):
                group_dir = os.path.join(docs_dir, group)
                if not os.path.isdir(group_dir):
                    continue
                for example_name in sorted(os.listdir(group_dir)):
                    example_dir = os.path.join(group_dir, example_name)
                    for wse_ver in ("2", "3"):
                        cmd_file = os.path.join(
                            example_dir, f"commands_wse{wse_ver}.sh"
                        )
                        if not os.path.isfile(cmd_file):
                            continue
                        yield lit.Test.Test(
                            testSuite,
                            ("docs", group, example_name, f"wse{wse_ver}"),
                            localConfig,
                        )

        # --- In-tree tests (e.g. resource-alloc/) ---
        # Discovers test directories directly under test/Integration/ that have
        # commands_wse{2,3}.sh scripts. No coro_mode iteration.
        _INTREE_SKIP = {"coro-benchmarks", "non-coro-benchmarks", "bin", "__pycache__"}
        for entry in sorted(os.listdir(localConfig.test_source_root)):
            if entry in _INTREE_SKIP:
                continue
            entry_dir = os.path.join(localConfig.test_source_root, entry)
            if not os.path.isdir(entry_dir):
                continue
            for wse_ver in ("2", "3"):
                cmd_file = os.path.join(entry_dir, f"commands_wse{wse_ver}.sh")
                if not os.path.isfile(cmd_file):
                    continue
                yield lit.Test.Test(
                    testSuite,
                    ("intree", entry, f"wse{wse_ver}"),
                    localConfig,
                )

    def execute(self, test, litConfig):
        config = test.config

        # Recover category/name/wseN[/coro_mode] from the test path.
        parts = test.path_in_suite
        is_coro = parts[0] == "coro-benchmarks"
        is_non_coro = parts[0] == "non-coro-benchmarks"
        is_intree = parts[0] == "intree"
        is_docs = parts[0] == "docs"
        if is_coro:
            # coro-benchmarks have an extra coro_mode component.
            category, example_name, wse_tag, coro_mode = (
                parts[-4], parts[-3], parts[-2], parts[-1],
            )
        else:
            category, example_name, wse_tag = parts[-3], parts[-2], parts[-1]
            coro_mode = None
        wse_ver = wse_tag[-1]  # "2" or "3"
        if is_coro:
            coro_dir = os.path.join(config.test_source_root, "coro-benchmarks")
            example_src = os.path.join(coro_dir, example_name)
        elif is_non_coro:
            non_coro_dir = os.path.join(
                config.test_source_root, "non-coro-benchmarks"
            )
            example_src = os.path.join(non_coro_dir, example_name)
        elif is_intree:
            example_src = os.path.join(config.test_source_root, example_name)
        elif is_docs:
            example_src = os.path.join(
                _docs_examples_dir(config), category, example_name
            )
        else:
            examples_dir = os.path.join(config.csl_extras_dir, "examples")
            example_src = os.path.join(examples_dir, category, example_name)

        cmd_file = f"commands_wse{wse_ver}.sh"

        if not os.path.isfile(os.path.join(example_src, cmd_file)):
            return lit.Test.Result(
                lit.Test.UNSUPPORTED,
                f"Command file {cmd_file} not found",
            )

        # Copy example to a temporary directory so scripts can write freely.
        tmp_dir = tempfile.mkdtemp(prefix=f"csl-int-{example_name}-")
        try:
            work_dir = os.path.join(tmp_dir, example_name)
            shutil.copytree(example_src, work_dir, ignore=_ignore_artifacts)

            # Symlink the library directories next to the temp copy, so that
            # relative imports such as "../../csl-libs/..." resolve as they do in
            # the tree. The csl++ libraries live in csl-libs/ at the repository
            # root; the SDK benchmarks use the SDK's benchmark-libs/.
            examples_dir = os.path.join(config.csl_extras_dir, "examples")
            cslpp_libs = os.path.normpath(
                os.path.join(config.test_source_root, "..", "..", "csl-libs")
            )
            if is_coro or is_docs:
                benchmark_libs = cslpp_libs
                os.symlink(cslpp_libs, os.path.join(tmp_dir, "csl-libs"))
            elif is_non_coro:
                # The in-tree baselines were vendored from the SDK benchmarks
                # and use the SDK's libraries.
                benchmark_libs = os.path.join(
                    examples_dir, "benchmarks", "benchmark-libs"
                )
            else:
                benchmark_libs = os.path.join(examples_dir, category, "benchmark-libs")
            if not (is_coro or is_docs) and os.path.isdir(benchmark_libs):
                os.symlink(benchmark_libs, os.path.join(tmp_dir, "benchmark-libs"))
            # Also symlink benchmark-libs-orig/ for the experiments that mix
            # the csl++ libraries with the SDK's callback-based ones.
            if is_coro:
                benchmark_libs_orig = os.path.join(coro_dir, "benchmark-libs-orig")
                if os.path.isdir(benchmark_libs_orig):
                    os.symlink(benchmark_libs_orig, os.path.join(tmp_dir, "benchmark-libs-orig"))

            env = dict(config.environment)
            env["PATH"] = config.integration_path
            if coro_mode:
                env["CORO_MODE"] = coro_mode

            # The copied example may contain an absolute symlink to a shared
            # benchmark library outside the temporary working directory. Host
            # SDK wrappers use CSL_IMPORT_PATH to make those directories
            # visible inside their execution environment.
            import_paths = []
            inherited_import_path = env.get("CSL_IMPORT_PATH")
            if inherited_import_path:
                import_paths.extend(inherited_import_path.split(os.pathsep))
            if os.path.isdir(benchmark_libs):
                import_paths.append(os.path.realpath(benchmark_libs))
            env["CSL_IMPORT_PATH"] = os.pathsep.join(
                dict.fromkeys(path for path in import_paths if path)
            )

            # Spawn in a new session so we can track and kill the whole tree.
            p = subprocess.Popen(
                ["bash", cmd_file],
                cwd=work_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                close_fds=True,
                start_new_session=True,
            )
            _record_pgid(p.pid)

            timed_out = False
            try:
                out, err = p.communicate(
                    timeout=config.csl_integration_test_timeout
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
                out, err = p.communicate()
            finally:
                _remove_pgid(p.pid)

            exit_code = p.returncode
            out = out.decode("utf-8", errors="replace")
            err = err.decode("utf-8", errors="replace")

            output = ""
            if out.strip():
                output += f"-- stdout --\n{out}\n"
            if err.strip():
                output += f"-- stderr --\n{err}\n"

            # Append simulation log if available.
            sim_log = os.path.join(work_dir, "out/sim.log")
            if os.path.isfile(sim_log):
                with open(sim_log, "r") as f:
                    output += f"-- sim.log --\n{f.read()}\n"

            # Append simulation stats if available.
            sim_stats = os.path.join(work_dir, "sim_stats.json")
            if os.path.isfile(sim_stats):
                with open(sim_stats, "r") as f:
                    output += f"-- sim_stats.json --\n{f.read()}\n"

            if exit_code != 0:
                if timed_out:
                    output = (
                        f"Timed out after {config.csl_integration_test_timeout} "
                        f"seconds\n{output}"
                    )
                return lit.Test.Result(
                    lit.Test.FAIL,
                    f"Exit code: {exit_code}\n{output}",
                )

            return lit.Test.Result(lit.Test.PASS, output)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
