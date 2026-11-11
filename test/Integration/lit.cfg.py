# -*- Python -*-

"""LIT configuration for CSL SDK integration tests."""

import glob
import os
import stat
import sys

import lit.formats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

config.name = "csl-mlir-integration"
config.suffixes = []  # No file-based discovery; custom format handles it.
config.excludes = ["__pycache__", "bin"]

config.test_exec_root = os.path.dirname(__file__)
config.test_source_root = os.path.dirname(__file__)

# Guard: when auto-discovered by the main test suite (without the site config),
# the required attributes won't be set.  Return an empty test format.
if not hasattr(config, "csl_pp_bin_path"):
    config.test_format = lit.formats.ShTest(False)
    config.unsupported = True
else:

    # Import the format from a proper module so it can be pickled by
    # lit's multiprocessing workers.
    sys.path.insert(0, os.path.dirname(__file__))
    from csl_integration_format import CslIntegrationTest

    config.test_format = CslIntegrationTest()

    # -------------------------------------------------------------------
    # cslc wrapper generation
    # -------------------------------------------------------------------

    csl_pp_path = config.csl_pp_bin_path
    sdk_dir = config.csl_sdk_dir
    host_tools_dir = config.csl_sdk_host_tools_dir
    bin_dir = config.csl_mlir_bin_dir

    try:
        max_parallelism = int(config.csl_integration_max_parallelism)
    except ValueError:
        lit_config.fatal("CSL_INTEGRATION_MAX_PARALLELISM must be an integer")
    if max_parallelism < 1:
        lit_config.fatal("CSL_INTEGRATION_MAX_PARALLELISM must be positive")
    lit_config.parallelism_groups["csl-sdk"] = max_parallelism
    config.parallelism_group = "csl-sdk"

    try:
        test_timeout = int(config.csl_integration_test_timeout)
    except ValueError:
        lit_config.fatal("CSL_INTEGRATION_TEST_TIMEOUT must be an integer")
    if test_timeout < 1:
        lit_config.fatal("CSL_INTEGRATION_TEST_TIMEOUT must be positive")
    config.csl_integration_test_timeout = test_timeout

    # Locate the .sif file in the SDK directory.
    sif_files = glob.glob(os.path.join(sdk_dir, "*.sif"))
    if not sif_files:
        lit_config.fatal(f"No .sif file found in CSL_SDK_DIR: {sdk_dir}")
    sif_path = sif_files[0]

    for tool in ("cslc", "cs_python"):
        tool_path = os.path.join(host_tools_dir, tool)
        if not os.path.isfile(tool_path) or not os.access(tool_path, os.X_OK):
            lit_config.fatal(
                f"Host SDK tool is not executable: {tool_path}. "
                "Set CSL_SDK_HOST_TOOLS_DIR to a directory containing "
                "host-runnable cslc and cs_python wrappers."
            )

    # Create wrapper bin directory inside the test execution root.
    wrapper_bin_dir = os.path.join(config.test_exec_root, "bin")
    os.makedirs(wrapper_bin_dir, exist_ok=True)

    # Write cslc wrapper script.
    # When CORO_MODE is set (by coro-benchmark tests), pass --coro-mode.
    wrapper_path = os.path.join(wrapper_bin_dir, "cslc")
    with open(wrapper_path, "w") as f:
        f.write(
            f"""#!/bin/bash
CORO_ARGS=""
if [ -n "${{CORO_MODE:-}}" ]; then
    CORO_ARGS="--coro-mode=$CORO_MODE"
fi
exec {csl_pp_path} --cslc={host_tools_dir}/cslc --system-module-sif={sif_path} $CORO_ARGS "$@"
"""
        )
    os.chmod(
        wrapper_path,
        os.stat(wrapper_path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH,
    )

    # The documentation examples call csl-pp the way a user does. Point it at
    # the SDK under test and at the libraries of the repository, which the
    # user adds with --import-path.
    csl_pp_wrapper_path = os.path.join(wrapper_bin_dir, "csl-pp")
    libs_dir = os.path.normpath(
        os.path.join(config.test_source_root, os.pardir, os.pardir, "csl-libs")
    )
    with open(csl_pp_wrapper_path, "w") as f:
        f.write(
            f"""#!/bin/bash
exec {csl_pp_path} --cslc={host_tools_dir}/cslc --system-module-sif={sif_path} --import-path={libs_dir} "$@"
"""
        )
    os.chmod(
        csl_pp_wrapper_path,
        os.stat(csl_pp_wrapper_path).st_mode
        | stat.S_IEXEC
        | stat.S_IXGRP
        | stat.S_IXOTH,
    )

    # -------------------------------------------------------------------
    # Environment: PATH and LD_LIBRARY_PATH
    # -------------------------------------------------------------------

    system_path = os.environ.get("PATH", "")
    config.integration_path = os.pathsep.join(
        [
            wrapper_bin_dir,
            host_tools_dir,
            bin_dir,
            sdk_dir,
            system_path,
        ]
    )

    ld_lib_path = os.pathsep.join(
        p for p in [bin_dir, os.environ.get("LD_LIBRARY_PATH", "")] if p
    )
    config.environment["PATH"] = config.integration_path
    config.environment["LD_LIBRARY_PATH"] = ld_lib_path
