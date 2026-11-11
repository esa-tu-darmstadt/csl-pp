# -*- Python -*-

import os
import lit.formats
import lit.util

# Configuration file for the 'lit' test runner.

execute_external = False

# Process paths related to the custom opt tool
config.csl_mlir_opt_tool_bin_path = os.path.normpath(config.csl_mlir_opt_tool_bin_path)
csl_mlir_opt_tool_bin_dir = os.path.abspath(
    os.path.dirname(config.csl_mlir_opt_tool_bin_path)
)
ld_lib_path = os.path.pathsep.join(
    (csl_mlir_opt_tool_bin_dir, config.environment.get("LD_LIBRARY_PATH", ""))
)

# Set actual lit-config
# Note: Setting config.test_exec_root is needed to ensure that test-output is
# not written into the source directory (but into "build").
config.name = "csl-mlir"
config.environment["LD_LIBRARY_PATH"] = ld_lib_path

# Add tools directory to PATH for FileCheck and csl-opt
path = os.environ.get("PATH", "")
config.environment["PATH"] = os.pathsep.join(
    (csl_mlir_opt_tool_bin_dir, config.llvm_tools_dir, path)
)

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)
config.test_source_ex = os.path.normpath(
    os.path.join(config.test_source_root, "test-resources")
)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.normpath(csl_mlir_opt_tool_bin_dir)

config.test_format = lit.formats.ShTest(execute_external)
config.suffixes = [".mlir", ".csl"]
config.excludes = ["Integration"]
config.substitutions.append(("%optcall", config.csl_mlir_opt_tool_bin_path))

# Set a default per-test timeout of 10 seconds. Setting a timeout per test
# requires that killProcessAndChildren() is supported on the platform and
# lit complains if the value is set but it is not supported.
supported, errormsg = lit_config.maxIndividualTestTimeIsSupported
if supported:
    config.maxIndividualTestTime = 10
else:
    lit_config.warning("Could not set a default per-test timeout. " + errormsg)
