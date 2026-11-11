"""Entry points that replace the Python process with a bundled executable.

The executables stay inside the package, because csl-pp finds
lib/libinject_mllvm.so relative to its own location, which it reads from
/proc/self/exe. argv[0] remains the name the command was invoked with, so
that messages name the command the user typed.
"""

import os
import sys

_PREFIX = os.path.dirname(os.path.abspath(__file__))


def _run(name):
    executable = os.path.join(_PREFIX, "bin", name)
    os.execv(executable, [sys.argv[0], *sys.argv[1:]])


def csl_pp():
    _run("csl-pp")


def csl_lsp_server():
    _run("csl-lsp-server")
