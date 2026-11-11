#!/usr/bin/env cs_python
"""Minimal fiber test runner.

Launches f_start on a single PE and verifies expected output in sim.log.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime,
)  # pylint: disable=no-name-in-module


def parse_messages(log_content):
    """Extract message strings from sim.log, stripping cycle and PE prefixes."""
    messages = []
    for line in log_content.strip().split("\n"):
        m = re.match(r"@\d+ (?:PE\(\d+,\d+\): )?(.*)", line)
        if m:
            messages.append(m.group(1))
    return messages


def verify(messages):
    """Check that all expected outputs appear in the correct order."""
    errors = []

    ordered = [
        "T1:begin",
        "T1:10",  # simple_void sets result=10
        "T2:begin",
        "T2:42",  # add_one(41) = 42
        "PASS",
    ]

    idx = 0
    for expected in ordered:
        found = False
        while idx < len(messages):
            if messages[idx] == expected:
                found = True
                idx += 1
                break
            idx += 1
        if not found:
            errors.append(f"Missing or out-of-order: '{expected}'")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="compile output dir")
    parser.add_argument("--cmaddr", default=None, help="IP:port for CS system")
    args = parser.parse_args()

    dirname = args.name

    simulator = SdkRuntime(dirname, cmaddr=args.cmaddr)
    simulator.load()
    simulator.run()

    simulator.launch("f_start", nonblock=False)

    simulator.stop()

    if args.cmaddr is None:
        dst = Path(f"{dirname}/sim.log")
        src = Path("sim.log")
        if src.exists():
            shutil.move(src, dst)

    sim_log = Path(f"{dirname}/sim.log")
    if not sim_log.exists():
        print("ERROR: sim.log not found", file=sys.stderr)
        sys.exit(1)

    log_content = sim_log.read_text(errors="replace")
    print(log_content)

    messages = parse_messages(log_content)
    errors = verify(messages)

    if errors:
        print("\n=== VERIFICATION ERRORS ===", file=sys.stderr)
        for e in errors:
            print(f"  FAIL: {e}", file=sys.stderr)
        sys.exit(1)

    print("SUCCESS!")


if __name__ == "__main__":
    main()
