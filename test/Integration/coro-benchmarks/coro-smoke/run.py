#!/usr/bin/env cs_python
"""Coroutine smoke test runner.

Launches the test entry point on a single PE and verifies that all
expected markers appear in sim.log in the correct order.
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
        # Format: "@NNN message" or "@NNN PE(X,Y): message"
        m = re.match(r"@\d+ (?:PE\(\d+,\d+\): )?(.*)", line)
        if m:
            messages.append(m.group(1))
    return messages


def verify(messages):
    """Check that all expected outputs appear in the correct order."""
    errors = []

    # These must appear in this exact order.
    # Parallel outputs may interleave between their begin/done markers.
    ordered = [
        "T1:begin",
        "T1:ok",
        "T1:done",
        "T2:begin",
        "T2:42",
        "T3:begin",
        "T3:100",  # add_one(99) = 100
        "T3b:begin",
        "T3b:42",  # add_two(40, 2) = 42
        "T3c:begin",
        "T3c:60",  # add_three(10, 20, 30) = 60
        "T4:begin",
        "T4:1",
        "T4:2",
        "T4:done",
        "T5:begin",
        "T5:1",
        "T5:2",
        "T5:3",
        "T5:done",
        "T6:begin",
        "T6:done",
        "T7:begin",
        "T7:a=5",
        "T7:b=5",
        "T8:begin",
        "T8:done",
        "T9:begin",
        "T9:10",  # inner() + 3 = 10
        "T10:begin",
        "T10:a=11",
        "T10:b=7",  # @await inside @await_parallel
        "T11:begin",
        "T11:a=1",
        "T11:b=1",  # nested @await_parallel
        "T12:begin",
        r"T12:tid=\d+",  # inline callee runs on caller's task
        "T15:begin",
        "T15:10",  # 0+1+2+3+4 = 10
        "PASS",
    ]

    idx = 0
    for expected in ordered:
        found = False
        while idx < len(messages):
            if messages[idx] == expected or re.fullmatch(expected, messages[idx]):
                found = True
                idx += 1
                break
            idx += 1
        if not found:
            errors.append(f"Missing or out-of-order: '{expected}'")

    # Both parallel functions must execute (Test 6).
    for tag in ("T6:A", "T6:B"):
        if tag not in messages:
            errors.append(f"Test 6: missing '{tag}'")

    # All four suspend phases must appear (Test 8).
    for tag in ("T8:A1", "T8:A2", "T8:B1", "T8:B2"):
        if tag not in messages:
            errors.append(f"Test 8: missing '{tag}'")

    # Nested parallel leaf must execute (Test 11).
    if "T11:leaf" not in messages:
        errors.append("Test 11: missing 'T11:leaf'")

    # Test 15: clobber task must see sink=0 each time (we reset before suspend),
    # and after resume the async task must see a non-zero post_sink.
    sink_zero_count = sum(1 for m in messages if m == "T15:sink=0")
    if sink_zero_count != 5:
        errors.append(f"Test 15: expected 5x 'T15:sink=0', got {sink_zero_count}")
    post_sink_msgs = [m for m in messages if m.startswith("T15:post_sink=")]
    if len(post_sink_msgs) != 5:
        errors.append(f"Test 15: expected 5x 'T15:post_sink=...', got {len(post_sink_msgs)}")
    for m in post_sink_msgs:
        val = int(m.split("=")[1])
        if val == 0:
            errors.append(f"Test 15: post_sink should be non-zero, got '{m}'")

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

    # Launch the test entry point and wait for completion.
    simulator.launch("f_start", nonblock=False)

    simulator.stop()

    # Move sim.log into the output directory.
    if args.cmaddr is None:
        dst = Path(f"{dirname}/sim.log")
        src = Path("sim.log")
        if src.exists():
            shutil.move(src, dst)

    sim_log = Path(f"{dirname}/sim.log")
    if not sim_log.exists():
        print("ERROR: sim.log not found", file=sys.stderr)
        sys.exit(1)

    log_content = sim_log.read_text()
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
