#!/usr/bin/env cs_python
"""Resource allocation test runner.

Runs the compiled test on the simulator and verifies that comptime and
runtime resource allocation produces correct IDs.
"""

import argparse
import re
import shutil
import sys
import time
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


def get_value(messages, prefix):
    """Find message starting with prefix and return the integer value after '='."""
    for m in messages:
        if m.startswith(prefix):
            return int(m.split("=")[1])
    return None


def verify(messages, arch):
    """Check invariants rather than exact values — robust against
    changes in memcpy resource blocking or allocation order."""
    errors = []

    ct_color = get_value(messages, "CT_COLOR=")
    ct_task = get_value(messages, "CT_TASK=")
    ct_iq = get_value(messages, "CT_IQ=")
    ct_oq = get_value(messages, "CT_OQ=")
    ct_ctrl = get_value(messages, "CT_CTRL=")
    rt_color = get_value(messages, "RT_COLOR=")
    rt_color2 = get_value(messages, "RT_COLOR2=")

    # All values must be present.
    for name, val in [("CT_COLOR", ct_color), ("CT_TASK", ct_task),
                      ("CT_IQ", ct_iq), ("CT_OQ", ct_oq),
                      ("CT_CTRL", ct_ctrl), ("RT_COLOR", rt_color),
                      ("RT_COLOR2", rt_color2)]:
        if val is None:
            errors.append(f"Missing {name}")
    if errors:
        return errors

    # Range checks.
    if not (0 <= ct_color <= 23):
        errors.append(f"CT_COLOR={ct_color} out of range 0-23")
    if arch == "wse2" and not (0 <= ct_task <= 28):
        errors.append(f"CT_TASK={ct_task} out of range 0-28 for WSE-2")
    if arch == "wse3" and not (8 <= ct_task <= 28):
        errors.append(f"CT_TASK={ct_task} out of range 8-28 for WSE-3")
    if not (0 <= ct_iq <= 7):
        errors.append(f"CT_IQ={ct_iq} out of range 0-7")
    oq_max = 5 if arch == "wse2" else 7
    if not (0 <= ct_oq <= oq_max):
        errors.append(f"CT_OQ={ct_oq} out of range 0-{oq_max}")
    if not (31 <= ct_ctrl <= 62):
        errors.append(f"CT_CTRL={ct_ctrl} out of range 31-62")

    # On WSE-2, color and task come from the same pool — must be distinct.
    if arch == "wse2" and ct_color == ct_task:
        errors.append(f"WSE-2: CT_COLOR and CT_TASK must differ (both={ct_color})")

    # Runtime color must differ from comptime color (bitmap was patched).
    if rt_color == ct_color:
        errors.append(f"RT_COLOR={rt_color} should differ from CT_COLOR={ct_color}")

    # Release + re-claim must return same ID.
    if rt_color2 != rt_color:
        errors.append(f"RT_COLOR2={rt_color2} should equal RT_COLOR={rt_color}")

    if "PASS" not in messages:
        errors.append("Missing PASS marker")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="compile output dir")
    parser.add_argument("--cmaddr", default=None, help="IP:port for CS system")
    parser.add_argument("--arch", required=True, choices=["wse2", "wse3"],
                        help="Target architecture")
    args = parser.parse_args()

    dirname = args.name
    arch = args.arch

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

    log_content = sim_log.read_text()
    print(log_content)

    messages = parse_messages(log_content)
    errors = verify(messages, arch)

    if errors:
        print("\n=== VERIFICATION ERRORS ===", file=sys.stderr)
        for e in errors:
            print(f"  FAIL: {e}", file=sys.stderr)
        sys.exit(1)

    print("SUCCESS!")


if __name__ == "__main__":
    main()
