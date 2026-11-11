#!/usr/bin/env python3
"""Keep complete build output in an artifact and bounded progress in CI.

An internal timeout leaves the runner time to upload logs and ccache. Failures
preserve the command's exit status; a timeout returns 124. No shell is used.
"""

import argparse
from collections import deque
import os
from pathlib import Path
import signal
import subprocess
import time


def tail(path, limit=12000):
    """Read at most the final limit bytes without loading a large build log."""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - limit))
        return stream.read().decode("utf-8", errors="replace")


def stop(process):
    """Terminate the command and its descendants, including compiler processes."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def interrupted(signum, frame):
    """Interrupt the wait so the command's process group can be cleaned up."""
    raise InterruptedError("Runner terminated the command")


def main():
    """Run a build with durable logs, periodic progress and an upload reserve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command or args.timeout <= 0:
        parser.error("A command and a positive timeout are required")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, interrupted)
    started = time.monotonic()
    print(f"Full output: {args.log}", flush=True)
    with args.log.open("wb") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while True:
                remaining = args.timeout - (time.monotonic() - started)
                if remaining <= 0:
                    stop(process)
                    code = 124
                    print("Command timed out; preserving time for cache/log upload.", flush=True)
                    break
                try:
                    code = process.wait(timeout=min(30, remaining))
                    break
                except subprocess.TimeoutExpired:
                    elapsed = int(time.monotonic() - started)
                    lines = tail(args.log, 2000).splitlines()
                    progress = lines[-1][-500:] if lines else "waiting for output"
                    print(f"[{elapsed}s] {progress}", flush=True)
        except (KeyboardInterrupt, InterruptedError) as error:
            stop(process)
            code = 130 if isinstance(error, KeyboardInterrupt) else 143
    elapsed = int(time.monotonic() - started)
    print(f"Command exited {code} after {elapsed}s. Full output: {args.log}", flush=True)
    # Successful logs still expose test summaries; failures show a bounded tail.
    print("\n".join(deque(tail(args.log).splitlines(), maxlen=120 if code else 20)), flush=True)
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
