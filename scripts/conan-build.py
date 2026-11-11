#!/usr/bin/env python3
"""Export the local recipes, then run Conan's normal build workflow.

The LLVM recipe builds the commit of the tracked llvm gitlink. When the
submodule is initialized it is used as the source; otherwise LLVM is cloned
only if its binary has to be built. With --upload, every package of the
resolved dependency graph is uploaded to the given remote afterwards.

Examples: scripts/conan-build.py install --build=missing -pr:a rhel8-gcc15
          scripts/conan-build.py create -pr:a rhel8-gcc15
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conan_support import llvm_pin


def main():
    """Export local recipes and forward caller-selected profiles to Conan."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("export", "install", "build", "create", "lock"))
    parser.add_argument("--upload", metavar="REMOTE",
                        help="upload the resolved dependency graph (install/create only)")
    args, forwarded = parser.parse_known_args()
    llvm_pin(ROOT)
    for recipe in (ROOT / "conan/llvm", ROOT / "conan/squashfs-tools-ng"):
        subprocess.run(["conan", "export", str(recipe)], check=True)
    if args.command == "export":
        return
    if args.upload and args.command not in ("install", "create"):
        parser.error("--upload requires install or create")
    command = ["conan", args.command]
    if args.command == "lock":
        command += ["create"]
    command += [str(ROOT), *forwarded]
    # Use an initialized submodule only when it is exactly the pinned commit;
    # a stale checkout (for example a reused CI directory) is ignored and the
    # recipe clones the pin if LLVM has to be built.
    checkout = ROOT / "llvm"
    if (checkout / "llvm/CMakeLists.txt").is_file():
        head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        if head == llvm_pin(ROOT)["commit"]:
            command += ["-c:a", f"user.llvm:checkout={checkout}"]
    if not args.upload:
        subprocess.run(command, check=True)
        return
    with tempfile.TemporaryDirectory() as scratch:
        graph = Path(scratch) / "graph.json"
        packages = Path(scratch) / "packages.json"
        with graph.open("w") as output:
            subprocess.run([*command, "--format=json"], stdout=output, check=True)
        with packages.open("w") as output:
            subprocess.run(["conan", "list", "--graph", str(graph), "--graph-recipes=*",
                            "--graph-binaries=*", "--format=json"],
                           stdout=output, check=True)
        if json.loads(packages.read_text()).get("Local Cache"):
            upload(packages, args.upload)


def upload(packages, remote, attempts=3, pause=120):
    """Upload a package list, retrying while a parallel job uploads the same revisions.

    The static and shared jobs share recipe revisions. When both upload one,
    the registry rejects the second copy as a duplicate until the first is
    complete; a later attempt then skips it as already present.
    """
    for attempt in range(1, attempts + 1):
        result = subprocess.run(["conan", "upload", "--list", str(packages), "-r", remote,
                                 "--confirm"])
        if result.returncode == 0:
            return
        if attempt < attempts:
            print(f"Upload failed; retrying in {pause}s ({attempt}/{attempts})", flush=True)
            time.sleep(pause)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
