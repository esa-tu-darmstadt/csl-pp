"""Read the LLVM gitlink without maintaining a second dependency pin."""

import json
from pathlib import Path
import posixpath
import re
import subprocess
from urllib.parse import urljoin, urlsplit, urlunsplit


def llvm_pin(folder):
    """Return the tracked LLVM commit and credential-free source URL.

    Exported recipes read their captured manifest. Working copies read the
    stage-zero gitlink; unresolved merges and missing pins are errors.
    """
    folder = Path(folder).resolve()
    manifest = folder / "llvm-pin.json"
    if manifest.is_file():
        pin = json.loads(manifest.read_text())
        if not re.fullmatch(r"[0-9a-f]{40}", pin["commit"]):
            raise ValueError("Invalid exported LLVM gitlink")
        return pin
    for root in (folder, *folder.parents):
        if not (root / ".gitmodules").is_file():
            continue
        result = subprocess.run(
            ["git", "-C", str(root), "config", "-f", ".gitmodules",
             "--get", "submodule.llvm.url"], text=True, capture_output=True)
        if result.returncode:
            continue
        entries = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--stage", "--", "llvm"],
            text=True).strip().splitlines()
        if len(entries) != 1:
            raise ValueError("LLVM needs one resolved, tracked submodule gitlink")
        match = re.fullmatch(r"160000 ([0-9a-f]{40}) 0\tllvm", entries[0])
        if not match:
            raise ValueError("LLVM must remain a tracked submodule")
        url = result.stdout.strip()
        if url.startswith(("../", "./")):
            origin = subprocess.check_output(
                ["git", "-C", str(root), "remote", "get-url", "origin"],
                text=True).strip()
            if "://" not in origin and ":" in origin:
                host, path = origin.split(":", 1)
                url = host + ":" + posixpath.normpath(posixpath.join(path, url))
            else:
                url = urljoin(origin.rstrip("/") + "/", url)
        parsed = urlsplit(url)
        if parsed.scheme in ("https", "http"):
            host = parsed.hostname or ""
            if parsed.port:
                host += ":" + str(parsed.port)
            url = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
        return {"commit": match[1], "url": url}
    raise ValueError("Cannot locate the LLVM gitlink or exported pin manifest")
