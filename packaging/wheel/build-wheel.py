#!/usr/bin/env python3
"""Build the csl-pp wheel from an installation of a static release build.

The wheel contains csl-pp, csl-lsp-server, libinject_mllvm.so and the csl++
libraries in the layout of the installation prefix, below the Python package
csl_pp, plus the launchers in packaging/wheel/csl_pp. Its license files are
LICENSE.TXT and the notices of the linked dependencies.

The debug information of the native files goes into a separate archive
<name>-<version>-debug-<platform>.tar.xz; the shipped files keep their symbol
tables, a build ID and a .gnu_debuglink to their debug file. Before packaging,
every native file is checked against the platform tag: it may only require
the shared libraries and glibc symbol versions that the manylinux policy of
the tag allows, and no libstdc++ or libgcc_s, which a static release build
links into the executables.

Usage:
  cmake --install build/Release --prefix stage
  packaging/wheel/build-wheel.py --prefix stage --version 0.1.0 --out-dir dist

The version is normalized according to PEP 440 (v1.2.0-rc.1 becomes 1.2.0rc1).
It must be the release that `csl-pp --version` reports, or a development
version (0.0.0.dev1) for a development build.
Requires the Python package `packaging` and objcopy and objdump from binutils
(override with $OBJCOPY and $OBJDUMP).
"""

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

from packaging.version import Version

NAME = "csl-pp"
PACKAGE = "csl_pp"
SUMMARY = "Compiler that translates csl++ to the Cerebras Software Language (CSL)"
LICENSE_EXPRESSION = "Apache-2.0 WITH LLVM-exception"
SOURCE_URL = "https://github.com/esa-tu-darmstadt/csl-pp"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: C++",
    "Topic :: Software Development :: Compilers",
]
ENTRY_POINTS = {
    "csl-pp": "csl_pp._launcher:csl_pp",
    "csl-lsp-server": "csl_pp._launcher:csl_lsp_server",
}

REPO = Path(__file__).resolve().parents[2]
PYTHON_SOURCES = REPO / "packaging" / "wheel"

# Files of the installation prefix that the wheel ships. Native files are
# split and checked; directories are copied recursively.
NATIVE_FILES = ["bin/csl-pp", "bin/csl-lsp-server", "lib/libinject_mllvm.so"]
EXECUTABLES = {"bin/csl-pp", "bin/csl-lsp-server"}
DATA_DIRECTORIES = ["share/csl-mlir/libs"]
LICENSE_FILE = "share/csl-mlir/LICENSE.TXT"
THIRD_PARTY_LICENSES = "share/csl-mlir/third-party-licenses"

# Shared libraries that every manylinux policy allows (PEP 600); csl-pp needs
# no others.
ALLOWED_LIBRARIES = {
    "libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0", "librt.so.1",
    "ld-linux-x86-64.so.2",
}
# Symbol version prefixes of the C++ runtime, which must be linked statically.
FORBIDDEN_VERSIONS = ("GLIBCXX_", "CXXABI_", "GCC_")


def run(*command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def glibc_of_platform(platform):
    match = re.fullmatch(r"manylinux_(\d+)_(\d+)_x86_64", platform)
    if not match:
        sys.exit(f"error: unsupported platform tag '{platform}'")
    return int(match[1]), int(match[2])


def check_native(path, max_glibc, objdump):
    """Reject a file that would not run on every system of the platform tag."""
    headers = run(objdump, "-p", str(path))
    errors = []
    for library in re.findall(r"^\s*NEEDED\s+(\S+)", headers, re.MULTILINE):
        if library not in ALLOWED_LIBRARIES:
            errors.append(f"requires {library}")
    for version in sorted(set(re.findall(r"\s([A-Z]+_[0-9.]+)$", headers, re.MULTILINE))):
        if version.startswith(FORBIDDEN_VERSIONS):
            errors.append(f"requires the shared C++ runtime ({version})")
        elif version.startswith("GLIBC_"):
            numbers = tuple(int(n) for n in version[len("GLIBC_"):].split("."))
            if numbers[:2] > max_glibc:
                errors.append(f"requires {version}, newer than glibc "
                              f"{max_glibc[0]}.{max_glibc[1]}")
    if "Build ID" not in run(os.environ.get("READELF", "readelf"), "-n", str(path)):
        errors.append("has no build ID (link with --build-id)")
    if errors:
        sys.exit(f"error: {path}: " + "; ".join(errors))


def split_debug_info(source, stripped, debug, objcopy):
    """Write the debug information to debug and the rest to stripped."""
    run(objcopy, "--only-keep-debug", "--decompress-debug-sections",
        str(source), str(debug))
    # The debuglink records the file name of the debug file and its CRC;
    # debuggers look for it next to the binary or in a debug directory.
    run(objcopy, "--strip-debug", f"--add-gnu-debuglink={debug}",
        str(source), str(stripped))


def check_release_version(executable, version):
    """Reject a wheel whose version differs from the release built into csl-pp."""
    first_line = run(str(executable), "--version").splitlines()[0]
    match = re.fullmatch(r"csl-pp \(csl\+\+\) (.+)", first_line)
    if not match:
        sys.exit(f"error: unexpected output of {executable} --version: {first_line}")
    if match[1] == "development build":
        if not version.is_devrelease:
            sys.exit(f"error: {executable} is a development build, but {version} is a "
                     "release; build it with -c user.csl-mlir:release_version="
                     f"{version}, or give the wheel a .dev version")
    elif Version(match[1]) != version:
        sys.exit(f"error: {executable} is release {match[1]}, but the wheel version "
                 f"is {version}")


def record_hash(data):
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


class WheelWriter:
    """Write a wheel with fixed timestamps and a RECORD of its contents."""

    def __init__(self, path, dist_info, timestamp):
        self.zip = zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED)
        self.dist_info = dist_info
        self.date_time = time.gmtime(max(timestamp, 315532800))[:6]
        self.records = []

    def add(self, name, data, executable=False):
        info = zipfile.ZipInfo(name, self.date_time)
        info.external_attr = (0o100755 if executable else 0o100644) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        self.zip.writestr(info, data)
        self.records.append((name, record_hash(data), str(len(data))))

    def close(self):
        record = f"{self.dist_info}/RECORD"
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerows(self.records)
        writer.writerow((record, "", ""))
        info = zipfile.ZipInfo(record, self.date_time)
        info.external_attr = 0o100644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        self.zip.writestr(info, buffer.getvalue())
        self.zip.close()


def metadata(version, license_files, documentation_url):
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {NAME}",
        f"Version: {version}",
        f"Summary: {SUMMARY}",
        f"License-Expression: {LICENSE_EXPRESSION}",
        *(f"License-File: {name}" for name in license_files),
        "Requires-Python: >=3.8",
        f"Project-URL: Source, {SOURCE_URL}",
        *([f"Project-URL: Documentation, {documentation_url}"] if documentation_url else []),
        *(f"Classifier: {classifier}" for classifier in CLASSIFIERS),
        "Description-Content-Type: text/markdown",
    ]
    readme = (PYTHON_SOURCES / "README.md").read_text()
    return "\n".join(lines) + "\n\n" + readme


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prefix", type=Path, required=True,
                        help="installation prefix of a static release build")
    parser.add_argument("--version", required=True, help="release version, e.g. 0.1.0")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--platform", default="manylinux_2_28_x86_64",
                        help="manylinux platform tag (default: %(default)s)")
    parser.add_argument("--documentation-url", help="URL of the documentation website")
    args = parser.parse_args()

    release = Version(args.version.removeprefix("v"))
    version = str(release)
    max_glibc = glibc_of_platform(args.platform)
    objcopy = os.environ.get("OBJCOPY", "objcopy")
    objdump = os.environ.get("OBJDUMP", "objdump")
    timestamp = int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))

    distribution = NAME.replace("-", "_")
    dist_info = f"{distribution}-{version}.dist-info"
    wheel_name = f"{distribution}-{version}-py3-none-{args.platform}.whl"
    debug_name = f"{NAME}-{version}-debug-{args.platform}"
    check_release_version(args.prefix / "bin" / "csl-pp", release)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        debug_dir = scratch / debug_name
        debug_dir.mkdir()
        wheel = WheelWriter(args.out_dir / wheel_name, dist_info, timestamp)

        for source in sorted((PYTHON_SOURCES / PACKAGE).glob("*.py")):
            wheel.add(f"{PACKAGE}/{source.name}", source.read_bytes())

        for name in NATIVE_FILES:
            source = args.prefix / name
            stripped = scratch / Path(name).name
            debug = debug_dir / (Path(name).name + ".debug")
            split_debug_info(source, stripped, debug, objcopy)
            check_native(stripped, max_glibc, objdump)
            wheel.add(f"{PACKAGE}/{name}", stripped.read_bytes(),
                      executable=name in EXECUTABLES)

        for directory in DATA_DIRECTORIES:
            root = args.prefix / directory
            if not root.is_dir():
                sys.exit(f"error: {root} does not exist")
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                relative = path.relative_to(args.prefix).as_posix()
                wheel.add(f"{PACKAGE}/{relative}", path.read_bytes())

        license_files = ["LICENSE.TXT"]
        wheel.add(f"{dist_info}/licenses/LICENSE.TXT",
                  (args.prefix / LICENSE_FILE).read_bytes())
        third_party = args.prefix / THIRD_PARTY_LICENSES
        if not third_party.is_dir():
            sys.exit(f"error: {third_party} does not exist; build with Conan, "
                     "which collects the license notices of the dependencies")
        for path in sorted(p for p in third_party.rglob("*") if p.is_file()):
            name = "third-party/" + path.relative_to(third_party).as_posix()
            license_files.append(name)
            wheel.add(f"{dist_info}/licenses/{name}", path.read_bytes())

        wheel.add(f"{dist_info}/METADATA",
                  metadata(version, license_files, args.documentation_url).encode())
        wheel.add(f"{dist_info}/WHEEL", (
            "Wheel-Version: 1.0\n"
            "Generator: csl-mlir packaging/wheel/build-wheel.py\n"
            "Root-Is-Purelib: false\n"
            f"Tag: py3-none-{args.platform}\n").encode())
        entry_points = "[console_scripts]\n" + "".join(
            f"{command} = {target}\n" for command, target in ENTRY_POINTS.items())
        wheel.add(f"{dist_info}/entry_points.txt", entry_points.encode())
        wheel.close()

        with tarfile.open(args.out_dir / f"{debug_name}.tar.xz", "w:xz") as archive:
            archive.add(debug_dir, arcname=debug_name)

    print(args.out_dir / wheel_name)
    print(args.out_dir / f"{debug_name}.tar.xz")


if __name__ == "__main__":
    main()
