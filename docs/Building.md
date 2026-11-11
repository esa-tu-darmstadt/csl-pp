# Building csl-mlir

csl-mlir is built with CMake against an installed LLVM/MLIR/LLD SDK. Conan
provides that SDK and the other dependencies, so a build needs no separately
installed LLVM.

## How dependencies are pinned

The `llvm` submodule pins the exact LLVM revision. The Conan recipe in
`conan/llvm/` packages that commit as `llvm/<commit>`, and `conanfile.py`
requires exactly this version, so the gitlink is the only place the LLVM
revision is recorded. The LLVM package installs LLVM's own CMake exports plus
native TableGen, FileCheck, lit and LLVM's test support libraries. CMake checks
the SDK's recorded revision against the gitlink and rejects any other LLVM.

`conan/squashfs-tools-ng/` packages the SIF reader dependency; all remaining
dependencies come from ConanCenter. `conan.lock` records the exact recipe
revision of every dependency.

csl-mlir itself is versioned by its Git commit (`csl-mlir/<commit>`); a build
from a working tree with uncommitted changes gets a `-dirty` suffix.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r conan/requirements.txt
conan config install conan/config
```

`conan/config` adds the `os.distro` setting and the `rhel8-gcc15` profile used
by CI: x86-64 Linux with glibc 2.28 (RHEL 8 compatible), GCC 15, libstdc++ and
C++20. Put a GCC 15 `gcc`/`g++` first on `PATH` to use it. Any other profile
works as well, for example a detected one with C++20:

```bash
conan profile detect
# then use -pr:a default -s:a compiler.cppstd=20 instead of -pr:a rhel8-gcc15
```

## Building

```bash
python3 scripts/conan-build.py install --build=missing -pr:a rhel8-gcc15
python3 scripts/conan-build.py build -pr:a rhel8-gcc15
```

`scripts/conan-build.py` exports the recipes in `conan/` and then runs the
given Conan command. If no LLVM binary for the pinned commit and profile is
available, `--build=missing` builds it once into the local Conan cache; with
the remote of the [package registry](#package-registry) and the
`rhel8-gcc15` profile, Conan downloads it instead. The
build uses the initialized `llvm` submodule as source when present
(`git submodule update --init llvm`) and otherwise clones the pinned commit.
Building LLVM takes considerably longer than csl-mlir itself.

The Conan build runs the regression and unit tests by default. To rerun them or
inspect a specific input from the repository root:

```bash
cmake --build build/Release --target check-csl
build/Release/bin/csl-opt test/Dialect/ParsedCSL/comptime_eval.mlir
build/Release/bin/csl-s2s test/Parse/CSL/parsed_csl_basic.csl --emit-after-parsing
python3 -m unittest discover -s test/Build -v
```

`python3 scripts/conan-build.py create -pr:a rhel8-gcc15` builds and installs
the package and compiles and runs the consumer in `conan/test_package`
against its exported headers, libraries and CMake targets (`CSLMLIR::`).

## Static and shared builds

Both variants are supported: `-o "&:shared=True" -o "llvm/*:shared=True"`
selects shared libraries, the default is static. Static libraries are built
position-independent, so they can be linked into a self-contained shared
library or executable. Build settings that change binary compatibility must
stay the same across the whole dependency graph.

csl-mlir is compiled with debug information (`-g`) on top of the build type's
optimization, so a Release package keeps `-O3` and still supports backtraces
and debugging; strip it from shipped binaries. `-o "&:debug_info=False"`
omits it. The LLVM package is built without debug information.

On Linux, static builds link libstdc++ and libgcc into the executables, so
that they require only glibc, and every binary gets a build ID, which matches
it to separately published debug information. Shared builds keep the shared
C++ runtime, which their shared libraries use as well.

## Licenses

csl-mlir is under the Apache License v2.0 with LLVM Exceptions
(`LICENSE.TXT`), like LLVM. The Conan build collects the license notices of
all linked dependencies into `build/Release/generators/third-party-licenses/`
and passes the directory as `CSL_THIRD_PARTY_LICENSES_DIR`. The installation
contains them in `share/csl-mlir/third-party-licenses/`, the wheel in its
`.dist-info/licenses/third-party/` and the VS Code extension in `licenses/`.
libsquashfs, which is linked into `csl-pp` and `csl-lsp-server`, is under
the LGPL 3.0 or later; its notices include the texts of the LGPL and the GPL.

## Package registry

The Conan packages of LLVM, of the other dependencies and of csl-mlir itself
are published in the package registry of the GitLab project
`cerebras/packages/csl-pp-packages`. The registry is public; a build with the
`rhel8-gcc15` profile downloads the LLVM package instead of building it:

```bash
conan remote add csl-pp https://git.esa.informatik.tu-darmstadt.de/api/v4/projects/162/packages/conan
```

It holds `llvm/<commit>` for the commit of the `llvm` gitlink,
`csl-mlir/<commit>` for every commit on main, which projects such as wespa
require, and the remaining dependencies, in static and shared variants for
every CI platform. Packages built for different platforms get different
package IDs, because each profile sets its own `os.distro` (listed in
`conan/config/settings_user.yml`) and compiler.

## Updating LLVM or a recipe

Move the `llvm` gitlink (or edit a recipe), then regenerate the lockfile:

```bash
python3 scripts/conan-build.py lock --lockfile="" -pr:a rhel8-gcc15 --lockfile-out conan.lock
```

Do not substitute an arbitrary LLVM release package.

GitHub CI never builds LLVM, because a build takes hours on its runners. A
change of the gitlink or of the LLVM recipe therefore needs new LLVM
packages in the registry before CI can pass:

1. Push the change to a branch and open a pull request. Its CI fails within
   minutes and names the missing LLVM package.
2. Run the pipeline of `cerebras/packages/csl-pp-packages` with "Run
   pipeline" on its pipelines page, with `CSL_PP_REF` set to the branch, or
   for a pull request from a fork to `refs/pull/<number>/head`. It builds LLVM for every
   platform and variant on the infrastructure of the ESA group and uploads it;
   only maintainers of that project can run it.
3. Rerun the CI of the pull request.

## Continuous integration

CI runs on GitHub Actions (`.github/workflows/ci.yml`). It builds csl-mlir
for a matrix of platforms, each in static and shared variants. A platform is
a name shared by the Conan profile `conan/config/profiles/<platform>`, a
container image and the toolchain setup in `.github/scripts/platform.sh`; the
matrix of the `build` job lists the platforms to build. Other projects can
source `.github/scripts/platform.sh` to build for the same platforms.
Currently the only platform is `rhel8-gcc15` (AlmaLinux 8 with
`gcc-toolset-15`).

Each build job runs in the platform's container
(`.github/scripts/container.sh`). It resolves the dependencies from the
package registry without credentials, builds missing dependencies other than
LLVM, and builds and tests csl-mlir as `csl-mlir/<commit>`
(`.github/scripts/build.sh`). On main, the jobs save every package they built
as an artifact, and the `upload` job uploads them to the registry. That job
runs no build code and is the only job with credentials for the registry: a
deploy token of `csl-pp-packages` with the scope `write_package_registry`, as
the secrets `CONAN_REGISTRY_USER` and `CONAN_REGISTRY_TOKEN` of the GitHub
environment `conan-registry`, which admits only main. Only builds of main
save the ccache cache, so that pull requests cannot change what later builds
restore. Complete build logs are kept as `ci-logs-*` artifacts.

`.github/workflows/website.yml` builds the website, checks its links and
publishes it with GitHub Pages from main. Dependabot keeps the pinned
actions and Python tools up to date (`.github/dependabot.yml`); actions are
pinned to commits.

## Releases

A release is a tag `v<major>.<minor>.<patch>` on a commit of main, optionally
with a pre-release suffix that is valid both as a semantic version and as a
PEP 440 version, such as `v0.2.0-rc.1` (`0.2.0rc1` on PyPI). The release
workflow (`.github/workflows/release.yml`) builds a static release on
`rhel8-gcc15`, the oldest supported platform, against the packages of the
registry (`.github/scripts/release.sh`), and produces:

- the wheel `csl_pp-<version>-py3-none-manylinux_2_28_x86_64.whl`, which runs
  on x86-64 Linux with glibc 2.28 or newer and any Python 3.8 or newer,
- the archive `csl-pp-<version>-debug-manylinux_2_28_x86_64.tar.xz` with the
  debug information of the native files in the wheel, and
- the VS Code extension `csl-lsp-linux-x64-<version>.vsix`.

The build gets no credentials. The tools are built with
`-c user.csl-mlir:release_version=<version>`, so that `--version` prints the
release; without it, a build prints `development build`. Every build prints
the commit it was built from, with `-dirty` for uncommitted changes.
`packaging/wheel/build-wheel.py` checks that the release of `csl-pp` is the
version of the wheel, and the extension takes the release as its version.

The jobs after the build publish the three files:

- `github-release` attaches them to a GitHub release of the tag, marked as a
  pre-release for a pre-release version, with build provenance attestations.
  `gh attestation verify <file> --repo esa-tu-darmstadt/csl-pp` checks that a
  file was built by this workflow.
- `pypi` uploads the wheel to PyPI with trusted publishing, which also
  uploads its PEP 740 attestations.
- `vscode-marketplace` publishes the extension to the Visual Studio
  Marketplace with trusted publishing. The Marketplace accepts only versions
  `major.minor.patch`, so pre-releases skip this job.

Trusted publishing needs no stored token: PyPI and the Marketplace accept
the OpenID Connect token of the job, issued for this repository, the workflow
`release.yml` and the job's environment (`pypi` or `vscode-marketplace`). Both
environments require the approval of a maintainer and admit only tags `v*`.

`packaging/wheel/build-wheel.py` builds the wheel from an installation prefix:

```bash
cmake --install build/Release --prefix stage
python3 packaging/wheel/build-wheel.py --prefix stage --version 0.2.0 --out-dir dist
```

The wheel contains the package `csl_pp` with `bin/csl-pp`,
`bin/csl-lsp-server`, `lib/libinject_mllvm.so` and `share/csl-mlir/libs`, in
the layout of the installation prefix, so that the tools find
`libinject_mllvm.so` and the csl++ libraries as in an installation. The commands that pip installs
are the launchers in `packaging/wheel/csl_pp/_launcher.py`, which replace
the Python process with the executable. The script moves the debug
information into the separate archive; the shipped files keep their symbol
tables and a `.gnu_debuglink` to their debug file. It rejects any native file
that requires a shared library other than glibc's, libstdc++ or libgcc_s, a
glibc symbol version newer than the platform tag allows, or that has no build
ID. `packaging/wheel/README.md` is the description of the package on PyPI.

To resolve a backtrace of a released `csl-pp`, unpack the debug archive of
its version and pass the debug file to `llvm-symbolizer` or `addr2line`, or
put it next to the executable for `gdb`. `readelf -n` shows the build ID of
both files, which must match.

To make a release:

1. Tag a commit on main and push the tag:

   ```bash
   git tag -a v1.0.0 -m "csl++ 1.0.0"
   git push origin v1.0.0
   ```

2. The release workflow builds the files and creates the GitHub release.
3. Approve the `pypi` and `vscode-marketplace` jobs.

A version cannot be uploaded to PyPI or the Marketplace twice; a broken
release is followed by a new version.

## Repository settings

The workflows rely on these settings of the GitHub repository:

- Environments: `conan-registry` (secrets `CONAN_REGISTRY_USER` and
  `CONAN_REGISTRY_TOKEN`, deployment branch main), `pypi` and
  `vscode-marketplace` (required reviewers, deployment tags `v*`) and
  `github-pages` (created by GitHub Pages).
- A tag ruleset for `v*` that restricts creating, updating and deleting
  release tags to maintainers.
- Immutable releases, so that the files of a published release cannot be
  replaced.
- Pages with "GitHub Actions" as the source.
- Private vulnerability reporting, which `SECURITY.md` refers to.

On PyPI, the project `csl-pp` of the organization `esa-tu-darmstadt` has a
trusted publisher for this repository, the workflow `release.yml` and the
environment `pypi`. On the Marketplace, the publisher `esa-tu-darmstadt` has
a trusted publishing policy for the same repository and workflow.
