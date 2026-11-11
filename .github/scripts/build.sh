#!/usr/bin/env bash
# Builds and tests csl-mlir in the platform container (see container.sh).
#
# Dependencies are resolved from the package registry, which the remote csl-pp
# (CONAN_REMOTE_URL) reads without credentials. LLVM is never built here: its
# packages are built on the infrastructure of the ESA group and uploaded to the
# registry (docs/Building.md, "Updating LLVM"). Other dependencies that are
# missing there are built. Then csl-mlir is built, tested and packaged as
# csl-mlir/$GITHUB_SHA.
#
# With SAVE_PACKAGES=1, the recipes of the graph and every binary built here
# are saved to ci-out/conan-packages.tgz. The upload job, the only one with
# credentials for the registry, uploads them; no build code runs with them.
set -euo pipefail

: "${CONAN_REMOTE_URL:?}" "${GITHUB_SHA:?}"
source .github/scripts/platform.sh
source .github/scripts/conan.sh
platform_setup
conan_setup
conan_args
conan remote add --index 0 --force csl-pp "$CONAN_REMOTE_URL"
ccache --zero-stats > /dev/null
mkdir -p ci-out ci-logs

"$CI_PYTHON" -m unittest discover -s test/Build -v

# The timeouts leave time within the job's limit to save logs and caches.
if ! "$CI_PYTHON" scripts/ci-run.py --log=ci-logs/dependencies.log --timeout=7200 -- \
  "$CI_PYTHON" scripts/conan-build.py install --build=missing --build='~llvm/*' \
  "${CONAN_ARGS[@]}" --format=json --out-file=ci-out/dependencies.json; then
  if grep -Eq "Missing prebuilt package for '(.*', ')?llvm/" ci-logs/dependencies.log; then
    echo "::error::The package registry has no LLVM package for the llvm gitlink" \
      "$(git rev-parse HEAD:llvm) on $PLATFORM (shared=$CSL_SHARED). Build it with" \
      "the pipeline of cerebras/packages/csl-pp-packages, then rerun this job;" \
      "docs/Building.md describes the steps."
  fi
  exit 1
fi
"$CI_PYTHON" scripts/ci-run.py --log=ci-logs/csl.log --timeout=10800 -- \
  "$CI_PYTHON" scripts/conan-build.py create --version="$GITHUB_SHA" \
  --build="missing:csl-mlir/*" "${CONAN_ARGS[@]}" \
  --format=json --out-file=ci-out/create.json
ccache --show-stats

if [ "${SAVE_PACKAGES:-0}" = 1 ]; then
  for graph in dependencies create; do
    conan list --graph="ci-out/$graph.json" --graph-recipes='*' --graph-binaries=build \
      --format=json --out-file="ci-out/built-$graph.json" > /dev/null
  done
  conan pkglist merge -l ci-out/built-dependencies.json -l ci-out/built-create.json \
    --format=json --out-file=ci-out/packages.json > /dev/null
  conan cache save --no-source --list=ci-out/packages.json --file=ci-out/conan-packages.tgz
fi
