#!/usr/bin/env bash
# Runs a command in the container of $PLATFORM, with the checkout at /src.
#
# CI_WORK is a host directory with room for the build: it holds the Conan home
# (conan/) and the ccache directory (ccache/), which the container sees as
# /conan and /ccache. The variables PLATFORM, CSL_SHARED and GITHUB_SHA, and
# those named in CI_PASS_ENV, are passed into the container.
#
# Example: .github/scripts/container.sh bash .github/scripts/build.sh
set -euo pipefail

: "${PLATFORM:?}" "${CI_WORK:?}"
# shellcheck source=platform.sh
source "$(dirname "$0")/platform.sh"
image=$(platform_image "$PLATFORM")
mkdir -p "$CI_WORK/conan" "$CI_WORK/ccache"

environment=(
  -e CONAN_HOME=/conan -e CCACHE_DIR=/ccache -e CCACHE_BASEDIR=/src
  -e CCACHE_COMPRESS=true -e CCACHE_MAXSIZE=5G -e CCACHE_COMPILERCHECK=content
)
for name in PLATFORM CSL_SHARED GITHUB_SHA ${CI_PASS_ENV:-}; do
  environment+=(-e "$name")
done

exec docker run --rm \
  -v "$PWD:/src" -v "$CI_WORK/conan:/conan" -v "$CI_WORK/ccache:/ccache" \
  -w /src "${environment[@]}" "$image" "$@"
