#!/usr/bin/env bash
# Makes room on a GitHub-hosted runner for builds of LLVM and csl-mlir.
#
# Removes preinstalled toolchains the builds do not use and selects the file
# system with the most free space for CI_WORK (see container.sh), which is
# appended to $GITHUB_ENV.
set -euo pipefail

sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /usr/local/.ghcup \
  /opt/hostedtoolcache/CodeQL /usr/share/swift
sudo docker image prune --all --force > /dev/null

free_kb() { df --output=avail -k "$1" | tail -n 1; }
work="$RUNNER_TEMP/ci-work"
if [ -d /mnt ] && [ "$(free_kb /mnt)" -gt "$(free_kb "$RUNNER_TEMP")" ]; then
  work=/mnt/ci-work
fi
sudo mkdir -p "$work"
sudo chown "$(id -u):$(id -g)" "$work"
echo "CI_WORK=$work" >> "$GITHUB_ENV"
df -h / "$work"
