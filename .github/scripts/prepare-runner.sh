#!/usr/bin/env bash
# Selects the directory for the build on a GitHub-hosted runner.
#
# CI_WORK (see container.sh) holds the Conan home, with the LLVM package of
# several gigabytes, and the ccache directory. It goes to the file system with
# the most free space, usually the runner's second disk at /mnt, and is
# appended to $GITHUB_ENV.
set -euo pipefail

free_kb() { df --output=avail -k "$1" | tail -n 1; }
work="$RUNNER_TEMP/ci-work"
if [ -d /mnt ] && [ "$(free_kb /mnt)" -gt "$(free_kb "$RUNNER_TEMP")" ]; then
  work=/mnt/ci-work
fi
sudo mkdir -p "$work"
sudo chown "$(id -u):$(id -g)" "$work"
echo "CI_WORK=$work" >> "$GITHUB_ENV"
df -h / "$work"
