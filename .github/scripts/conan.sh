# shellcheck shell=bash
# Conan setup shared by the CI scripts, sourced after platform_setup.

# Installs the pinned Conan tools into /opt/conan-venv and the settings and
# profiles of conan/config. The checkout belongs to the runner's user, not to
# the container's root, so Git is told to trust it.
conan_setup() {
  git config --global --add safe.directory '*'
  "$CI_PYTHON" -m venv /opt/conan-venv
  /opt/conan-venv/bin/pip install -q --no-cache-dir -r conan/requirements.txt
  export PATH="/opt/conan-venv/bin:$PATH"
  conan config install conan/config > /dev/null
}

# Arguments of every Conan command of a build: the platform's profile, the
# static or shared variant (CSL_SHARED, True or False) for csl-mlir and LLVM,
# CI_JOBS parallel jobs (default: all cores), ccache, and the lockfile.
# shellcheck disable=SC2034 # CONAN_ARGS is used by the sourcing scripts.
conan_args() {
  CONAN_ARGS=(
    -pr:a "$PLATFORM"
    -o "llvm/*:shared=$CSL_SHARED" -o "&:shared=$CSL_SHARED"
    -c "tools.build:jobs=${CI_JOBS:-$(nproc)}"
    -c 'tools.cmake.cmaketoolchain:extra_variables={"CMAKE_C_COMPILER_LAUNCHER":"ccache","CMAKE_CXX_COMPILER_LAUNCHER":"ccache"}'
    --lockfile=conan.lock
  )
}
