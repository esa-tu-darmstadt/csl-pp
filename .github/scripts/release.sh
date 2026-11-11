#!/usr/bin/env bash
# Builds a release of csl++ in the platform container (see container.sh): the
# wheel csl-pp with the archive of its debug information and the VS Code
# extension, in dist/.
#
# Releases are static builds. LLVM and the other dependencies come from the
# package registry (CONAN_REMOTE_URL), as in CI; a missing LLVM package fails
# the build. CSL_VERSION is the release, e.g. 1.2.0 or 1.2.0-rc.1, which the
# tools print with --version and the extension takes as its version.
# DOCUMENTATION_URL is the documentation website named in the wheel.
set -euo pipefail

: "${CONAN_REMOTE_URL:?}" "${CSL_VERSION:?}" "${DOCUMENTATION_URL:?}" "${GITHUB_SHA:?}"
source .github/scripts/platform.sh
source .github/scripts/conan.sh
platform_setup
conan_setup
CSL_SHARED=False conan_args
conan remote add --index 0 --force csl-pp "$CONAN_REMOTE_URL"
dnf module enable -y -q nodejs:20 > /dev/null
dnf install -y -q nodejs npm > /dev/null
mkdir -p ci-logs dist

"$CI_PYTHON" scripts/ci-run.py --log=ci-logs/release.log --timeout=14400 -- \
  "$CI_PYTHON" scripts/conan-build.py build --version="$GITHUB_SHA" \
  --build=missing --build='~llvm/*' "${CONAN_ARGS[@]}" \
  -c "user.csl-mlir:release_version=$CSL_VERSION"
cmake --build build/Release --target csl-lsp-vscode
cmake --install build/Release --prefix stage > /dev/null

"$CI_PYTHON" -m venv /opt/wheel-venv
/opt/wheel-venv/bin/pip install -q --no-cache-dir -r packaging/requirements.txt
/opt/wheel-venv/bin/python packaging/wheel/build-wheel.py --prefix stage \
  --version "$CSL_VERSION" --out-dir dist --documentation-url "$DOCUMENTATION_URL"
/opt/wheel-venv/bin/twine check --strict dist/*.whl
/opt/wheel-venv/bin/auditwheel show dist/*.whl

# The wheel must work on its own, without the build tree.
"$CI_PYTHON" -m venv /opt/smoke-venv
/opt/smoke-venv/bin/pip install -q --no-cache-dir dist/*.whl
/opt/smoke-venv/bin/csl-pp --version
/opt/smoke-venv/bin/csl-lsp-server --version
cp build/Release/vscode/*.vsix dist/
