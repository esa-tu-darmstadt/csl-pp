# shellcheck shell=bash
# Toolchain setup of a CI platform, sourced inside the platform's container.
#
# A platform is a name shared by a Conan profile
# (conan/config/profiles/<platform>), a container image (platform_image below)
# and the setup in this file. After `source .github/scripts/platform.sh` with
# PLATFORM set, gcc, g++, git, ccache, make, perl and $CI_PYTHON
# (Python >= 3.10) are on PATH. Other projects can source this file to build
# for the same platforms.

# Prints the container image of the platform given as the first argument.
platform_image() {
  case "$1" in
    rhel8-gcc15) echo "almalinux:8" ;;
    *) echo "Unknown CI platform '$1'" >&2; return 1 ;;
  esac
}

# Installs the toolchain of $PLATFORM; must run inside platform_image.
platform_setup() {
  case "$PLATFORM" in
    rhel8-gcc15)
      # AlmaLinux 8 (glibc 2.28) with gcc-toolset-15. The old glibc lets the
      # binaries run on every distribution with glibc 2.28 or newer.
      grep -q '^PLATFORM_ID="platform:el8"$' /etc/os-release \
        || { echo "rhel8-gcc15 requires an AlmaLinux 8 container" >&2; return 1; }
      dnf install -y -q epel-release > /dev/null
      dnf install -y -q --setopt=install_weak_deps=False ccache \
        gcc-toolset-15-gcc-c++ gcc-toolset-15-binutils git python3.12 make perl \
        patch pkgconf-pkg-config xz bzip2 tar findutils diffutils which > /dev/null
      # shellcheck disable=SC1091
      source /opt/rh/gcc-toolset-15/enable
      CI_PYTHON=python3.12
      ;;
    *)
      echo "Unknown CI platform '$PLATFORM'" >&2
      return 1
      ;;
  esac
  export CI_PYTHON
}
