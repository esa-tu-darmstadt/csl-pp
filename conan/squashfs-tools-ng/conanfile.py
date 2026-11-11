"""Build csl-mlir's SIF reader dependency through Conan."""

import os

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.files import copy, get
from conan.tools.gnu import Autotools, AutotoolsToolchain, PkgConfigDeps

required_conan_version = ">=2.32 <3"


class SquashfsConan(ConanFile):
    """Package a position-independent libsquashfs with all existing codecs."""

    name = "squashfs-tools-ng"
    version = "1.3.2"
    package_type = "static-library"
    license = "LGPL-3.0-or-later"
    settings = "os", "arch", "compiler", "build_type"
    options = {"fPIC": [True, False]}
    default_options = {"fPIC": True}

    def configure(self):
        """Discard C++ ABI settings for this C library."""
        self.settings.rm_safe("compiler.cppstd")
        self.settings.rm_safe("compiler.libcxx")

    def validate(self):
        """Reject platforms unsupported by the existing autotools dependency."""
        if self.settings.os not in ("Linux", "Macos", "FreeBSD"):
            raise ConanInvalidConfiguration("libsquashfs requires a POSIX build host")

    def requirements(self):
        """Keep gzip, xz, zstd and lz4 SIF decoding explicitly available."""
        self.requires("zlib/1.3.1")
        self.requires("xz_utils/5.8.1")
        self.requires("zstd/1.5.7")
        self.requires("lz4/1.10.0")

    def source(self):
        """Download and verify the original dependency release."""
        get(self,
            "https://infraroot.at/pub/squashfs/squashfs-tools-ng-1.3.2.tar.xz",
            sha256="0d907ac3e735c351e47c867fb51d94bffa3b05fb95bec01f31e848b7c44215a9",
            strip_root=True)

    def generate(self):
        """Configure the static library without command-line tools or SELinux."""
        PkgConfigDeps(self).generate()
        tc = AutotoolsToolchain(self)
        tc.configure_args += ["--without-tools", "--without-selinux",
                              "--disable-shared", "--enable-static"]
        tc.generate()

    def build(self):
        """Build with Conan's selected compiler and bounded parallelism."""
        autotools = Autotools(self)
        autotools.configure()
        autotools.make()

    def package(self):
        """Install library, public headers and original license notices."""
        Autotools(self).install()
        licenses = os.path.join(self.package_folder, "licenses")
        copy(self, "COPYING*", self.source_folder, licenses)
        # libsquashfs is LGPL-3.0-or-later, which adds to the terms of the
        # GPLv3, and contains BSD- and MIT-licensed hash code.
        for notice in ("LGPLv3.txt", "GPLv3.txt", "xxhash.txt", "hash_table.txt"):
            copy(self, notice, os.path.join(self.source_folder, "licenses"), licenses)

    def package_info(self):
        """Expose the static library and its codec dependencies to CMake."""
        self.cpp_info.libs = ["squashfs"]
        self.cpp_info.set_property("cmake_file_name", "Squashfs")
        self.cpp_info.set_property("cmake_target_name", "Squashfs::Squashfs")
