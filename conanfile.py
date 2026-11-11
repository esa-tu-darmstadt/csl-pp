"""Build and package csl-mlir independently of LLVM's source build."""

import json
import os
import re
import shutil

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import build_jobs, check_min_cppstd
from conan.tools.cmake import CMake, CMakeConfigDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy, save
from conan.tools.scm import Git

from conan_support import llvm_pin

required_conan_version = ">=2.32 <3"


class CSLMLIRConan(ConanFile):
    """Export typed compiler libraries, generated headers and command-line tools.

    The package version is the csl-mlir commit, and LLVM is required at the
    commit of the tracked ``llvm`` gitlink, so both pins live in Git only.
    The LLVM and squashfs recipes in ``conan/`` are exported by
    ``scripts/conan-build.py``.
    """

    name = "csl-mlir"
    extension_properties = {"compatibility_cppstd": False}
    package_type = "library"
    package_id_non_embed_mode = "full_mode"
    # The consumer that `conan create` builds against the package.
    test_package_folder = "conan/test_package"
    settings = "os", "arch", "compiler", "build_type"
    options = {"shared": [True, False], "fPIC": [True, False],
               "with_tests": [True, False], "debug_info": [True, False]}
    default_options = {"shared": False, "fPIC": True, "with_tests": True,
                       "debug_info": True}

    def set_version(self):
        """Use the checked-out commit unless the caller names one explicitly.

        A working tree with uncommitted changes gets a ``-dirty`` suffix so a
        local package can never be mistaken for the published commit.
        """
        if self.version:
            if not re.fullmatch(r"[0-9a-f]{40}(-dirty)?", str(self.version)):
                raise ConanInvalidConfiguration("csl-mlir versions are full commits")
            return
        git = Git(self, self.recipe_folder)
        # Changes inside submodule working trees do not alter csl-mlir's
        # sources; a moved gitlink still counts. Skipping them also avoids
        # scanning the large llvm checkout.
        dirty = git.run("status --porcelain --untracked-files=no --ignore-submodules=dirty")
        self.version = git.get_commit() + ("-dirty" if dirty.strip() else "")

    def export(self):
        """Capture the gitlink for recipes used outside a Git checkout."""
        shutil.copy2(os.path.join(self.recipe_folder, "conan_support.py"),
                     self.export_folder)
        save(self, os.path.join(self.export_folder, "llvm-pin.json"),
             json.dumps(llvm_pin(self.recipe_folder), sort_keys=True))

    def export_sources(self):
        """Export compiler sources without traversing the LLVM submodule."""
        for file in ("CMakeLists.txt", "LICENSE.TXT"):
            shutil.copy2(os.path.join(self.recipe_folder, file),
                         self.export_sources_folder)
        for directory in ("cmake", "include", "lib", "csl-libs", "tools", "test", "unittests"):
            copy(self, "*", os.path.join(self.recipe_folder, directory),
                 os.path.join(self.export_sources_folder, directory),
                 excludes=["__pycache__", "*.pyc"])
        copy(self, "*", os.path.join(self.recipe_folder, "scripts/cerebras-sdk-oci"),
             os.path.join(self.export_sources_folder, "scripts/cerebras-sdk-oci"),
             excludes=["__pycache__", "*.pyc"])

    def requirements(self):
        """Use the LLVM package of the gitlink commit."""
        self.requires("llvm/" + llvm_pin(self.recipe_folder)["commit"],
                      transitive_headers=True, transitive_libs=True)
        self.requires("squashfs-tools-ng/1.3.2")

    def validate(self):
        """Require C++20, the language level of the exported compiler headers."""
        check_min_cppstd(self, "20")

    def layout(self):
        """Keep all generated files under the selected build directory."""
        cmake_layout(self)

    def generate(self):
        """Supply native package locations and the gitlink verification contract."""
        CMakeConfigDeps(self).generate()
        llvm = self.dependencies["llvm"].package_folder
        # The license notices of every library linked into the tools, which
        # the installation and the VS Code extension ship next to them.
        licenses = os.path.join(self.generators_folder, "third-party-licenses")
        shutil.rmtree(licenses, ignore_errors=True)
        for require, dependency in self.dependencies.items():
            if require.build or require.test:
                continue
            copy(self, "*", os.path.join(dependency.package_folder, "licenses"),
                 os.path.join(licenses, dependency.ref.name))
        tc = CMakeToolchain(self)
        tc.variables.update({
            "CMAKE_INSTALL_LIBDIR": "lib",
            "BUILD_SHARED_LIBS": bool(self.options.shared),
            "CSL_BUILD_TESTS": bool(self.options.with_tests),
            "CSL_LLVM_REVISION": llvm_pin(self.recipe_folder)["commit"],
            "MLIR_DIR": os.path.join(llvm, "lib/cmake/mlir"),
            "LLVM_DIR": os.path.join(llvm, "lib/cmake/llvm"),
            "LLVM_EXTERNAL_LIT": os.path.join(llvm, "share/llvm-sdk/lit/lit.py"),
            "LLVM_LIT_ARGS": "-sv -j" + str(build_jobs(self)),
            "CSL_THIRD_PARTY_LICENSES_DIR": licenses,
        })
        # --version prints them. Only release builds name a release, with
        # -c user.csl-mlir:release_version=<version>. Unlike tc.variables,
        # cache variables are passed with -D on every configure, so a value
        # from an earlier configuration does not persist in the CMake cache.
        tc.cache_variables["CSL_GIT_COMMIT"] = str(self.version)
        tc.cache_variables["CSL_RELEASE_VERSION"] = self.conf.get(
            "user.csl-mlir:release_version", default="", check_type=str)
        if self.options.debug_info:
            # Debug information on top of the selected optimization level, so
            # released code is the code that was debugged; strip it to ship.
            tc.extra_cflags.append("-g")
            tc.extra_cxxflags.append("-g")
        tc.generate()

    def build(self):
        """Build compiler tools and run the existing regression/unit suite."""
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        if self.options.with_tests and not self.conf.get("tools.build:skip_test", default=False):
            cmake.build(target="check-csl")

    def package(self):
        """Install the compiler package through its CMake export rules."""
        CMake(self).install()

    def package_info(self):
        """Preserve native exported targets and expose the installed tools."""
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.set_property("cmake_file_name", "CSLMLIR")
        self.cpp_info.builddirs = ["lib/cmake/CSLMLIR"]
        self.buildenv_info.prepend_path("PATH", os.path.join(self.package_folder, "bin"))
