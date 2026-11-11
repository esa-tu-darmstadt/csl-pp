"""Package the exact LLVM/MLIR/LLD source selected by csl-mlir's gitlink."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.cmake import CMake, CMakeConfigDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy, save
from conan.tools.scm import Git

required_conan_version = ">=2.32 <3"

_helper = Path(__file__).with_name("conan_support.py")
if not _helper.is_file():
    _helper = Path(__file__).resolve().parents[2] / "conan_support.py"
_spec = importlib.util.spec_from_file_location("csl_llvm_pin", _helper)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class LLVMConan(ConanFile):
    """Provide native compiler libraries, generated headers and test tools."""

    name = "llvm"
    extension_properties = {"compatibility_cppstd": False}
    package_type = "library"
    package_id_non_embed_mode = "full_mode"
    license = "Apache-2.0 WITH LLVM-exception"
    settings = "os", "arch", "compiler", "build_type"
    no_copy_source = True
    options = {"shared": [True, False], "fPIC": [True, False],
               "assertions": [True, False], "rtti": [True, False],
               "exceptions": [True, False], "targets": ["ANY"]}
    default_options = {"shared": False, "fPIC": True, "assertions": True,
                       "rtti": False, "exceptions": False, "targets": "Native"}

    def set_version(self):
        """Use the full gitlink commit as the package version."""
        pin = _module.llvm_pin(self.recipe_folder)
        if self.version and str(self.version) != pin["commit"]:
            raise ConanInvalidConfiguration("LLVM version must equal its gitlink")
        self.version = pin["commit"]

    def export(self):
        """Capture the pin and helper without exporting the LLVM source tree."""
        save(self, os.path.join(self.export_folder, "llvm-pin.json"),
             json.dumps(_module.llvm_pin(self.recipe_folder), sort_keys=True))
        shutil.copy2(_helper, self.export_folder)

    def requirements(self):
        """Declare the compression library referenced by installed LLVM targets."""
        self.requires("zlib/1.3.1")

    def layout(self):
        """Keep source, build products and package contents separate."""
        cmake_layout(self, src_folder="src")

    def validate(self):
        """Require the language level used by the qualified compiler build."""
        from conan.tools.build import check_min_cppstd
        check_min_cppstd(self, "20")

    def source(self):
        """Materialize only the pinned committed source, locally or from Git."""
        pin = _module.llvm_pin(self.recipe_folder)
        checkout = self.conf.get("user.llvm:checkout", default=None)
        if checkout:
            head = subprocess.check_output(
                ["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(
                ["git", "-C", checkout, "status", "--porcelain",
                 "--untracked-files=no"], text=True).strip()
            if head != pin["commit"] or dirty:
                raise ConanInvalidConfiguration(
                    "The initialized LLVM submodule must match its gitlink and be unmodified")
            archive = subprocess.Popen(
                ["git", "-C", checkout, "archive", pin["commit"]],
                stdout=subprocess.PIPE)
            try:
                subprocess.run(["tar", "-x", "-C", self.source_folder],
                               stdin=archive.stdout, check=True)
            finally:
                archive.stdout.close()
                status = archive.wait()
            if status:
                raise ConanInvalidConfiguration("Could not export pinned LLVM sources")
        else:
            git = Git(self, self.source_folder)
            git.clone(pin["url"], target=".", args=["--filter=blob:none", "--no-checkout"])
            git.checkout(pin["commit"])

    def generate(self):
        """Configure one ABI variant with upstream package exports and test support."""
        # LLVM's llvm-config support reads imported library locations directly.
        # CMakeConfigDeps supplies real imported targets, including ZLIB::ZLIB.
        CMakeConfigDeps(self).generate()
        tc = CMakeToolchain(self)
        tc.variables.update({
            "CMAKE_INSTALL_LIBDIR": "lib",
            "BUILD_SHARED_LIBS": bool(self.options.shared),
            "LLVM_ENABLE_PROJECTS": "mlir;lld",
            "LLVM_TARGETS_TO_BUILD": str(self.options.targets),
            "LLVM_ENABLE_ASSERTIONS": bool(self.options.assertions),
            "LLVM_ENABLE_RTTI": bool(self.options.rtti),
            "LLVM_ENABLE_EH": bool(self.options.exceptions),
            "LLVM_APPEND_VC_REV": False,
            "LLVM_PARALLEL_LINK_JOBS": 2,
            "LLVM_INCLUDE_TESTS": True, "LLVM_BUILD_TESTS": False,
            "LLVM_INSTALL_GTEST": True, "LLVM_INSTALL_UTILS": True,
            "LLVM_INCLUDE_BENCHMARKS": False, "LLVM_BUILD_BENCHMARKS": False,
            "LLVM_BUILD_EXAMPLES": False, "LLVM_ENABLE_BINDINGS": False,
            "LLVM_ENABLE_ZLIB": "FORCE_ON", "LLVM_ENABLE_ZSTD": "OFF",
            "LLVM_ENABLE_LIBXML2": False, "LLVM_ENABLE_LIBEDIT": False,
            "LLVM_ENABLE_CURL": False,
            "MLIR_INCLUDE_TESTS": False, "MLIR_ENABLE_BINDINGS_PYTHON": False,
            "CMAKE_DISABLE_PRECOMPILE_HEADERS": True,
        })
        tc.generate()

    def build(self):
        """Build the installed development SDK, not LLVM's own test executables."""
        cmake = CMake(self)
        cmake.configure(build_script_folder="llvm")
        cmake.build()

    def package(self):
        """Install upstream exports, matching lit and exact source provenance."""
        CMake(self).install()
        copy(self, "*", os.path.join(self.source_folder, "llvm/utils/lit"),
             os.path.join(self.package_folder, "share/llvm-sdk/lit"),
             excludes=["tests/*", "*.pyc", "__pycache__/*"])
        pin = _module.llvm_pin(self.recipe_folder)
        save(self, os.path.join(self.package_folder, "share/llvm-sdk/provenance.json"),
             json.dumps({"schema": 1, "revision": pin["commit"]}, sort_keys=True))
        for project in ("llvm", "mlir", "lld"):
            destination = os.path.join(self.package_folder, "licenses", project)
            os.makedirs(destination, exist_ok=True)
            shutil.copy2(os.path.join(self.source_folder, project, "LICENSE.TXT"),
                         destination)

    def package_info(self):
        """Use upstream CMake configs and expose executable tools to native builds."""
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs = ["lib/cmake/llvm", "lib/cmake/mlir", "lib/cmake/lld"]
        self.buildenv_info.prepend_path("PATH", os.path.join(self.package_folder, "bin"))
