"""Check that a consumer needs only the installed compiler package."""
import os
import shlex
from conan import ConanFile
from conan.tools.build import can_run
from conan.tools.cmake import CMake, CMakeConfigDeps, CMakeToolchain, cmake_layout

required_conan_version = ">=2.32 <3"


class CSLMLIRConsumer(ConanFile):
    """Compile and execute a generated-header/dialect registration smoke test."""
    settings = "os", "arch", "compiler", "build_type"
    test_type = "explicit"

    def requirements(self):
        """Test the exact package just created, including its transitive SDK."""
        self.requires(self.tested_reference_str)

    def layout(self):
        """Place generated consumer files outside the fixture sources."""
        cmake_layout(self)

    def generate(self):
        """Locate native CMake exports through the resolved package graph."""
        CMakeConfigDeps(self).generate()
        tc = CMakeToolchain(self)
        tc.variables["CSL_CONSUMER_SHARED"] = bool(self.dependencies["csl-mlir"].options.shared)
        tc.generate()

    def build(self):
        """Compile against installed public and generated headers only."""
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def test(self):
        """Execute the typed dialect smoke test when host binaries can run."""
        if can_run(self):
            self.run(os.path.join(self.cpp.build.bindirs[0], "csl-package-consumer"),
                     env="conanrun")
            package = self.dependencies["csl-mlir"].package_folder
            for tool in ("csl-opt", "csl-s2s", "csl-pp", "csl-lsp-server",
                         "csl-lex-roundtrip"):
                executable = shlex.quote(os.path.join(package, "bin", tool))
                self.run(executable + " --help", env="conanrun")
