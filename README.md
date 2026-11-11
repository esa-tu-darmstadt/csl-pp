# csl++

[![CI](https://github.com/esa-tu-darmstadt/csl-pp/actions/workflows/ci.yml/badge.svg)](https://github.com/esa-tu-darmstadt/csl-pp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/csl-pp)](https://pypi.org/project/csl-pp/)
[![Documentation](https://img.shields.io/badge/docs-esa--tu--darmstadt.github.io%2Fcsl--pp-blue)](https://esa-tu-darmstadt.github.io/csl-pp/)
[![License](https://img.shields.io/badge/license-Apache--2.0%20WITH%20LLVM--exception-blue)](LICENSE.TXT)

csl++ is a place to develop new language features for the Cerebras Software Language (CSL) and to test them at wafer scale. It extends CSL, so every CSL program is also a csl++ program. It currently adds async functions and tasks, and automatic allocation of colors, task IDs and queues.

The csl++ compiler `csl-pp` is a drop-in replacement for `cslc` and takes the same command-line arguments. It translates a program to vanilla CSL and then compiles the result with `cslc`, the compiler of the Cerebras SDK.

csl++ is developed by the Embedded Systems and Applications Group at the Technical University of Darmstadt. Contributions are welcome.

```csl
async fn recv_squared(comptime in: fabin_dsd) f32 {
  var x: f32 = 0.0;
  @fmovs(&x, in, .{ .async = true, .activate = @current_task() });
  suspend;  // continues here when the value has arrived
  x *= x;
  return x;
}

async task main() void {
  // Receive from both neighbors at the same time.
  const r = @await_parallel(recv_squared(west_dsd), recv_squared(east_dsd));
  sum = r[0] + r[1];
}
```

## Documentation

The documentation at https://esa-tu-darmstadt.github.io/csl-pp/ contains installation instructions, a quickstart, a tutorial, and the reference of the language extensions. Its source is in `website/`.

## Installation

```bash
pipx install csl-pp
```

The package contains `csl-pp`, `csl-lsp-server` and the libraries for x86-64 Linux with glibc 2.28 or newer. `csl-pp` runs `cslc` of the Cerebras SDK, which is installed separately; put the SDK directory on your `PATH`.

## Building

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r conan/requirements.txt
conan config install conan/config
conan profile detect
python3 scripts/conan-build.py install --build=missing -pr:a default -s:a compiler.cppstd=20
python3 scripts/conan-build.py build -pr:a default -s:a compiler.cppstd=20
```

The tools are in `build/Release/bin`. `docs/Building.md` describes the build in detail.

## Repository

| Directory | Content |
| --- | --- |
| `lib/`, `include/` | The compiler. |
| `csl-libs/` | CSL libraries written in csl++: `allreduce`, `stencil_3d_7pts`, `collectives_2d`. |
| `tools/` | `csl-pp`, the language server `csl-lsp-server`, developer tools. |
| `editors/vscode/` | The VS Code extension. |
| `test/`, `unittests/` | Regression, integration and unit tests. |
| `website/` | The documentation website and its example programs. |
| `conan/`, `conanfile.py` | Conan recipes for csl++ and its dependencies, profiles, and the test package. |
| `packaging/` | The Python package `csl-pp` published on PyPI. |
| `docs/` | Documentation of the build and the release process. |

## License

csl++ is under the Apache License v2.0 with LLVM Exceptions; see `LICENSE.TXT`. `NOTICE` lists the parts derived from the examples of the Cerebras SDK.
