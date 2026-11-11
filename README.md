# csl++

csl++ is an open-source frontend for the Cerebras Software Language (CSL),
developed by the Embedded Systems and Applications Group at the Technical
University of Darmstadt. It adds async functions and tasks and automatic
allocation of colors, task IDs and queues to CSL. Its compiler, `csl-pp`,
evaluates the program at compile time, transforms it in its own IR, and
lowers it to vanilla CSL, which `cslc` of the Cerebras SDK compiles. `csl-pp`
accepts the command-line arguments of `cslc`. Contributions are welcome.

```csl
async task sum_task() void {
  if (layout_mod.get_x_coord() > 0) {
    // Add the partial sum arriving from the west.
    @fadds(x_dsd, x_dsd, in_dsd, .{ .async = true, .activate = @current_task() });
    suspend;
  }
  if (layout_mod.get_x_coord() < @as(u16, width - 1)) {
    // Send the partial sum east.
    @fmovs(out_dsd, x_dsd, .{ .async = true, .activate = @current_task() });
    suspend;
  }
  sys_mod.unblock_cmd_stream();
}
```

## Documentation

The documentation at https://esa-tu-darmstadt.github.io/csl-pp/ contains
installation instructions, a quickstart, a tutorial, and the reference of the
language extensions. Its source is in `website/`.

## Installation

```bash
pipx install csl-pp
```

The package contains `csl-pp`, `csl-lsp-server` and the libraries for x86-64
Linux with glibc 2.28 or newer. `csl-pp` runs `cslc` of the Cerebras SDK,
which is installed separately; put the SDK directory on your `PATH`.

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

The tools are in `build/Release/bin`. `docs/Building.md` describes the build
in detail.

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

csl++ is under the Apache License v2.0 with LLVM Exceptions; see
`LICENSE.TXT`. `NOTICE` lists the parts derived from the examples of the
Cerebras SDK.
