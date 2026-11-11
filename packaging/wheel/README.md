# csl++

csl++ is a place to develop new language features for the Cerebras Software Language (CSL) and to test them at wafer scale. It extends CSL, so every CSL program is also a csl++ program.

The csl++ compiler `csl-pp` is a drop-in replacement for `cslc` and takes the same command-line arguments. It translates a program to vanilla CSL and then compiles the result with `cslc`, the compiler of the Cerebras SDK.

csl++ currently adds async functions and tasks, and automatic allocation of colors, task IDs and queues:

```
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

The [documentation](https://esa-tu-darmstadt.github.io/csl-pp/) contains a quickstart, a tutorial and the reference of the language extensions.

## Installation

This package installs command-line tools, not a Python library: `csl-pp`, the language server `csl-lsp-server` and the csl++ libraries. It is built for x86-64 Linux with glibc 2.28 or newer.

```bash
pipx install csl-pp
```

`uv tool install csl-pp` and `pip install csl-pp` in a virtual environment work as well.

`csl-pp` runs `cslc` of the [Cerebras SDK](https://sdk.cerebras.ai), which is installed separately; csl++ is tested with SDK 2.10. Put the SDK directory, which contains `cslc`, on your `PATH`.

## Usage

Call `csl-pp` instead of `cslc`, with the same arguments:

```bash
csl-pp ./layout.csl --arch=wse3 --fabric-dims=8,3 --fabric-offsets=4,1 \
  -o out --memcpy --channels=1
```

The csl++ libraries are installed with the package; `csl-pp` and `csl-lsp-server` find them without `--import-path`.

## Editor support

The VS Code extension [CSL Language Server](https://marketplace.visualstudio.com/items?itemName=esa-tu-darmstadt.csl-lsp) provides diagnostics, hover, completion and go-to-definition for CSL and csl++. It bundles the language server and does not need this package.

## Bug reports

Please include the output of `csl-pp --version` and the backtrace in bug reports. The debug information of the executables of each release is attached to the release in the [csl++ repository](https://github.com/esa-tu-darmstadt/csl-pp/releases), so backtraces of crashes can be resolved to source lines.

## License

csl++ is under the Apache License v2.0 with LLVM Exceptions. It is developed by the Embedded Systems and Applications Group at the Technical University of Darmstadt and is not a product of Cerebras Systems.
