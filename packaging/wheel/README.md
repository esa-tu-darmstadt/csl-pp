# csl++

csl++ is a frontend for the Cerebras Software Language (CSL), developed by
the Embedded Systems and Applications Group at the Technical University of
Darmstadt. It adds async functions and tasks and automatic allocation of
colors, task IDs and queues to CSL. Its compiler, `csl-pp`, evaluates the
program at compile time, translates it to vanilla CSL and compiles the result
with `cslc` of the Cerebras SDK. `csl-pp` accepts the command-line arguments
of `cslc`.

This package contains `csl-pp`, the language server `csl-lsp-server` and the
csl++ libraries for x86-64 Linux with glibc 2.28 or newer.

## Installation

```bash
pipx install csl-pp
```

`uv tool install csl-pp` and `pip install csl-pp` in a virtual environment
work as well.

`csl-pp` runs `cslc` of the [Cerebras SDK](https://sdk.cerebras.ai), which is
installed separately: put the SDK directory, which contains `cslc`, on your
`PATH`.

## Usage

Call `csl-pp` instead of `cslc`, with the same arguments:

```bash
csl-pp --arch=wse3 ./layout.csl --fabric-dims=8,3 --fabric-offsets=4,1 \
  -o out --memcpy --channels 1
```

The csl++ libraries are installed with the package; `csl-pp` and
`csl-lsp-server` find them without `--import-path`.

## Bug reports

The debug information of the executables of each release is attached to the
release in the [csl++ repository](https://github.com/esa-tu-darmstadt/csl-pp/releases), so backtraces of crashes
can be resolved to source lines. Please include the
output of `csl-pp --version` and the backtrace in bug reports.
