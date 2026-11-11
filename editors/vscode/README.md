# CSL Language Server

Unofficial language server for the Cerebras Software Language (CSL), providing rich editor features beyond syntax highlighting.

<!-- ![CSL Language Server in action](images/demo.gif) -->

## Required Setup

Before using the extension, set **`csl.systemModuleSif`** to point to the `.sif` file in the Cerebras SDK. The language server extracts system modules from that container to resolve imports with angle bracket syntax (`<...>`). Alternatively, you can set `csl.systemImportPath` to a directory containing the extracted system modules.

## Features

- **Diagnostics** -- real-time error and warning reporting as you type
- **Hover** -- type information and documentation on hover
- **Go to Definition** -- jump to definitions of functions, variables, and imports
- **Auto-completion** -- context-aware code completion suggestions
- **Dot completion** -- member/field completions after `.`
- **Function signature help** -- call argument hints with active parameter highlighting
- **Document Symbols** -- outline view and symbol search within a file

## Why this language server exists

The language server uses the frontend of `csl-pp`, the compiler of csl++, which is developed in a research project at TU Darmstadt.
`csl-pp` performs whole-program transformations, such as the lowering of async functions.

Those transformations depend on accurate static types for the full program.  
In CSL (similar to Zig), types are first-class values, and thus infering static type information requires comptime evaluation of the complete program.

For that reason, comptime-related analysis in this extension is intended to model CSL semantics fully, not as a best-effort approximation for editor tooling.

If you observe a mismatch with the official CSL compiler (especially after SDK updates), please open an issue and include a minimal repro.

## Settings

Configure the extension under `csl.*` in your VS Code settings:

| Setting                | Description                                              | Default     |
|------------------------|----------------------------------------------------------|-------------|
| `csl.systemModuleSif`  | Path to a `.sif` container providing system modules. Required for resolving imports with angle bracket syntax (`<...>`). | (empty) |
| `csl.systemImportPath` | Search path for system module imports (`<...>` syntax). | `/cb/toolchains/cslang/*/*/csl-libs/` if `csl.systemModuleSif` is set, otherwise empty |
| `csl.arch`             | Target architecture (optional).                          | `wse2`      |
| `csl.params`           | Parameter bindings as key-value pairs (optional). Numbers containing `.` are `comptime_float`, otherwise `comptime_int`. | `{}` |
| `csl.rectangle`        | Rectangle size as `width,height` for `@get_rectangle`.   | `100,100`   |
| `csl.fabricDims`       | Fabric dimensions as `width,height`.                     | `100,100`   |
| `csl.fabricOffsets`    | Fabric offsets as `x,y`.                                 | `3,3`       |
| `csl.serverArgs`       | Additional CLI arguments passed to the server.           | `[]`        |
| `csl.trace.server`     | Trace LSP communication (`off`, `messages`, `verbose`).  | `off`       |

### Example Configuration

```json
{
  "csl.systemModuleSif": "/opt/Cerebras-SDK-1.4.0-202505230211-4-d9070058/sdk-cbcore-202505230211-4-9382352f.sif",
  "csl.arch": "wse3",
  "csl.params": {
    "M": 4,
    "N": 4,
    "width": 8
  }
}
```

## About

This extension is part of a research project at TU Darmstadt (Embedded Systems and Applications Group) that builds MLIR-based compiler infrastructure for Cerebras Wafer-Scale Engines (WSE).

## Feedback and Contact

Bug reports and feature requests can be filed on the [GitHub issue tracker](https://github.com/esa-tu-darmstadt/csl-lsp-vscode/issues).

If you are interested in the project or in collaborating, feel free to reach out by email: noack@esa.tu-darmstadt.de

## Note

If you have the official "Cerebras Software Language (CSL)" extension installed, you can keep it -- both extensions work side by side without conflicts. That extension provides syntax highlighting only; this extension includes syntax highlighting plus full language server features.
