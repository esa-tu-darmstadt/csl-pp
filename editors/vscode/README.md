# Language Server for CSL

Language support for the Cerebras Software Language (CSL) and its extension [csl++](https://esa-tu-darmstadt.github.io/csl-pp/) in VS Code. The extension is developed at TU Darmstadt and is not a product of Cerebras Systems.

## Features

- Diagnostics while you type
- The type of a value on hover
- Go to definition, including into imported modules and the system modules of the SDK
- Completion, including members after `.`
- Signature help for calls
- The outline of a file
- Syntax highlighting

The extension runs `csl-lsp-server`, the language server of csl++. The server uses the frontend of the csl++ compiler `csl-pp`. In CSL, types are values that are computed at compile time, so the type of an expression is only known after the comptime code of the whole program has been evaluated. The server performs this evaluation in full and does not approximate it.

## Requirements

- Linux on x86-64. On another system, use the extension in a remote window on a Linux machine, with the Remote - SSH or WSL extension of VS Code.
- The Cerebras SDK, from which the server reads the system modules.

## Setup

Set `csl.systemModuleSif` to the `*.sif` file in the directory of the Cerebras SDK. The server reads the system modules of the SDK, such as `<memcpy/memcpy>`, from this file. Without the setting, it reports an error for every import of a system module.

To change a setting, open the Settings editor with `Ctrl+,` and search for `csl`, or run "Preferences: Open User Settings (JSON)" from the Command Palette and add the setting to the file:

```json
{
  "csl.systemModuleSif": "/home/user/cerebras-sdk/sdk-cbcore-2.10.0.sif"
}
```

The architecture and the parameters of a program belong in the workspace settings, the file `.vscode/settings.json` in the folder of the program:

```json
{
  "csl.arch": "wse3",
  "csl.params": { "width": 4, "N": 8 }
}
```

Keep the layout file of a program open while you edit its PE programs. The server checks a PE program with the parameters that the `@set_tile_code` calls of the open layout file pass to it.

The [documentation](https://esa-tu-darmstadt.github.io/csl-pp/tools/vscode/) describes the setup in more detail.

## Settings

| Setting | Meaning | Default |
| --- | --- | --- |
| `csl.systemModuleSif` | The SDK image (`*.sif`) from which the server reads the system modules. | none |
| `csl.systemImportPath` | A directory with the system modules, in place of the image. If both settings are given, it is the directory of the system modules within the image. | none |
| `csl.arch` | The target architecture, `wse2` or `wse3`. | `wse2` |
| `csl.params` | The values of the `param` declarations of the layout file, as an object of names and numbers. A number with a decimal point is a `comptime_float`, any other number a `comptime_int`. | `{}` |
| `csl.fabricDims` | The fabric dimensions, as `width,height`. | `100,100` |
| `csl.fabricOffsets` | The offset of the program rectangle in the fabric, as `x,y`. | `3,3` |
| `csl.rectangle` | The size of the program rectangle that `@get_rectangle` returns, as `width,height`. | `100,100` |
| `csl.serverPath` | A `csl-lsp-server` to run in place of the one in the extension. | the server in the extension |
| `csl.serverArgs` | Further command-line arguments of the server. | `[]` |
| `csl.filterOverlappingErrors` | If several errors overlap, show only the innermost one. | `false` |
| `csl.trace.server` | Log the messages between VS Code and the server: `off`, `messages` or `verbose`. | `off` |

## Feedback

Report problems and feature requests in the [issue tracker](https://github.com/esa-tu-darmstadt/csl-pp/issues). If the language server and the CSL compiler of the SDK disagree about a program, please include a small program that shows the difference.

For questions about the project or about a collaboration, write to noack@esa.tu-darmstadt.de.

The extension works next to the extension "Cerebras Software Language (CSL)" of Cerebras, which provides syntax highlighting only.

## License

The extension is under the Apache License v2.0 with LLVM Exceptions. It is developed by the Embedded Systems and Applications Group at the Technical University of Darmstadt as part of csl++.
