# csl++ website

The documentation website of csl++, built with [Starlight](https://starlight.astro.build). It needs Node.js 22 or newer.

```bash
npm ci
npm run dev     # local server with live reload
npm run check   # build into dist/ and check all links
```

`SITE_URL` sets the public URL of the site, e.g. `SITE_URL=https://esa-tu-darmstadt.github.io/csl-pp`; the base path is taken from it. CI builds the site and publishes it with GitHub Pages (`.github/workflows/website.yml`).

## Layout

| Path | Content |
| --- | --- |
| `src/content/docs/` | The pages. The sidebar is defined in `astro.config.mjs`. |
| `examples/` | Complete programs shown on the pages. |
| `src/components/Example.astro` | Shows an example file, or a region of it. |
| `scripts/check-links.mjs` | Checks the links and anchors of the built site. |

CSL code is highlighted with the grammar of the VS Code extension in `editors/vscode`.

## Examples

Every example in `examples/<group>/<name>/` is a complete program with a layout file and its PE programs in `src/`, a host program `run.py` and the scripts `commands_wse2.sh` and `commands_wse3.sh`. The integration tests (`check-csl-integration`) run these scripts, so the code on the site is compiled and run in the simulator.

A page shows a file, or a region of it, with

```mdx
<Example file="tutorial/4-parallel/src/receiver.csl" region="task" />
```

where the region is marked in the file by `// region: task` and `// endregion` lines. Marker lines are not shown. Only the examples carry such markers. Code of the libraries in `csl-libs/` that a page shows is written into the page.
