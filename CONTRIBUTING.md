# Contributing to csl++

Bug reports, questions and pull requests are welcome in this repository.

`docs/Building.md` describes the build from source, the tests and the release process. The Contributing page of the website, whose source is `website/src/content/docs/project/contributing.mdx`, gives an overview of the repository.

Run the regression tests before opening a pull request:

```bash
cmake --build build/Release --target check-csl
```

Contributions are licensed under the license of the project, the Apache License v2.0 with LLVM Exceptions (`LICENSE.TXT`).
