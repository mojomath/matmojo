# MatMojo <!-- omit in toc -->

A matrix computation library in Mojo, built for [Mojo Miji](https://mojo-lang.com/miji/) practice.

**[Docs](docs/index.md)** | **[Miji: Design](https://mojo-lang.com/miji/apply/design.html)** | **[Miji: Make it work](https://mojo-lang.com/miji/apply/work.html)**

- [Overview](#overview)
- [Goals](#goals)
- [Install](#install)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Status](#status)
- [License](#license)

## Overview

MatMojo is a learning-focused Mojo package that implements a 2D matrix type and its core operations. The initial goal is to support [Mojo Miji](https://mojo-lang.com/miji/) practice content, focus on two-dimensional matrix computing, provide simple and intuitive syntax, and apply a series of targeted optimizations. Throughout the source code, detailed comments and explanations are provided, under the tag `[Mojo Miji]` to help readers understand the design decisions and implementation details.

Compared to a general-purpose multi-dimensional array library, MatMojo is more specialized and optimized for 2D matrices. This allows us to keep the API small, clean, and focused, while still providing powerful functionality for matrix computations.

If you need multi-dimensional arrays, consider the [NuMojo package](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo).

## Goals

- Keep the API small and easy to read while learning Mojo.
- Provide simple and intuitive syntax for matrix creation and operations.
- Use safe Mojo features and avoid unsafe code as much as possible.
- Emphasize contiguous storage for 2D matrices, but also support non-contiguous views through strides.
- Optimize core operations like matrix multiplication which makes this package a better tool if you want to only use 2D matrices.

## Install

This project uses pixi for environment management.

```bash
pixi install
```

## Quick start

Build the package and run the interactive test:

```bash
pixi run test
```

Minimal usage example:

```mojo
fn main() raises:
    var mat = Matrix(
        [
            [1.1, 1.2, 1.3, 1.4, 1.5],
            [2.1, 2.2, 2.3, 2.4, 2.5],
            [3.1, 3.2, 3.3, 3.4, 3.5],
            [4.1, 4.2, 4.3, 4.4, 4.5],
            [5.1, 5.2, 5.3, 5.4, 5.5],
        ]
    )
    print(mat)
```

## Project structure

```sh
matmojo
├── pixi.toml
├── src
│   └── matmojo
│       ├── __init__.mojo
│       ├── prelude.mojo
│       ├── types/
│       │   ├── __init__.mojo
│       │   ├── matrix.mojo
│       │   ├── matrix_view.mojo
│       │   └── errors.mojo
│       ├── routines/
│       │   ├── __init__.mojo
│       │   ├── creation.mojo
│       │   ├── math.mojo
│       │   ├── linalg.mojo
│       │   ├── random.mojo
│       │   └── statistics.mojo
│       └── utils/
│           ├── __init__.mojo
│           ├── io.mojo
│           ├── str.mojo
│           └── validation.mojo
└── tests
    ├── matmojo.mojopkg
    └── test.mojo
```

## Status

MatMojo is under active development as a tutorial companion project. Issues and suggestions are welcome.

## License

Apache License 2.0. See [LICENSE](LICENSE).
