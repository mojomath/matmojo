# MatMojo <!-- omit in toc -->

A matrix and numeric computing library for Mojo.

**[Docs](docs/index.md)** | **[Miji: Design](https://mojo-lang.com/miji/apply/design.html)** | **[Miji: Make it work](https://mojo-lang.com/miji/apply/work.html)**

- [Overview](#overview)
- [Goals](#goals)
- [Install](#install)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Status](#status)
- [License](#license)

## Overview

MatMojo focuses on efficient matrix operations and provides the foundations for linear algebra workflows in Mojo.

Compared to a general-purpose multi-dimensional array library, MatMojo is more specialized and optimized for 2D matrices. This allows us to keep the API small, clean, and focused, while still providing powerful functionality for matrix computations. If you need multi-dimensional arrays, consider the [NuMojo package](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo).

Below are some differences between **MatMojo** (this package) and **NuMojo** (a general-purpose multi-dimensional array library):

| Feature                  | **MatMojo**                                       | **NuMojo**                                         |
| ------------------------ | ------------------------------------------------- | -------------------------------------------------- |
| **Primary goal**         | Linear algebra & matrix computation               | General-purpose ndarray / tensor computing         |
| **Supported dimensions** | 2D only (matrices)                                | Arbitrary dimensions (N-D arrays)                  |
| **Core abstraction**     | Matrix as a mathematical object                   | N-dimensional array container                      |
| **Target domain**        | BLAS / LAPACK style workflows                     | NumPy-style scientific computing                   |
| **Storage model**        | Matrix-specific storage (row/col strides)         | Generic strided N-D storage                        |
| **Static shapes**        | First-class support (compile-time sizes)          | Not a primary focus                                |
| **View semantics**       | Safe read-only + mutable views                    | General slicing & broadcasting                     |
| **Indexing model**       | Strict matrix indexing (row, col)                 | N-dimensional indexing                             |
| **Negative indexing**    | Not supported (explicit & safe)                   | Typically supported                                |
| **Broadcasting**         | Minimal / linear-algebra oriented                 | Full NumPy-style broadcasting                      |
| **Specialized kernels**  | Matmul / decompositions / solvers                 | Elementwise & tensor ops                           |
| **Performance focus**    | SIMD & BLAS-style kernels                         | Generic tensor operations                          |
| **API philosophy**       | Mathematical clarity & safety                     | Flexibility & generality                           |
| **Typical use cases**    | Solvers, decompositions, numerical linear algebra | Scientific computing, ML preprocessing, tensor ops |

## Goals

The initial goal is to support [Mojo Miji](https://mojo-lang.com/miji/) practice content, focus on two-dimensional matrix computing, provide simple and intuitive syntax, and apply a series of targeted optimizations. Throughout the source code, detailed comments and explanations are provided, under the tag `[Mojo Miji]` to help readers understand the design decisions and implementation details.

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
│       │   ├── matrix.mojo          # Dynamic Matrix (owns data)
│       │   ├── matrix_view.mojo     # Non-owning view on Matrix
│       │   ├── static_matrix.mojo   # Compile-time sized Matrix
│       │   └── errors.mojo
│       ├── routines/
│       │   ├── __init__.mojo
│       │   ├── creation.mojo        # matrix(), smatrix() constructors
│       │   ├── math.mojo            # add, sub, mul, div, matmul
│       │   ├── linalg.mojo          # (planned) decompositions, solvers
│       │   ├── random.mojo          # (planned) random matrix generation
│       │   └── statistics.mojo      # (planned) mean, var, cov, etc.
│       ├── traits/
│       │   ├── __init__.mojo
│       │   └── matrix_like.mojo     # MatrixLike trait
│       └── utils/
│           ├── __init__.mojo
│           ├── indexing.mojo
│           ├── str.mojo
│           └── io.mojo              # (planned)
└── tests
    ├── test_all.sh
    ├── matrix/
    │   ├── test_matrix_creation.mojo
    │   ├── test_matrix_indexing.mojo
    │   ├── test_matrix_lifecycle.mojo
    │   └── test_matrix_str.mojo
    ├── matrix_view/
    │   └── test_matrix_view.mojo
    ├── static_matrix/
    │   └── test_static_matrix.mojo
    └── routines/
        └── test_math.mojo
```

## Status

MatMojo is under active development as a tutorial companion project. Issues and suggestions are welcome.

## License

Apache License 2.0. See [LICENSE](LICENSE).
