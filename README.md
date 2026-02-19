# MatMojo <!-- omit in toc -->

A matrix and linear algebra library for Mojo.

**[Docs](docs/index.md)** | **[Roadmap](docs/ROADMAP.md)** | **[Miji: Design](https://mojo-lang.com/miji/apply/design.html)** | **[Miji: Make it work](https://mojo-lang.com/miji/apply/work.html)**

- [Overview](#overview)
- [Goals](#goals)
- [Install](#install)
- [Quick start](#quick-start)
  - [Create matrices](#create-matrices)
  - [Arithmetic](#arithmetic)
  - [Linear algebra](#linear-algebra)
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

Run the test suite:

```bash
pixi run test
```

### Create matrices

```mojo
import matmojo as mm

fn main() raises:
    # From nested lists
    var A = mm.matrix[DType.float64](
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0],
         [7.0, 8.0, 9.0]]
    )
    print(A)

    # Convenience constructors
    var I = mm.eye[DType.float64](3)       # 3×3 identity
    var Z = mm.zeros[DType.float64](2, 4)  # 2×4 zeros
    var O = mm.ones[DType.float64](3, 3)   # 3×3 ones
```

### Arithmetic

```mojo
    # Element-wise operators
    var B = A + O   # addition
    var C = A * A   # Hadamard product
    var D = A @ A   # matrix multiplication

    # Scalar operations
    from matmojo.routines.math import scalar_mul
    var scaled = scalar_mul(A, 2.0)
```

### Linear algebra

```mojo
    # Transpose & trace
    var At = mm.transpose(A)
    var t  = mm.trace(A)

    # LU decomposition (PA = LU)
    var lup = mm.lu(A)
    var L   = lup[0].copy()
    var U   = lup[1].copy()
    var piv = lup[2].copy()

    # Cholesky (A = LL^T, requires SPD matrix)
    var spd = mm.matrix[DType.float64](
        [[4.0, 12.0, -16.0],
         [12.0, 37.0, -43.0],
         [-16.0, -43.0, 98.0]]
    )
    var Lc = mm.cholesky(spd)

    # QR decomposition (A = QR)
    var qr_result = mm.qr(A)
    var Q = qr_result[0].copy()
    var R = qr_result[1].copy()
```

## Project structure

```sh
matmojo
├── pixi.toml
├── src/matmojo
│   ├── __init__.mojo
│   ├── types/
│   │   ├── matrix.mojo          # Dynamic Matrix (row/col-major)
│   │   ├── matrix_view.mojo     # Non-owning view with slicing
│   │   ├── static_matrix.mojo   # Compile-time sized Matrix
│   │   └── errors.mojo          # ValueError, IndexError, etc.
│   ├── routines/
│   │   ├── creation.mojo        # matrix, zeros, ones, full, eye, diag
│   │   ├── math.mojo            # add, sub, mul, div, matmul, scalar ops
│   │   └── linalg.mojo          # transpose, trace, lu, cholesky, qr
│   ├── traits/
│   │   └── matrix_like.mojo     # MatrixLike trait
│   └── utils/
│       ├── indexing.mojo
│       └── str.mojo
└── tests/
    ├── test_all.sh
    ├── matrix/                   # Matrix creation, indexing, lifecycle, str
    ├── matrix_view/              # View slicing, view-on-view
    ├── static_matrix/            # StaticMatrix tests
    └── routines/                 # creation, linalg, math, decompositions
```

## Status

MatMojo is under active development. Current progress:

- **Phase 0** ✅ Core types (`Matrix`, `StaticMatrix`, `MatrixView`), basic ops, CI
- **Phase 1** ✅ Creation routines, transpose, trace, element-wise & scalar ops
- **Phase 2** ✅ LU, Cholesky, QR decompositions

See the [Roadmap](docs/ROADMAP.md) for upcoming phases (solvers, eigenvalues, statistics, etc.).

## License

Apache License 2.0. See [LICENSE](LICENSE).
