# Roadmap <!-- omit in toc -->

MatMojo development roadmap. Phases are prioritized for use as the linear algebra foundation of [stamojo](https://github.com/forfudan/stamojo) (a statistical modeling library, similar to statsmodels).

Last reviewed: **2026-02-18**

- [Phase 0 — Core Types \& Basic Operations](#phase-0--core-types--basic-operations)
- [Phase 1 — Matrix Fundamentals](#phase-1--matrix-fundamentals)
- [Phase 2 — Decompositions](#phase-2--decompositions)
- [Phase 3 — Solvers \& Inverse](#phase-3--solvers--inverse)
- [Phase 4 — Eigenvalue Problems](#phase-4--eigenvalue-problems)
- [Phase 5 — Statistics Primitives](#phase-5--statistics-primitives)
- [Phase 6 — Norms \& Conditioning](#phase-6--norms--conditioning)
- [Phase 7 — Random Matrix Generation](#phase-7--random-matrix-generation)
- [Phase 8 — Performance \& Polish](#phase-8--performance--polish)

---

## Phase 0 — Core Types & Basic Operations

> **Status: ✅ Done**

| Item | Module | Status |
|------|--------|--------|
| `Matrix` type (dynamic, row/col-major) | `types/matrix.mojo` | ✅ |
| `StaticMatrix` type (compile-time sized) | `types/static_matrix.mojo` | ✅ |
| `MatrixView` (non-owning view, slicing, view-on-view) | `types/matrix_view.mojo` | ✅ |
| `MatrixLike` trait | `traits/matrix_like.mojo` | ✅ |
| `matrix()` / `smatrix()` creation routines | `routines/creation.mojo` | ✅ |
| Element-wise `add`, `sub`, `mul`, `div` (StaticMatrix) | `routines/math.mojo` | ✅ |
| `matmul` (naive + SIMD/parallel for dynamic) | `routines/math.mojo` | ✅ |
| Custom error types (`ValueError`, `IndexError`, etc.) | `types/errors.mojo` | ✅ |
| Unit test suite (51 tests, TestSuite.discover_tests) | `tests/` | ✅ |
| CI: GitHub Actions + pre-commit (mojo format) | `.github/workflows/` | ✅ |

---

## Phase 1 — Matrix Fundamentals

> **Status: 🔲 Not started**
>
> *stamojo dependency: blocking — nearly every statistical model needs these.*

| Item | Module | stamojo use |
|------|--------|-------------|
| `transpose()` / `.T` property | `routines/linalg.mojo` | Design matrices, X^T X |
| `eye()` / `identity()` | `routines/creation.mojo` | Ridge regression, regularization |
| `diag()` (extract / construct diagonal) | `routines/creation.mojo` | Variance extraction from cov matrix |
| `trace()` | `routines/linalg.mojo` | Matrix diagnostics |
| `zeros()` / `ones()` / `full()` | `routines/creation.mojo` | Convenience constructors |
| Element-wise ops for dynamic `Matrix` (`add`, `sub`, `mul`, `div`) | `routines/math.mojo` | Residual computation |
| Scalar–matrix operations (`a * M`, `M + c`) | `routines/math.mojo` | Scaling, centering |

---

## Phase 2 — Decompositions

> **Status: 🔲 Not started**
>
> *stamojo dependency: blocking — cannot implement OLS, GLS, or WLS without these.*

| Item | Module | stamojo use |
|------|--------|-------------|
| LU decomposition (with partial pivoting) | `routines/linalg.mojo` | `solve()`, `inv()`, `det()` |
| Cholesky decomposition | `routines/linalg.mojo` | Efficient solve for positive-definite (covariance) matrices |
| QR decomposition (Householder) | `routines/linalg.mojo` | Numerically stable least squares (OLS) |

---

## Phase 3 — Solvers & Inverse

> **Status: 🔲 Not started**
>
> *stamojo dependency: blocking — regression coefficients require `solve` or `inv`.*

| Item | Module | stamojo use |
|------|--------|-------------|
| `inv()` — matrix inverse | `routines/linalg.mojo` | β̂ = (X^T X)^{-1} X^T y |
| `solve()` — solve Ax = b | `routines/linalg.mojo` | Linear system solving |
| `det()` — determinant (via LU) | `routines/linalg.mojo` | Singularity check |
| `lstsq()` — least squares (via QR) | `routines/linalg.mojo` | OLS regression |

---

## Phase 4 — Eigenvalue Problems

> **Status: 🔲 Not started**
>
> *stamojo dependency: important for PCA and diagnostics, not blocking for basic regression.*

| Item | Module | stamojo use |
|------|--------|-------------|
| `eig()` — eigenvalues + eigenvectors | `routines/linalg.mojo` | PCA, principal component regression |
| `eigvals()` — eigenvalues only | `routines/linalg.mojo` | Condition number, multicollinearity |
| `svd()` — singular value decomposition | `routines/linalg.mojo` | Pseudo-inverse, rank, PCA |

---

## Phase 5 — Statistics Primitives

> **Status: 🔲 Not started**
>
> *stamojo dependency: important — descriptive stats and residual analysis.*

| Item | Module | stamojo use |
|------|--------|-------------|
| `sum()` (along axis / full) | `routines/statistics.mojo` | Data aggregation |
| `mean()` (along axis / full) | `routines/statistics.mojo` | Centering, descriptive stats |
| `var()` / `std()` (along axis) | `routines/statistics.mojo` | Variance estimation, standardization |
| `cov()` — covariance matrix | `routines/statistics.mojo` | Covariance estimation |
| `corrcoef()` — correlation matrix | `routines/statistics.mojo` | Correlation analysis |

---

## Phase 6 — Norms & Conditioning

> **Status: 🔲 Not started**
>
> *stamojo dependency: useful for diagnostics and numerical stability.*

| Item | Module | stamojo use |
|------|--------|-------------|
| `norm()` (Frobenius, L1, L2, inf) | `routines/linalg.mojo` | Residual norms, convergence |
| `cond()` — condition number | `routines/linalg.mojo` | Multicollinearity detection |
| `rank()` — matrix rank | `routines/linalg.mojo` | Rank-deficiency check |
| `pinv()` — pseudo-inverse (Moore–Penrose) | `routines/linalg.mojo` | Rank-deficient least squares |

---

## Phase 7 — Random Matrix Generation

> **Status: 🔲 Not started**
>
> *stamojo dependency: needed for simulation, bootstrap, MCMC.*

| Item | Module | stamojo use |
|------|--------|-------------|
| `rand()` — uniform random matrix | `routines/random.mojo` | Monte Carlo simulation |
| `randn()` — normal random matrix | `routines/random.mojo` | Error simulation, bootstrap |
| `seed()` — set RNG seed | `routines/random.mojo` | Reproducibility |

---

## Phase 8 — Performance & Polish

> **Status: 🔲 Not started**

| Item | Module | Notes |
|------|--------|-------|
| Optimized matmul for all layout combos (C@C, F@F, C@F, F@C, V@*) | `routines/math.mojo` | See [API.md](API.md) |
| Tiled / blocked matmul | `routines/math.mojo` | Cache-friendly |
| SIMD-optimized decompositions | `routines/linalg.mojo` | Performance |
| Parallel row/col operations | `routines/math.mojo` | Multi-core utilization |
| Comprehensive benchmarks | `benches/` | Compare vs. NumPy/LAPACK |

---

## Review Log

| Date | Reviewer | Notes |
|------|----------|-------|
| 2026-02-18 | ZHU | Initial roadmap created. Phase 0 complete. |
