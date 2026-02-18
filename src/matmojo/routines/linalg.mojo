"""
Defines linear algebra routines for matrices.
"""

from math import sqrt

from matmojo.types.errors import ValueError
from matmojo.types.matrix import Matrix
from matmojo.utils.indexing import get_offset

# ===---------------------------------------------------------------------- ===#
# Transpose
# ===---------------------------------------------------------------------- ===#


fn transpose[dtype: DType](mat: Matrix[dtype]) -> Matrix[dtype]:
    """Returns the transpose of a matrix.

    The result is always stored in row-major (C) order regardless of the
    input layout.

    Parameters:
        dtype: The data type of the matrix elements.

    Args:
        mat: The input matrix.

    Returns:
        A new matrix that is the transpose of the input matrix, with shape
        (ncols, nrows).
    """
    var nrows = mat.ncols  # transposed
    var ncols = mat.nrows  # transposed
    var data = List[Scalar[dtype]](unsafe_uninit_length=nrows * ncols)
    for i in range(mat.nrows):
        for j in range(mat.ncols):
            # result[j, i] = mat[i, j]
            data[j * ncols + i] = mat.data[
                get_offset(i, j, mat.row_stride, mat.col_stride)
            ]
    return Matrix[dtype](
        data=data^,
        nrows=nrows,
        ncols=ncols,
        row_stride=ncols,
        col_stride=1,
    )


# ===---------------------------------------------------------------------- ===#
# Trace
# ===---------------------------------------------------------------------- ===#


fn trace[dtype: DType](mat: Matrix[dtype]) raises ValueError -> Scalar[dtype]:
    """Computes the trace (sum of diagonal elements) of a square matrix.

    Parameters:
        dtype: The data type of the matrix elements.

    Args:
        mat: The input matrix. Must be square.

    Returns:
        The sum of the diagonal elements.

    Raises:
        ValueError: If the matrix is not square.
    """
    if mat.nrows != mat.ncols:
        raise ValueError(
            file="src/matmojo/routines/linalg.mojo",
            function="trace()",
            message="Matrix must be square to compute trace.",
            previous_error=None,
        )
    var result: Scalar[dtype] = 0
    for i in range(mat.nrows):
        result += mat.data[get_offset(i, i, mat.row_stride, mat.col_stride)]
    return result


# ===---------------------------------------------------------------------- ===#
# LU Decomposition (with partial pivoting)
# ===---------------------------------------------------------------------- ===#


fn lu[
    dtype: DType
](
    mat: Matrix[dtype],
) raises ValueError -> Tuple[
    Matrix[dtype], Matrix[dtype], List[Int]
]:
    """Computes the LU decomposition with partial pivoting: PA = LU.

    The input matrix is decomposed into a unit lower-triangular matrix L,
    an upper-triangular matrix U, and a permutation vector P such that
    the rows of A permuted by P equal L @ U.

    Parameters:
        dtype: The data type of the matrix elements.

    Args:
        mat: The input square matrix.

    Returns:
        A tuple (L, U, piv) where:
        - L is a unit lower-triangular matrix (ones on diagonal).
        - U is an upper-triangular matrix.
        - piv is a list of row indices representing the permutation.
          piv[i] = j means that original row j is now row i.

    Raises:
        ValueError: If the matrix is not square.
    """
    if mat.nrows != mat.ncols:
        raise ValueError(
            file="src/matmojo/routines/linalg.mojo",
            function="lu()",
            message="Matrix must be square for LU decomposition.",
            previous_error=None,
        )
    var n = mat.nrows

    # Work on a row-major copy for uniform indexing.
    var u_data = List[Scalar[dtype]](unsafe_uninit_length=n * n)
    for i in range(n):
        for j in range(n):
            u_data[i * n + j] = mat.data[
                get_offset(i, j, mat.row_stride, mat.col_stride)
            ]

    var l_data = List[Scalar[dtype]](length=n * n, fill=0)

    # Initialise piv as identity permutation.
    var piv = List[Int](unsafe_uninit_length=n)
    for i in range(n):
        piv[i] = i

    for k in range(n):
        # --- partial pivoting: find row with largest |u[i,k]| for i >= k ---
        var max_val: Scalar[dtype] = 0
        var max_row = k
        for i in range(k, n):
            var val = u_data[i * n + k]
            # Manual abs (works for both float and int dtypes)
            if val < 0:
                val = -val
            if val > max_val:
                max_val = val
                max_row = i

        # Swap rows in u_data
        if max_row != k:
            for j in range(n):
                var tmp = u_data[k * n + j]
                u_data[k * n + j] = u_data[max_row * n + j]
                u_data[max_row * n + j] = tmp
            # Swap already-computed L columns
            for j in range(k):
                var tmp = l_data[k * n + j]
                l_data[k * n + j] = l_data[max_row * n + j]
                l_data[max_row * n + j] = tmp
            # Swap piv
            var tmp_piv = piv[k]
            piv[k] = piv[max_row]
            piv[max_row] = tmp_piv

        # --- elimination ---
        var pivot = u_data[k * n + k]
        for i in range(k + 1, n):
            var factor = u_data[i * n + k] / pivot
            l_data[i * n + k] = factor
            for j in range(k, n):
                u_data[i * n + j] = (
                    u_data[i * n + j] - factor * u_data[k * n + j]
                )

    # Set L diagonal to 1.
    for i in range(n):
        l_data[i * n + i] = 1

    # Zero out below-diagonal in U (should be ~0 already, but be clean).
    for i in range(n):
        for j in range(i):
            u_data[i * n + j] = 0

    var L = Matrix[dtype](
        data=l_data^, nrows=n, ncols=n, row_stride=n, col_stride=1
    )
    var U = Matrix[dtype](
        data=u_data^, nrows=n, ncols=n, row_stride=n, col_stride=1
    )
    return (L^, U^, piv^)


# ===---------------------------------------------------------------------- ===#
# Cholesky Decomposition
# ===---------------------------------------------------------------------- ===#


fn cholesky[
    dtype: DType
](mat: Matrix[dtype]) raises ValueError -> Matrix[dtype]:
    """Computes the Cholesky decomposition: A = L L^T.

    The input must be a symmetric positive-definite matrix. The result is
    a lower-triangular matrix L such that A = L @ L^T.

    Parameters:
        dtype: The data type of the matrix elements.

    Args:
        mat: A symmetric positive-definite matrix.

    Returns:
        A lower-triangular matrix L.

    Raises:
        ValueError: If the matrix is not square or not positive-definite.
    """
    if mat.nrows != mat.ncols:
        raise ValueError(
            file="src/matmojo/routines/linalg.mojo",
            function="cholesky()",
            message="Matrix must be square for Cholesky decomposition.",
            previous_error=None,
        )
    var n = mat.nrows
    var l_data = List[Scalar[dtype]](length=n * n, fill=0)

    for i in range(n):
        for j in range(i + 1):
            var s: Scalar[dtype] = 0
            for k in range(j):
                s += l_data[i * n + k] * l_data[j * n + k]

            if i == j:
                var diag_val = (
                    mat.data[get_offset(i, i, mat.row_stride, mat.col_stride)]
                    - s
                )
                if diag_val <= 0:
                    raise ValueError(
                        file="src/matmojo/routines/linalg.mojo",
                        function="cholesky()",
                        message=(
                            "Matrix is not positive-definite (non-positive"
                            " diagonal encountered)."
                        ),
                        previous_error=None,
                    )
                l_data[i * n + j] = sqrt(diag_val)
            else:
                l_data[i * n + j] = (
                    mat.data[get_offset(i, j, mat.row_stride, mat.col_stride)]
                    - s
                ) / l_data[j * n + j]

    return Matrix[dtype](
        data=l_data^, nrows=n, ncols=n, row_stride=n, col_stride=1
    )


# ===---------------------------------------------------------------------- ===#
# QR Decomposition (Householder reflections)
# ===---------------------------------------------------------------------- ===#


fn qr[
    dtype: DType
](mat: Matrix[dtype],) raises ValueError -> Tuple[Matrix[dtype], Matrix[dtype]]:
    """Computes the QR decomposition: A = Q R (Householder reflections).

    Works for any m x n matrix with m >= n.

    Parameters:
        dtype: The data type of the matrix elements.

    Args:
        mat: The input matrix (m x n, m >= n).

    Returns:
        A tuple (Q, R) where Q is m x m orthogonal and R is m x n
        upper-triangular.

    Raises:
        ValueError: If m < n.
    """
    var m = mat.nrows
    var n = mat.ncols
    if m < n:
        raise ValueError(
            file="src/matmojo/routines/linalg.mojo",
            function="qr()",
            message="QR decomposition requires nrows >= ncols.",
            previous_error=None,
        )

    # Copy A into row-major R workspace (m x n).
    var r_data = List[Scalar[dtype]](unsafe_uninit_length=m * n)
    for i in range(m):
        for j in range(n):
            r_data[i * n + j] = mat.data[
                get_offset(i, j, mat.row_stride, mat.col_stride)
            ]

    # Q starts as identity (m x m).
    var q_data = List[Scalar[dtype]](length=m * m, fill=0)
    for i in range(m):
        q_data[i * m + i] = 1

    for k in range(n):
        # --- Build Householder vector v for column k below row k ---
        # Compute norm of r_data[k:m, k].
        var sigma: Scalar[dtype] = 0
        for i in range(k, m):
            sigma += r_data[i * n + k] * r_data[i * n + k]
        var norm_x = sqrt(sigma)

        if norm_x == 0:
            continue  # Column is already zero; skip.

        # Choose sign to avoid cancellation.
        var x_k = r_data[k * n + k]
        var sign: Scalar[dtype] = 1
        if x_k < 0:
            sign = -1
        var v_k0 = x_k + sign * norm_x

        # Store v in a temporary list (length m - k). v[0] = v_k0, rest = r[i,k].
        var v_len = m - k
        var v = List[Scalar[dtype]](unsafe_uninit_length=v_len)
        v[0] = v_k0
        for i in range(1, v_len):
            v[i] = r_data[(k + i) * n + k]

        # Compute tau = 2 / (v^T v).
        var vtv: Scalar[dtype] = 0
        for i in range(v_len):
            vtv += v[i] * v[i]
        var tau = Scalar[dtype](2) / vtv

        # --- Apply Householder to R: R[k:m, k:n] -= tau * v * (v^T * R[k:m, k:n]) ---
        for j in range(k, n):
            var dot: Scalar[dtype] = 0
            for i in range(v_len):
                dot += v[i] * r_data[(k + i) * n + j]
            for i in range(v_len):
                r_data[(k + i) * n + j] = (
                    r_data[(k + i) * n + j] - tau * v[i] * dot
                )

        # --- Accumulate Q: Q[:, k:m] -= tau * (Q[:, k:m] * v) * v^T ---
        for i in range(m):
            var dot: Scalar[dtype] = 0
            for j2 in range(v_len):
                dot += q_data[i * m + (k + j2)] * v[j2]
            for j2 in range(v_len):
                q_data[i * m + (k + j2)] = (
                    q_data[i * m + (k + j2)] - tau * dot * v[j2]
                )

    var Q = Matrix[dtype](
        data=q_data^, nrows=m, ncols=m, row_stride=m, col_stride=1
    )
    var R = Matrix[dtype](
        data=r_data^, nrows=m, ncols=n, row_stride=n, col_stride=1
    )
    return (Q^, R^)
