"""
Defines mathematical routines for matrices.
"""

from algorithm import vectorize, parallelize
from sys import simd_width_of

from matmojo.types.errors import ValueError
from matmojo.types.static_matrix import StaticMatrix
from matmojo.types.matrix import Matrix
from matmojo.types.matrix_view import MatrixView
from matmojo.traits.matrix_like import MatrixLike
from matmojo.utils.indexing import get_offset

# [Mojo Miji]
# We use the `MatrixLike` trait to perform item-wise, naive matrix operations.
# It allows us to quickly make things "work".
# We can further optimize these operations by implementing specific algorithms
# for specific matrix types.

# ===---------------------------------------------------------------------- ===#
# Matrix addition
# ===---------------------------------------------------------------------- ===#


# FIXME: When Mojo support parameterized traits.
# fn add[M: MatrixLike](a: M, b: M) raises ValueError -> M:
#     """Performs element-wise addition of two matrices.

#     Parameters:
#         M: The type of the input matrices, which must implement the MatrixLike trait.

#     Args:
#         a: The first input matrix.
#         b: The second input matrix.

#     Returns:
#         A new matrix containing the element-wise sum of a and b.

#     Raises:
#         ValueError: If the shapes of a and b do not match.
#     """
#     if a.get_nrows() != b.get_nrows() or a.get_ncols() != b.get_ncols():
#         raise ValueError(
#             file="src/matmojo/routines/math.mojo",
#             function="add()",
#             message="Input matrices must have the same shape.",
#             previous_error=None,
#         )
#     var result = a.copy()
#     for i in range(a.get_nrows()):
#         for j in range(a.get_ncols()):
#             result[i, j] = a[i, j] + b[i, j]
#     return result^


fn add[
    dtype: DType, nrows: Int, ncols: Int
](
    a: StaticMatrix[dtype, nrows, ncols], b: StaticMatrix[dtype, nrows, ncols]
) -> StaticMatrix[dtype, nrows, ncols]:
    """Performs element-wise addition of two matrices.

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new matrix containing the element-wise sum of a and b.
    """
    return StaticMatrix[dtype, nrows, ncols](a.data + b.data)


fn sub[
    dtype: DType, nrows: Int, ncols: Int
](
    a: StaticMatrix[dtype, nrows, ncols], b: StaticMatrix[dtype, nrows, ncols]
) -> StaticMatrix[dtype, nrows, ncols]:
    """Performs element-wise subtraction of two matrices.

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new matrix containing the element-wise difference of a and b.
    """
    return StaticMatrix[dtype, nrows, ncols](a.data - b.data)


fn mul[
    dtype: DType, nrows: Int, ncols: Int
](
    a: StaticMatrix[dtype, nrows, ncols], b: StaticMatrix[dtype, nrows, ncols]
) -> StaticMatrix[dtype, nrows, ncols]:
    """Performs element-wise multiplication of two matrices.

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new matrix containing the element-wise product of a and b.
    """
    return StaticMatrix[dtype, nrows, ncols](a.data * b.data)


fn div[
    dtype: DType, nrows: Int, ncols: Int
](
    a: StaticMatrix[dtype, nrows, ncols], b: StaticMatrix[dtype, nrows, ncols]
) -> StaticMatrix[dtype, nrows, ncols]:
    """Performs element-wise division of two matrices.

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new matrix containing the element-wise quotient of a and b.
    """
    return StaticMatrix[dtype, nrows, ncols](a.data / b.data)


fn matmul[
    dtype: DType, nrows: Int, ncols: Int, inner_dim: Int
](
    a: StaticMatrix[dtype, nrows, inner_dim],
    b: StaticMatrix[dtype, inner_dim, ncols],
) -> StaticMatrix[dtype, nrows, ncols]:
    """Performs matrix multiplication of two matrices.

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new matrix containing the product of a and b.
    """
    var result = StaticMatrix[dtype, nrows, ncols]()
    for i in range(nrows):
        for j in range(ncols):
            var sum: Scalar[dtype] = 0
            for k in range(inner_dim):
                sum += a[i, k] * b[k, j]
            result.data[i * result.BUFFER_COL_LEN + j] = sum
    return result^


# --------------------------------------------------------------------------- #
# Core view-based matmul implementation
# --------------------------------------------------------------------------- #
# [Mojo Miji]
# The canonical SIMD-optimised matmul operates on MatrixView.
# Converting Matrix → MatrixView via `.view()` is free (metadata copy only;
# the underlying data lives in a Span that borrows from the Matrix's List).
# This avoids duplicating the implementation for every Matrix/View combination.


fn _matmul_view_simd[
    dtype: DType,
    origin_a: Origin,
    origin_b: Origin,
](
    a: MatrixView[dtype, origin_a],
    b: MatrixView[dtype, origin_b],
) raises ValueError -> Matrix[dtype]:
    """Core matrix multiplication on MatrixView operands.

    Dispatches to:
    * **SIMD fast-path** – when both operands are C-contiguous
      (vectorize across b's columns + parallelize over a's rows).
    * **General fallback** – stride-aware scalar access, parallelize over rows.
      Handles arbitrary memory layouts (non-contiguous views, sliced views, etc.).

    The result is always a freshly allocated, C-contiguous Matrix.
    """
    comptime simd_w = simd_width_of[dtype]()

    if a.ncols != b.nrows:
        raise ValueError(
            file="src/matmojo/routines/math.mojo",
            function="matmul()",
            message=(
                "Inner dimensions of a and b must match for matrix"
                " multiplication."
            ),
            previous_error=None,
        )

    var M = a.nrows
    var N = b.ncols
    var K = a.ncols

    # Result is always C-contiguous (row-major).
    var result = Matrix[dtype](M, N, N, 1)

    # ---------------------------------------------------------------------- #
    # Fast path: both a and b are C-contiguous → SIMD vectorize + parallelize
    # ---------------------------------------------------------------------- #
    if a.is_c_contiguous() and b.is_c_contiguous():
        # Obtain raw pointers for SIMD load/store.
        # Span.unsafe_ptr() gives the UnsafePointer to the start of the
        # underlying buffer; `offset` advances to the view's first element.
        var a_ptr = a.data.unsafe_ptr()
        var b_ptr = b.data.unsafe_ptr()
        var a_off = a.offset
        var b_off = b.offset
        var a_rs = a.row_stride
        var b_rs = b.row_stride

        @parameter
        fn process_row(i: Int):
            for k in range(K):
                # [Mojo Miji]
                # Use each element in row i of a to multiply the whole row k
                # of b, and accumulate into row i of result.
                @parameter
                fn vec_col[
                    w: Int
                ](j: Int) unified {
                    mut result,
                    read a_ptr,
                    read b_ptr,
                    read a_off,
                    read b_off,
                    read a_rs,
                    read b_rs,
                    read N,
                    read i,
                    read k,
                }:
                    var r_idx = i * N + j
                    result.data._data.store[width=w](
                        r_idx,
                        result.data._data.load[width=w](r_idx)
                        + a_ptr.load[width=1](a_off + i * a_rs + k)
                        * b_ptr.load[width=w](b_off + k * b_rs + j),
                    )

                vectorize[simd_w](N, vec_col)

        parallelize[process_row](M, M)

    # ---------------------------------------------------------------------- #
    # General fallback: any memory layout (strided, sliced, transposed…)
    # ---------------------------------------------------------------------- #
    else:
        var a_off = a.offset
        var b_off = b.offset
        var a_rs = a.row_stride
        var a_cs = a.col_stride
        var b_rs = b.row_stride
        var b_cs = b.col_stride

        @parameter
        fn process_row_general(i: Int):
            for j in range(N):
                var sum: Scalar[dtype] = 0
                for k in range(K):
                    sum += (
                        a.data[a_off + i * a_rs + k * a_cs]
                        * b.data[b_off + k * b_rs + j * b_cs]
                    )
                result.data._data.store[width=1](i * N + j, sum)

        parallelize[process_row_general](M, M)

    return result^


# --------------------------------------------------------------------------- #
# Public matmul overloads
# --------------------------------------------------------------------------- #


fn matmul[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises ValueError -> Matrix[dtype]:
    """Performs matrix multiplication of two dynamic matrices.

    Delegates to the SIMD-optimised, view-based core implementation.
    Converting Matrix → MatrixView via `.view()` is free (metadata copy).

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new C-contiguous matrix containing the product of a and b.
    """
    return _matmul_view_simd(a.view(), b.view())


fn matmul[
    dtype: DType,
    origin_a: Origin,
    origin_b: Origin,
](
    a: MatrixView[dtype, origin_a],
    b: MatrixView[dtype, origin_b],
) raises ValueError -> Matrix[dtype]:
    """Performs matrix multiplication of two matrix views.

    This is the canonical entry-point for view × view multiplication.

    Args:
        a: The first input matrix view.
        b: The second input matrix view.

    Returns:
        A new C-contiguous matrix containing the product of a and b.
    """
    return _matmul_view_simd(a, b)


fn matmul[
    dtype: DType,
    origin_b: Origin,
](
    a: Matrix[dtype],
    b: MatrixView[dtype, origin_b],
) raises ValueError -> Matrix[dtype]:
    """Performs matrix multiplication of a matrix and a matrix view.

    Args:
        a: The first input matrix.
        b: The second input matrix view.

    Returns:
        A new C-contiguous matrix containing the product of a and b.
    """
    return _matmul_view_simd(a.view(), b)


fn matmul[
    dtype: DType,
    origin_a: Origin,
](
    a: MatrixView[dtype, origin_a],
    b: Matrix[dtype],
) raises ValueError -> Matrix[dtype]:
    """Performs matrix multiplication of a matrix view and a matrix.

    Args:
        a: The first input matrix view.
        b: The second input matrix.

    Returns:
        A new C-contiguous matrix containing the product of a and b.
    """
    return _matmul_view_simd(a, b.view())


# ===---------------------------------------------------------------------- ===#
# Dynamic Matrix element-wise operations
# ===---------------------------------------------------------------------- ===#


fn add[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises ValueError -> Matrix[dtype]:
    """Performs element-wise addition of two dynamic matrices.

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new matrix containing the element-wise sum of a and b.

    Raises:
        ValueError: If the shapes of a and b do not match.
    """
    if a.nrows != b.nrows or a.ncols != b.ncols:
        raise ValueError(
            file="src/matmojo/routines/math.mojo",
            function="add()",
            message="Input matrices must have the same shape.",
            previous_error=None,
        )
    var data = List[Scalar[dtype]](unsafe_uninit_length=a.nrows * a.ncols)
    for i in range(a.nrows):
        for j in range(a.ncols):
            data[i * a.ncols + j] = (
                a.data[get_offset(i, j, a.row_stride, a.col_stride)]
                + b.data[get_offset(i, j, b.row_stride, b.col_stride)]
            )
    return Matrix[dtype](
        data=data^,
        nrows=a.nrows,
        ncols=a.ncols,
        row_stride=a.ncols,
        col_stride=1,
    )


fn sub[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises ValueError -> Matrix[dtype]:
    """Performs element-wise subtraction of two dynamic matrices.

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new matrix containing the element-wise difference of a and b.

    Raises:
        ValueError: If the shapes of a and b do not match.
    """
    if a.nrows != b.nrows or a.ncols != b.ncols:
        raise ValueError(
            file="src/matmojo/routines/math.mojo",
            function="sub()",
            message="Input matrices must have the same shape.",
            previous_error=None,
        )
    var data = List[Scalar[dtype]](unsafe_uninit_length=a.nrows * a.ncols)
    for i in range(a.nrows):
        for j in range(a.ncols):
            data[i * a.ncols + j] = (
                a.data[get_offset(i, j, a.row_stride, a.col_stride)]
                - b.data[get_offset(i, j, b.row_stride, b.col_stride)]
            )
    return Matrix[dtype](
        data=data^,
        nrows=a.nrows,
        ncols=a.ncols,
        row_stride=a.ncols,
        col_stride=1,
    )


fn mul[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises ValueError -> Matrix[dtype]:
    """Performs element-wise multiplication of two dynamic matrices.

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new matrix containing the element-wise product of a and b.

    Raises:
        ValueError: If the shapes of a and b do not match.
    """
    if a.nrows != b.nrows or a.ncols != b.ncols:
        raise ValueError(
            file="src/matmojo/routines/math.mojo",
            function="mul()",
            message="Input matrices must have the same shape.",
            previous_error=None,
        )
    var data = List[Scalar[dtype]](unsafe_uninit_length=a.nrows * a.ncols)
    for i in range(a.nrows):
        for j in range(a.ncols):
            data[i * a.ncols + j] = (
                a.data[get_offset(i, j, a.row_stride, a.col_stride)]
                * b.data[get_offset(i, j, b.row_stride, b.col_stride)]
            )
    return Matrix[dtype](
        data=data^,
        nrows=a.nrows,
        ncols=a.ncols,
        row_stride=a.ncols,
        col_stride=1,
    )


fn div[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises ValueError -> Matrix[dtype]:
    """Performs element-wise division of two dynamic matrices.

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new matrix containing the element-wise quotient of a and b.

    Raises:
        ValueError: If the shapes of a and b do not match.
    """
    if a.nrows != b.nrows or a.ncols != b.ncols:
        raise ValueError(
            file="src/matmojo/routines/math.mojo",
            function="div()",
            message="Input matrices must have the same shape.",
            previous_error=None,
        )
    var data = List[Scalar[dtype]](unsafe_uninit_length=a.nrows * a.ncols)
    for i in range(a.nrows):
        for j in range(a.ncols):
            data[i * a.ncols + j] = (
                a.data[get_offset(i, j, a.row_stride, a.col_stride)]
                / b.data[get_offset(i, j, b.row_stride, b.col_stride)]
            )
    return Matrix[dtype](
        data=data^,
        nrows=a.nrows,
        ncols=a.ncols,
        row_stride=a.ncols,
        col_stride=1,
    )


# ===---------------------------------------------------------------------- ===#
# Scalar–Matrix operations
# ===---------------------------------------------------------------------- ===#


fn scalar_add[
    dtype: DType
](mat: Matrix[dtype], scalar: Scalar[dtype]) -> Matrix[dtype]:
    """Adds a scalar to every element of a matrix.

    Args:
        mat: The input matrix.
        scalar: The scalar value to add.

    Returns:
        A new matrix with every element incremented by the scalar.
    """
    var data = List[Scalar[dtype]](unsafe_uninit_length=mat.nrows * mat.ncols)
    for i in range(mat.nrows):
        for j in range(mat.ncols):
            data[i * mat.ncols + j] = (
                mat.data[get_offset(i, j, mat.row_stride, mat.col_stride)]
                + scalar
            )
    return Matrix[dtype](
        data=data^,
        nrows=mat.nrows,
        ncols=mat.ncols,
        row_stride=mat.ncols,
        col_stride=1,
    )


fn scalar_sub[
    dtype: DType
](mat: Matrix[dtype], scalar: Scalar[dtype]) -> Matrix[dtype]:
    """Subtracts a scalar from every element of a matrix.

    Args:
        mat: The input matrix.
        scalar: The scalar value to subtract.

    Returns:
        A new matrix with every element decremented by the scalar.
    """
    var data = List[Scalar[dtype]](unsafe_uninit_length=mat.nrows * mat.ncols)
    for i in range(mat.nrows):
        for j in range(mat.ncols):
            data[i * mat.ncols + j] = (
                mat.data[get_offset(i, j, mat.row_stride, mat.col_stride)]
                - scalar
            )
    return Matrix[dtype](
        data=data^,
        nrows=mat.nrows,
        ncols=mat.ncols,
        row_stride=mat.ncols,
        col_stride=1,
    )


fn scalar_mul[
    dtype: DType
](mat: Matrix[dtype], scalar: Scalar[dtype]) -> Matrix[dtype]:
    """Multiplies every element of a matrix by a scalar.

    Args:
        mat: The input matrix.
        scalar: The scalar value to multiply.

    Returns:
        A new matrix with every element multiplied by the scalar.
    """
    var data = List[Scalar[dtype]](unsafe_uninit_length=mat.nrows * mat.ncols)
    for i in range(mat.nrows):
        for j in range(mat.ncols):
            data[i * mat.ncols + j] = (
                mat.data[get_offset(i, j, mat.row_stride, mat.col_stride)]
                * scalar
            )
    return Matrix[dtype](
        data=data^,
        nrows=mat.nrows,
        ncols=mat.ncols,
        row_stride=mat.ncols,
        col_stride=1,
    )


fn scalar_div[
    dtype: DType
](mat: Matrix[dtype], scalar: Scalar[dtype]) -> Matrix[dtype]:
    """Divides every element of a matrix by a scalar.

    Args:
        mat: The input matrix.
        scalar: The scalar value to divide by.

    Returns:
        A new matrix with every element divided by the scalar.
    """
    var data = List[Scalar[dtype]](unsafe_uninit_length=mat.nrows * mat.ncols)
    for i in range(mat.nrows):
        for j in range(mat.ncols):
            data[i * mat.ncols + j] = (
                mat.data[get_offset(i, j, mat.row_stride, mat.col_stride)]
                / scalar
            )
    return Matrix[dtype](
        data=data^,
        nrows=mat.nrows,
        ncols=mat.ncols,
        row_stride=mat.ncols,
        col_stride=1,
    )
