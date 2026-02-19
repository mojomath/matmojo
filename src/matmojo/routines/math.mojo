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

    Dispatches to one of four SIMD-optimised paths based on the memory layout
    of the two operands, falling back to a general stride-aware loop when
    neither operand has contiguous rows or columns.

    **Path 1 - B row-contiguous** (covers R×R, F×R, any×R):
        Vectorize across B's row (SIMD load), parallelize over A's rows.
        A is accessed element-wise with the generic offset formula.

    **Path 2 - C×F** (A row-contiguous, B column-contiguous):
        Each result element is a dot-product of a contiguous A-row and a
        contiguous B-column.  SIMD vectorize the K loop with `reduce_add`.

    **Path 3 - A column-contiguous** (covers F×F, F×weird):
        Vectorize across A's column (SIMD load), accumulate in a temporary
        contiguous column buffer, then scatter to the C-contiguous result.
        Parallelize over B's columns.

    **Path 4 - General fallback**:
        Stride-aware scalar triple loop, parallelize over rows.

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

    # Shared pointer setup – used by all SIMD paths.
    var a_ptr = a.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()
    var a_off = a.offset
    var b_off = b.offset
    var a_rs = a.row_stride
    var a_cs = a.col_stride
    var b_rs = b.row_stride
    var b_cs = b.col_stride

    # ---------------------------------------------------------------------- #
    # Path 1: B row-contiguous (col_stride == 1)
    # Covers R×R, F×R, and any layout where B's rows are contiguous.
    # Vectorize over B's columns + parallelize over A's rows.
    # ---------------------------------------------------------------------- #
    if b.is_row_contiguous():

        @parameter
        fn process_row_br(i: Int):
            for k in range(K):
                # [Mojo Miji]
                # Broadcast A[i,k] (scalar) and SIMD-multiply with row k of B,
                # accumulating into row i of result.
                @parameter
                fn vec_col_br[
                    w: Int
                ](j: Int) unified {
                    mut result,
                    read a_ptr,
                    read b_ptr,
                    read a_off,
                    read b_off,
                    read a_rs,
                    read a_cs,
                    read b_rs,
                    read N,
                    read i,
                    read k,
                }:
                    var r_idx = i * N + j
                    result.data._data.store[width=w](
                        r_idx,
                        result.data._data.load[width=w](r_idx)
                        + a_ptr.load[width=1](a_off + i * a_rs + k * a_cs)
                        * b_ptr.load[width=w](b_off + k * b_rs + j),
                    )

                vectorize[simd_w](N, vec_col_br)

        parallelize[process_row_br](M, M)

    # ---------------------------------------------------------------------- #
    # Path 2: A row-contiguous, B column-contiguous  (C × F)
    # Both A's row and B's column are contiguous over the K dimension,
    # so each result element is a SIMD dot-product with reduce_add.
    # ---------------------------------------------------------------------- #
    elif a.is_row_contiguous() and b.is_col_contiguous():

        @parameter
        fn process_row_cxf(i: Int):
            for j in range(N):
                var dot_sum: Scalar[dtype] = 0

                @parameter
                fn dot_k[
                    w: Int
                ](k: Int) unified {
                    mut dot_sum,
                    read a_ptr,
                    read b_ptr,
                    read a_off,
                    read b_off,
                    read a_rs,
                    read b_cs,
                    read i,
                    read j,
                }:
                    dot_sum += (
                        a_ptr.load[width=w](a_off + i * a_rs + k)
                        * b_ptr.load[width=w](b_off + j * b_cs + k)
                    ).reduce_add()

                vectorize[simd_w](K, dot_k)
                result.data._data.store[width=1](i * N + j, dot_sum)

        parallelize[process_row_cxf](M, M)

    # ---------------------------------------------------------------------- #
    # Path 3: A column-contiguous  (covers F × F, F × weird)
    # A's columns are contiguous → vectorize over rows of A.
    # Accumulate in a temporary contiguous column buffer, then scatter to
    # the C-contiguous result.  Parallelize over B's columns.
    # ---------------------------------------------------------------------- #
    elif a.is_col_contiguous():

        @parameter
        fn process_col_ff(j: Int):
            # Temporary column buffer for SIMD accumulation.
            var temp = List[Scalar[dtype]](length=M, fill=0)
            var temp_ptr = temp._data

            for k in range(K):
                var b_kj = b_ptr.load[width=1](b_off + k * b_rs + j * b_cs)

                @parameter
                fn vec_rows_ff[
                    w: Int
                ](i: Int) unified {
                    read temp_ptr,
                    read a_ptr,
                    read a_off,
                    read a_cs,
                    read b_kj,
                    read k,
                }:
                    temp_ptr.store[width=w](
                        i,
                        temp_ptr.load[width=w](i)
                        + a_ptr.load[width=w](a_off + k * a_cs + i) * b_kj,
                    )

                vectorize[simd_w](M, vec_rows_ff)

            # Scatter temp column into result's column j.
            for i in range(M):
                result.data._data.store[width=1](
                    i * N + j, temp_ptr.load[width=1](i)
                )

        parallelize[process_col_ff](N, N)

    # ---------------------------------------------------------------------- #
    # Path 4: General fallback – any memory layout
    # ---------------------------------------------------------------------- #
    else:

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
# Element-wise operation primitives
# ===---------------------------------------------------------------------- ===#
# [Mojo Miji]
# We define a single, generic `_elementwise_view` function that takes a
# `func: fn(Scalar, Scalar) -> Scalar` compile-time parameter. This avoids
# duplicating near-identical code for add, sub, mul, div. Each public function
# is a thin wrapper that plugs in the right SIMD dunder method directly
# (e.g. `Scalar[dtype].__add__`).
#
# For C-contiguous operands, a SIMD-vectorised fast path is used (SIMD
# loads/stores with per-lane func application).  For non-contiguous views the
# fallback is a stride-aware double loop.
#
# Four overloads are provided for each binary operation so that any combination
# of Matrix / MatrixView works directly, mirroring the matmul design.
#
# The same pattern is applied to scalar–matrix operations via
# `_scalar_elementwise_view`.

# --------------------------------------------------------------------------- #
# Core view-based element-wise implementation
# --------------------------------------------------------------------------- #


fn _elementwise_view[
    dtype: DType,
    func: fn(Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
    origin_a: Origin,
    origin_b: Origin,
](
    a: MatrixView[dtype, origin_a],
    b: MatrixView[dtype, origin_b],
) raises ValueError -> Matrix[dtype]:
    """Core element-wise binary operation on two MatrixView operands.

    When both operands are C-contiguous, a SIMD-vectorised fast path is
    taken (linear memory traversal with vectorize).  Otherwise, a stride-aware
    double loop is used.

    The result is always a freshly allocated, C-contiguous Matrix.
    """
    if a.nrows != b.nrows or a.ncols != b.ncols:
        raise ValueError(
            file="src/matmojo/routines/math.mojo",
            function="_elementwise_view()",
            message="Input matrices must have the same shape.",
            previous_error=None,
        )
    var M = a.nrows
    var N = a.ncols
    var total = M * N
    var result = Matrix[dtype](M, N, N, 1)

    if a.is_c_contiguous() and b.is_c_contiguous():
        # Fast SIMD path: both operands are dense row-major, so we can do a
        # single linear pass with vectorized loads/stores.
        comptime simd_w = simd_width_of[dtype]()
        var a_ptr = a.data.unsafe_ptr()
        var b_ptr = b.data.unsafe_ptr()
        var a_off = a.offset
        var b_off = b.offset

        @parameter
        fn vec_op[
            w: Int
        ](idx: Int) unified {
            mut result,
            read a_ptr,
            read b_ptr,
            read a_off,
            read b_off,
        }:
            var a_chunk = a_ptr.load[width=w](a_off + idx)
            var b_chunk = b_ptr.load[width=w](b_off + idx)
            var res = SIMD[dtype, w](0)

            @parameter
            for lane in range(w):
                res[lane] = func(a_chunk[lane], b_chunk[lane])
            result.data._data.store[width=w](idx, res)

        vectorize[simd_w](total, vec_op)
    else:
        # General stride-aware fallback.
        for i in range(M):
            for j in range(N):
                result.data[i * N + j] = func(a[i, j], b[i, j])

    return result^


# --------------------------------------------------------------------------- #
# Core view-based scalar element-wise implementation
# --------------------------------------------------------------------------- #


fn _scalar_elementwise_view[
    dtype: DType,
    func: fn(Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
    origin: Origin,
](mat: MatrixView[dtype, origin], scalar: Scalar[dtype],) -> Matrix[dtype]:
    """Core scalar–matrix element-wise operation on a MatrixView operand.

    When the operand is C-contiguous, a SIMD-vectorised fast path is taken.
    Otherwise, a stride-aware double loop is used.

    The result is always a freshly allocated, C-contiguous Matrix.
    """
    var M = mat.nrows
    var N = mat.ncols
    var total = M * N
    var result = Matrix[dtype](M, N, N, 1)

    if mat.is_c_contiguous():
        comptime simd_w = simd_width_of[dtype]()
        var m_ptr = mat.data.unsafe_ptr()
        var m_off = mat.offset

        @parameter
        fn vec_scalar[
            w: Int
        ](idx: Int) unified {mut result, read m_ptr, read m_off, read scalar,}:
            var m_chunk = m_ptr.load[width=w](m_off + idx)
            var s_chunk = SIMD[dtype, w](scalar)
            var res = SIMD[dtype, w](0)

            @parameter
            for lane in range(w):
                res[lane] = func(m_chunk[lane], s_chunk[lane])
            result.data._data.store[width=w](idx, res)

        vectorize[simd_w](total, vec_scalar)
    else:
        for i in range(M):
            for j in range(N):
                result.data[i * N + j] = func(mat[i, j], scalar)

    return result^


# ===---------------------------------------------------------------------- ===#
# Dynamic element-wise operations: add, sub, mul, div
# ===---------------------------------------------------------------------- ===#
# Each operation has 4 overloads: view×view, mat×mat, mat×view, view×mat.
# All delegate to `_elementwise_view` with the appropriate `_op` helper.


# --------------------------------------------------------------------------- #
# add
# --------------------------------------------------------------------------- #


fn add[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[dtype, origin_a], b: MatrixView[dtype, origin_b]
) raises ValueError -> Matrix[dtype]:
    """Element-wise addition of two matrix views."""
    return _elementwise_view[func = Scalar[dtype].__add__](a, b)


fn add[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises ValueError -> Matrix[dtype]:
    """Element-wise addition of two dynamic matrices."""
    return _elementwise_view[func = Scalar[dtype].__add__](a.view(), b.view())


fn add[
    dtype: DType, origin_b: Origin
](a: Matrix[dtype], b: MatrixView[dtype, origin_b]) raises ValueError -> Matrix[
    dtype
]:
    """Element-wise addition of a matrix and a matrix view."""
    return _elementwise_view[func = Scalar[dtype].__add__](a.view(), b)


fn add[
    dtype: DType, origin_a: Origin
](a: MatrixView[dtype, origin_a], b: Matrix[dtype]) raises ValueError -> Matrix[
    dtype
]:
    """Element-wise addition of a matrix view and a matrix."""
    return _elementwise_view[func = Scalar[dtype].__add__](a, b.view())


# --------------------------------------------------------------------------- #
# sub
# --------------------------------------------------------------------------- #


fn sub[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[dtype, origin_a], b: MatrixView[dtype, origin_b]
) raises ValueError -> Matrix[dtype]:
    """Element-wise subtraction of two matrix views."""
    return _elementwise_view[func = Scalar[dtype].__sub__](a, b)


fn sub[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises ValueError -> Matrix[dtype]:
    """Element-wise subtraction of two dynamic matrices."""
    return _elementwise_view[func = Scalar[dtype].__sub__](a.view(), b.view())


fn sub[
    dtype: DType, origin_b: Origin
](a: Matrix[dtype], b: MatrixView[dtype, origin_b]) raises ValueError -> Matrix[
    dtype
]:
    """Element-wise subtraction of a matrix and a matrix view."""
    return _elementwise_view[func = Scalar[dtype].__sub__](a.view(), b)


fn sub[
    dtype: DType, origin_a: Origin
](a: MatrixView[dtype, origin_a], b: Matrix[dtype]) raises ValueError -> Matrix[
    dtype
]:
    """Element-wise subtraction of a matrix view and a matrix."""
    return _elementwise_view[func = Scalar[dtype].__sub__](a, b.view())


# --------------------------------------------------------------------------- #
# mul
# --------------------------------------------------------------------------- #


fn mul[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[dtype, origin_a], b: MatrixView[dtype, origin_b]
) raises ValueError -> Matrix[dtype]:
    """Element-wise multiplication of two matrix views."""
    return _elementwise_view[func = Scalar[dtype].__mul__](a, b)


fn mul[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises ValueError -> Matrix[dtype]:
    """Element-wise multiplication of two dynamic matrices."""
    return _elementwise_view[func = Scalar[dtype].__mul__](a.view(), b.view())


fn mul[
    dtype: DType, origin_b: Origin
](a: Matrix[dtype], b: MatrixView[dtype, origin_b]) raises ValueError -> Matrix[
    dtype
]:
    """Element-wise multiplication of a matrix and a matrix view."""
    return _elementwise_view[func = Scalar[dtype].__mul__](a.view(), b)


fn mul[
    dtype: DType, origin_a: Origin
](a: MatrixView[dtype, origin_a], b: Matrix[dtype]) raises ValueError -> Matrix[
    dtype
]:
    """Element-wise multiplication of a matrix view and a matrix."""
    return _elementwise_view[func = Scalar[dtype].__mul__](a, b.view())


# --------------------------------------------------------------------------- #
# div
# --------------------------------------------------------------------------- #


fn div[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[dtype, origin_a], b: MatrixView[dtype, origin_b]
) raises ValueError -> Matrix[dtype]:
    """Element-wise division of two matrix views."""
    return _elementwise_view[func = Scalar[dtype].__truediv__](a, b)


fn div[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises ValueError -> Matrix[dtype]:
    """Element-wise division of two dynamic matrices."""
    return _elementwise_view[func = Scalar[dtype].__truediv__](
        a.view(), b.view()
    )


fn div[
    dtype: DType, origin_b: Origin
](a: Matrix[dtype], b: MatrixView[dtype, origin_b]) raises ValueError -> Matrix[
    dtype
]:
    """Element-wise division of a matrix and a matrix view."""
    return _elementwise_view[func = Scalar[dtype].__truediv__](a.view(), b)


fn div[
    dtype: DType, origin_a: Origin
](a: MatrixView[dtype, origin_a], b: Matrix[dtype]) raises ValueError -> Matrix[
    dtype
]:
    """Element-wise division of a matrix view and a matrix."""
    return _elementwise_view[func = Scalar[dtype].__truediv__](a, b.view())


# ===---------------------------------------------------------------------- ===#
# Scalar–Matrix operations
# ===---------------------------------------------------------------------- ===#
# Each scalar operation has 2 overloads: Matrix and MatrixView.


fn scalar_add[
    dtype: DType
](mat: Matrix[dtype], scalar: Scalar[dtype]) -> Matrix[dtype]:
    """Adds a scalar to every element of a matrix."""
    return _scalar_elementwise_view[func = Scalar[dtype].__add__](
        mat.view(), scalar
    )


fn scalar_add[
    dtype: DType, origin: Origin
](mat: MatrixView[dtype, origin], scalar: Scalar[dtype]) -> Matrix[dtype]:
    """Adds a scalar to every element of a matrix view."""
    return _scalar_elementwise_view[func = Scalar[dtype].__add__](mat, scalar)


fn scalar_sub[
    dtype: DType
](mat: Matrix[dtype], scalar: Scalar[dtype]) -> Matrix[dtype]:
    """Subtracts a scalar from every element of a matrix."""
    return _scalar_elementwise_view[func = Scalar[dtype].__sub__](
        mat.view(), scalar
    )


fn scalar_sub[
    dtype: DType, origin: Origin
](mat: MatrixView[dtype, origin], scalar: Scalar[dtype]) -> Matrix[dtype]:
    """Subtracts a scalar from every element of a matrix view."""
    return _scalar_elementwise_view[func = Scalar[dtype].__sub__](mat, scalar)


fn scalar_mul[
    dtype: DType
](mat: Matrix[dtype], scalar: Scalar[dtype]) -> Matrix[dtype]:
    """Multiplies every element of a matrix by a scalar."""
    return _scalar_elementwise_view[func = Scalar[dtype].__mul__](
        mat.view(), scalar
    )


fn scalar_mul[
    dtype: DType, origin: Origin
](mat: MatrixView[dtype, origin], scalar: Scalar[dtype]) -> Matrix[dtype]:
    """Multiplies every element of a matrix view by a scalar."""
    return _scalar_elementwise_view[func = Scalar[dtype].__mul__](mat, scalar)


fn scalar_div[
    dtype: DType
](mat: Matrix[dtype], scalar: Scalar[dtype]) -> Matrix[dtype]:
    """Divides every element of a matrix by a scalar."""
    return _scalar_elementwise_view[func = Scalar[dtype].__truediv__](
        mat.view(), scalar
    )


fn scalar_div[
    dtype: DType, origin: Origin
](mat: MatrixView[dtype, origin], scalar: Scalar[dtype]) -> Matrix[dtype]:
    """Divides every element of a matrix view by a scalar."""
    return _scalar_elementwise_view[func = Scalar[dtype].__truediv__](
        mat, scalar
    )
