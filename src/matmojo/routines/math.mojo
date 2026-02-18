"""
Defines mathematical routines for matrices.
"""

from algorithm import vectorize, parallelize
from sys import simd_width_of

from matmojo.types.errors import ValueError
from matmojo.types.static_matrix import StaticMatrix
from matmojo.types.matrix import Matrix
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


fn matmul[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises ValueError -> Matrix[dtype]:
    """Performs matrix multiplication of two matrices.

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new matrix containing the product of a and b.
    """
    comptime simd_width_of_dtype = simd_width_of[dtype]()
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

    # a and b are both row-major
    var result = Matrix[dtype](a.nrows, b.ncols, b.ncols, 1)  # row-major

    @parameter
    fn closure_a_rows(row_a: Int):
        for inner in range(a.ncols):
            # [Mojo Miji]
            # Use each element in a certain row of a to mutiply the whole row
            # of b, and accumulate the results in the corresponding row of result.
            @parameter
            fn closure_b_cols[
                simd_width: Int
            ](col_b: Int) unified {
                mut result, read a, read b, read row_a, read inner
            }:
                result.data._data.store[width=simd_width](
                    row_a * result.row_stride + col_b * result.col_stride,
                    result.data._data.load[width=simd_width](
                        row_a * result.row_stride + col_b * result.col_stride
                    )
                    + a.data._data.load[width=1](
                        row_a * a.row_stride + inner * a.col_stride
                    )
                    * b.data._data.load[width=simd_width](
                        inner * b.row_stride + col_b * b.col_stride
                    ),
                )

            vectorize[simd_width_of_dtype](b.ncols, closure_b_cols)

    parallelize[closure_a_rows](a.nrows, a.nrows)

    return result^


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
