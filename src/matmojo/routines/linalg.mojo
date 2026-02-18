"""
Defines linear algebra routines for matrices.
"""

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
