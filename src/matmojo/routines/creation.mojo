"""
This module defines routines for creating matrices and matrix views in MatMojo.
"""

from matmojo.types.errors import ValueError
from matmojo.types.matrix import Matrix
from matmojo.types.matrix_view import MatrixView
from matmojo.types.static_matrix import StaticMatrix

# ===---------------------------------------------------------------------- ===#
# Matrix creation routines
# - Create matrices from nested lists.
# - Create matrices from flat lists with specified shapes.
# - Create static matrices from nested lists with compile-time dimensions.
# - Create static matrices from flat lists with compile-time dimensions.
# ===---------------------------------------------------------------------- ===#


fn matrix[
    dtype: DType
](
    list: List[List[Scalar[dtype]]], order: String = "C"
) raises ValueError -> Matrix[dtype]:
    """Initializes the matrix with a list of lists.

    Parameters:
        dtype: The data type of the matrix elements.

    Args:
        list: A list of lists where each inner list represents a row of the matrix.
        order: A string indicating the memory layout order. "C" for row-major
            and "F" for column-major. Defaults to "C".

    Raises:
        ValueError: If the input list is empty, if the rows have different
        lengths, or if the order is invalid.
    """

    if len(list) == 0:
        raise ValueError(
            file="src/matmojo/routines/creation.mojo",
            function="matrix()",
            message="Data cannot be an empty list.",
            previous_error=None,
        )

    nrows = len(list)
    ncols = len(list[0])

    if order == "C":
        row_stride = ncols
        col_stride = 1  # Row-major order
    elif order == "F":
        col_stride = nrows
        row_stride = 1  # Column-major order
    else:
        raise ValueError(
            file="src/matmojo/routines/creation.mojo",
            function="matrix()",
            message="Invalid order. Must be 'C' or 'F'.",
            previous_error=None,
        )

    for row in list:
        if len(row) != ncols:
            raise ValueError(
                file="src/matmojo/routines/creation.mojo",
                function="matrix()",
                message="All rows must have the same length.",
                previous_error=None,
            )

    var data = List[Scalar[dtype]](unsafe_uninit_length=nrows * ncols)
    var row_index = 0
    for row in list:
        var col_index = 0
        for element in row:
            data[row_index * row_stride + col_index * col_stride] = element
            col_index += 1
        row_index += 1
    return Matrix[dtype](
        data=data^,
        nrows=nrows,
        ncols=ncols,
        row_stride=row_stride,
        col_stride=col_stride,
    )


fn matrix[
    dtype: DType
](
    *,
    var flat_list: List[Scalar[dtype]],
    nrows: Int,
    ncols: Int,
    order: String = "C",
) raises ValueError -> Matrix[dtype]:
    """Initializes the matrix with a list and shape.

    Parameters:
        dtype: The data type of the matrix elements.

    Args:
        flat_list: A list of elements to initialize the matrix.
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        order: The memory layout order, either "C" for row-major or
            "F" for column-major. Defaults to "C".

    Raises:
        ValueError: If the length of the flat_list does not match the
        product of the shape dimensions.
    """
    if len(flat_list) == 0:
        raise ValueError(
            file="src/matmojo/routines/creation.mojo",
            function="matrix()",
            message="Data cannot be an empty list.",
            previous_error=None,
        )
    if len(flat_list) != nrows * ncols:
        raise ValueError(
            file="src/matmojo/routines/creation.mojo",
            function="matrix()",
            message="Data length does not match the specified shape.",
            previous_error=None,
        )
    if order == "C":
        row_stride = ncols  # Row-major order
        col_stride = 1
    elif order == "F":
        row_stride = 1  # Column-major order
        col_stride = nrows
    else:
        raise ValueError(
            file="src/matmojo/routines/creation.mojo",
            function="matrix()",
            message="Invalid order. Must be 'C' or 'F'.",
            previous_error=None,
        )
    return Matrix[dtype](
        data=flat_list^,
        nrows=nrows,
        ncols=ncols,
        row_stride=row_stride,
        col_stride=col_stride,
    )


fn matrix[
    nrows: Int, ncols: Int, dtype: DType = DType.float64
](var list: List[List[Scalar[dtype]]]) raises ValueError -> StaticMatrix[
    dtype, nrows, ncols
]:
    """Initializes the static matrix with a list of lists.

    Parameters:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        dtype: The data type of the matrix elements. Defaults to `DType.float64`.

    Args:
        list: A list of lists representing the rows of the matrix.
    """

    if len(list) != nrows:
        raise ValueError(
            file="src/matmojo/routines/creation.mojo",
            function="matrix()",
            message="Number of rows in list does not match nrows.",
            previous_error=None,
        )
    var result = StaticMatrix[dtype, nrows, ncols]()  # Initialize with zeros
    for row_index in range(len(list)):
        if len(list[row_index]) != ncols:
            raise ValueError(
                file="src/matmojo/routines/creation.mojo",
                function="matrix()",
                message="All rows must have the same length as ncols.",
                previous_error=None,
            )
        for col_index in range(ncols):
            result.data[
                row_index * result.row_stride + col_index * result.col_stride
            ] = list[row_index][col_index]
    return result^


fn matrix[
    nrows: Int, ncols: Int, dtype: DType = DType.float64
](*, var flat_list: List[Scalar[dtype]]) raises ValueError -> StaticMatrix[
    dtype, nrows, ncols
]:
    """Initializes the static matrix with a list of values.

    Parameters:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        dtype: The data type of the matrix elements.

    Args:
        flat_list: A list of values.
    """
    if len(flat_list) != nrows * ncols:
        raise ValueError(
            file="src/matmojo/routines/creation.mojo",
            function="matrix()",
            message="Number of rows in list does not match nrows.",
            previous_error=None,
        )
    var result = StaticMatrix[dtype, nrows, ncols]()  # Initialize with zeros
    var offset = 0
    for i in range(nrows):
        for j in range(ncols):
            result.data[
                i * result.row_stride + j * result.col_stride
            ] = flat_list[offset]
            offset += 1
    return result^
