"""
This module defines routines for creating matrices and matrix views in MatMojo.
"""

from matmojo.traits.storage_like import StorageLike
from matmojo.types.errors import IndexError, ValueError
from matmojo.types.matrix import Matrix
from matmojo.types.dynamic_storage import DynamicStorage


fn matrix(
    list: List[List[Float64]], order: String
) raises -> Matrix[DynamicStorage]:
    if len(list) == 0:
        raise Error(
            ValueError(
                file="src/matmojo/routines/creation.mojo",
                function="matrix()",
                message="Data cannot be an empty list.",
                previous_error=None,
            )
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
        raise Error(
            ValueError(
                file="src/matmojo/routines/creation.mojo",
                function="matrix()",
                message="Invalid order. Must be 'C' or 'F'.",
                previous_error=None,
            )
        )

    for row in list:
        if len(row) != ncols:
            raise Error(
                ValueError(
                    file="src/matmojo/routines/creation.mojo",
                    function="matrix()",
                    message="All rows must have the same length.",
                    previous_error=None,
                )
            )

    var data = List[Float64](unsafe_uninit_length=nrows * ncols)
    var row_index = 0
    for row in list:
        var col_index = 0
        for element in row:
            data[row_index * row_stride + col_index * col_stride] = element
            col_index += 1
        row_index += 1
    return Matrix[DynamicStorage](
        storage=DynamicStorage(
            data=data,
            nrows=nrows,
            ncols=ncols,
            row_stride=row_stride,
            col_stride=col_stride,
        )
    )
