"""Indexing and memory layout utilities for MatMojo."""


fn get_offset(
    indices: Tuple[Int, Int],
    strides: Tuple[Int, Int],
    initial_offset: Int = 0,
) -> Int:
    """Calculates the linear offset in a buffer for given indices.

    This function computes the position of an element in a 1D buffer
    based on its 2D indices, strides, and an optional initial offset.

    Args:
        indices: The 2D indices (row, col) of the element to access.
        strides: The strides for each dimension (row_stride, col_stride).
        initial_offset: An optional offset to add to the computed position.
            Defaults to 0. Useful for views or slices.

    Returns:
        The linear offset in the buffer where the element is located.

    Examples:
        # For a matrix with row-major layout (row_stride=4, col_stride=1):
        get_offset((2, 3), (4, 1))  # Returns 11 (2*4 + 3*1).

        # For a matrix with column-major layout (row_stride=1, col_stride=4):
        get_offset((2, 3), (1, 4))  # Returns 14 (2*1 + 3*4).
    """
    var row = indices[0]
    var col = indices[1]
    return initial_offset + row * strides[0] + col * strides[1]
