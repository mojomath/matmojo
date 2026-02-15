struct MatrixView[dtype: DType = DType.float64]:
    """A 2D matrix view type.

    Parameters:
        dtype: The data type of the matrix elements. Defaults to `DType.float64`.
    """

    var shape: Tuple[Int, Int]
    """The shape of the matrix as a tuple (rows, cols)."""
    var strides: Tuple[Int, Int]
    """The strides of the matrix in memory (row stride, col stride)."""
    var offset: Int
    """The offset in the data list where the matrix view starts."""
    var size: Int
    """The total number of elements in the matrix."""
