trait MatrixLike(Copyable):
    """A trait for types that behave like matrices, providing a common interface
    for matrix operations.
    """

    fn get_data_ptr(
        self,
    ) -> UnsafePointer[List[Float64], Self.origin]:
        """Returns a pointer to the underlying data buffer of the matrix-like
        object."""
        ...

    fn get_nrows(self) -> Int:
        """Returns the number of rows in the matrix-like object."""
        ...

    fn get_ncols(self) -> Int:
        """Returns the number of columns in the matrix-like object."""
        ...

    fn get_row_stride(self) -> Int:
        """Returns the row stride of the matrix-like object."""
        ...

    fn get_col_stride(self) -> Int:
        """Returns the column stride of the matrix-like object."""
        ...

    fn get_offset(self) -> Int:
        """Returns the offset in the underlying data buffer for the matrix-like
        object."""
        ...

    fn get_size(self) -> Int:
        """Returns the total number of elements in the matrix-like object."""
        ...

    fn copy(self) -> Self:
        """Returns a copy of the matrix-like object."""
        ...

    fn __str__(self) -> String:
        """Returns a string representation of the matrix-like object."""
        ...

    # fn __getitem__(self, row: Int, col: Int) -> Scalar:
    #     """Returns the element at the specified row and column indices."""
    #     ...
