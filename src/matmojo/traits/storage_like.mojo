from matmojo.types.errors import IndexError


trait StorageLike(Copyable, ImplicitlyDestructible, Movable):
    fn nrows(self) -> Int:
        """Returns the number of rows in the storage."""
        ...

    fn ncols(self) -> Int:
        """Returns the number of columns in the storage."""
        ...

    fn row_stride(self) -> Int:
        """Returns the row stride of the storage."""
        ...

    fn col_stride(self) -> Int:
        """Returns the column stride of the storage."""
        ...

    fn offset(self) -> Int:
        """Returns the offset of the storage."""
        ...

    fn load(self, row: Int, col: Int) raises IndexError -> Float64:
        """Loads the element at the given row and column."""
        ...

    fn store(
        mut self, row: Int, col: Int, value: Float64
    ) raises IndexError -> None:
        """Stores the given value at the specified row and column."""
        ...

    fn unsafe_load(self, row: Int, col: Int) -> Float64:
        """Loads the element at the given row and column without bounds checking.
        """
        ...

    fn unsafe_store(mut self, row: Int, col: Int, value: Float64) -> None:
        """Stores the given value at the specified row and column without bounds
        checking.
        """
        ...

    fn type_as_string(self) -> String:
        """Returns a string representation of the type of this storage."""
        ...
