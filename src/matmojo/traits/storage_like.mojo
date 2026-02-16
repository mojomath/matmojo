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

    fn load(self, row: Int, col: Int) raises IndexError -> Float64:
        """Loads the element at the given row and column."""
        ...
