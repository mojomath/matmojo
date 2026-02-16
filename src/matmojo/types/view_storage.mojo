from matmojo.traits.storage_like import StorageLike
from matmojo.types.dynamic_storage import DynamicStorage


struct ViewStorage[mut: Bool, //, dtype: DType, origin: Origin[mut=mut]](
    ImplicitlyDestructible, Movable, StorageLike
):
    """A struct that represents view storage for a matrix.
    compile time.
    """

    var _src: Span[Float64, Self.origin]
    """A span to the base storage that this view references."""
    var _nrows: Int
    """The number of rows in the storage."""
    var _ncols: Int
    """The number of columns in the storage."""
    var _row_stride: Int
    """The row stride of the storage."""
    var _col_stride: Int
    """The column stride of the storage."""
    var _offset: Int
    """The offset in the data where the matrix view starts."""

    @always_inline
    fn nrows(self) -> Int:
        """Returns the number of rows in the storage."""
        return self._nrows

    @always_inline
    fn ncols(self) -> Int:
        """Returns the number of columns in the storage."""
        return self._ncols

    @always_inline
    fn row_stride(self) -> Int:
        """Returns the row stride of the storage."""
        return self._row_stride

    @always_inline
    fn col_stride(self) -> Int:
        """Returns the column stride of the storage."""
        return self._col_stride

    @always_inline
    fn load(self, row: Int, col: Int) -> Float64:
        """Loads the element at the given row and column."""
        var offset = (
            self._offset + row * self._row_stride + col * self._col_stride
        )
        return self._src[offset]

    @always_inline
    fn data(self) -> Span[Float64, Self.origin]:
        """Returns a span of the underlying data buffer.

        Warning: This provides direct, unsafe memory access.
        """
        return self._src
