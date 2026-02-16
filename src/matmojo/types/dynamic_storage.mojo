from matmojo.types.errors import IndexError
from matmojo.traits.storage_like import StorageLike
from memory import Pointer


struct DynamicStorage(Copyable, ImplicitlyDestructible, Movable, StorageLike):
    """A struct that represents dynamic storage for a matrix. It is used to
    store the data of a matrix when the shape and strides are not known at
    compile time.
    """

    comptime ElementType = Float64
    """The type of the elements in the matrix, derived from the dtype."""

    var _data: List[Float64]
    """The underlying data buffer for the matrix elements."""
    var _nrows: Int
    """The number of rows in the storage."""
    var _ncols: Int
    """The number of columns in the storage."""
    var _row_stride: Int
    """The row stride of the storage."""
    var _col_stride: Int
    """The column stride of the storage."""

    fn __init__(
        out self,
        data: List[Float64],
        nrows: Int,
        ncols: Int,
        row_stride: Int,
        col_stride: Int,
    ):
        """Initializes a DynamicStorage instance with the given parameters."""
        self._data = data.copy()
        self._nrows = nrows
        self._ncols = ncols
        self._row_stride = row_stride
        self._col_stride = col_stride

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
    fn load(self, row: Int, col: Int) raises IndexError -> Float64:
        """Loads the element at the given row and column."""
        if (
            (row < 0)
            or (row >= self._nrows)
            or (col < 0)
            or (col >= self._ncols)
        ):
            raise IndexError(
                file="src/matmojo/types/dynamic_storage.mojo",
                function="DynamicStorage.load",
                message="Index out of bounds",
                previous_error=None,
            )
        offset = row * self._row_stride + col * self._col_stride
        return self._data[offset]

    @always_inline
    fn data(self) -> Span[Float64, origin_of(self._data)]:
        """Returns a span of the underlying data buffer.

        Warning: This provides direct, unsafe memory access.
        """
        return Span(self._data)
