from matmojo.types.errors import IndexError
from matmojo.traits.storage_like import StorageLike
from matmojo.types.dynamic_storage import DynamicStorage
from matmojo.utils.indexing import indices_out_of_bounds, get_offset


struct ViewStorage[mut: Bool, //, origin: Origin[mut=mut]](
    ImplicitlyDestructible, Movable, StorageLike
):
    """A struct that represents view storage for a matrix.
    compile time.
    """

    comptime ElementType = Float64

    var _data: Span[Float64, Self.origin]
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

    fn __init__(
        out self,
        data: Span[Float64, Self.origin],
        nrows: Int,
        ncols: Int,
        row_stride: Int,
        col_stride: Int,
        offset: Int,
    ):
        self._data = data
        self._nrows = nrows
        self._ncols = ncols
        self._row_stride = row_stride
        self._col_stride = col_stride
        self._offset = offset

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
    fn offset(self) -> Int:
        """Returns the offset of the storage."""
        return self._offset

    @always_inline
    fn load(self, row: Int, col: Int) raises IndexError -> Float64:
        """Loads the element at the given row and column."""
        if indices_out_of_bounds(row, col, self._nrows, self._ncols):
            raise IndexError(
                file="src/matmojo/types/view_storage.mojo",
                function="ViewStorage.load",
                message="Index out {} of bounds {}".format(
                    (row, col), (self._nrows, self._ncols)
                ),
                previous_error=None,
            )
        var offset = get_offset(
            row, col, self._row_stride, self._col_stride, self._offset
        )
        return self._data[offset]

    fn store(
        mut self, row: Int, col: Int, value: Float64
    ) raises IndexError -> None:
        """Stores the given value at the specified row and column."""
        if indices_out_of_bounds(row, col, self._nrows, self._ncols):
            raise IndexError(
                file="src/matmojo/types/view_storage.mojo",
                function="ViewStorage.store",
                message="Index out {} of bounds {}".format(
                    (row, col), (self._nrows, self._ncols)
                ),
                previous_error=None,
            )
        var offset = get_offset(
            row, col, self._row_stride, self._col_stride, self._offset
        )

        # TODO: Need to find out the syntax that allows us to set up branches
        # at compile time for whether this view is mutable or not, and allows
        # us to take different actions.
        comptime assert self.mut, "Cannot store to an immutable view."
        raise IndexError(
            file="src/matmojo/types/view_storage.mojo",
            function="ViewStorage.store",
            message="MatMojo currently does not support storing to views.",
            previous_error=None,
        )

    fn unsafe_load(self, row: Int, col: Int) -> Float64:
        """Loads the element at the given row and column without bounds checking.
        """
        debug_assert(
            indices_out_of_bounds(row, col, self._nrows, self._ncols) == False,
            "Debug assertion failed: Indices out of bounds in `unsafe_load`",
        )
        var offset = get_offset(
            row, col, self._row_stride, self._col_stride, self._offset
        )
        return self._data[offset]

    fn unsafe_store(mut self, row: Int, col: Int, value: Float64) -> None:
        """Stores the given value at the specified row and column without bounds
        checking.
        """
        debug_assert(
            indices_out_of_bounds(row, col, self._nrows, self._ncols) == False,
            "Debug assertion failed: Indices out of bounds in `unsafe_store`",
        )
        var offset = get_offset(
            row, col, self._row_stride, self._col_stride, self._offset
        )
        # TODO: Need to find out the syntax that allows us to set up branches
        # at compile time for whether this view is mutable or not, and allows
        # us to take different actions.

    @always_inline
    fn data(self) -> Span[Float64, Self.origin]:
        """Returns a span of the underlying data buffer.

        Warning: This provides direct, unsafe memory access.
        """
        return self._data

    fn type_as_string(self) -> String:
        """Returns a string representation of the type of this storage."""
        return (
            "ViewStorage[dtype={}, nrows={}, ncols={}, row_stride={},"
            " col_stride={}, offset={}]".format(
                Self.ElementType.dtype,
                self._nrows,
                self._ncols,
                self._row_stride,
                self._col_stride,
                self._offset,
            )
        )
