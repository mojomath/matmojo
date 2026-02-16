"""
This module defines the `Matrix` type.
"""

from matmojo.traits.storage_like import StorageLike
from matmojo.types.errors import IndexError, ValueError
from matmojo.types.dynamic_storage import DynamicStorage


struct Matrix[S: StorageLike = DynamicStorage](
    Copyable, Movable, Stringable, Writable
):
    """A 2D matrix type.
    A matrix owns its data and can write to it. The elements are stored in a
    contiguous block of memory in either row-major (C-contiguous) or
    column-major (Fortran-contiguous) order.
    """

    comptime ElementType = Float64
    """The type of the elements in the matrix, derived from the dtype."""

    var storage: Self.S
    """The underlying data buffer for the matrix elements."""

    # ===--------------------------------------------------------------------===#
    # Life Cycle Management
    # ===--------------------------------------------------------------------===#
    fn __init__(out self, var storage: Self.S):
        """Initializes an matrix with the given storage."""
        self.storage = storage^

    # ===--------------------------------------------------------------------===#
    # Element Access and Mutation
    # View Access
    # ===--------------------------------------------------------------------===#
    fn __getitem__(self, row: Int, col: Int) raises -> Self.ElementType:
        """Gets the element at the specified indices.

        Args:
            row: The row index.
            col: The column index.

        Raises:
            IndexError: If the indices are out of bounds.

        Returns:
            The element at the specified indices.
        """
        if row < 0 or row >= self.nrows() or col < 0 or col >= self.ncols():
            raise Error(
                IndexError(
                    file="src/matmojo/types/matrix.mojo",
                    function="Matrix.__getitem__(self, row: Int, col: Int)",
                    message="Index out of bounds.",
                    previous_error=None,
                )
            )
        return self.storage.load(row, col)

    fn __setitem__(
        mut self, row: Int, col: Int, value: Self.ElementType
    ) raises -> None:
        """Sets the element at the specified indices.

        Args:
            row: The row index.
            col: The column index.
            value: The value to set at the specified indices.

        Raises:
            IndexError: If the indices are out of bounds.
        """
        if row < 0 or row >= self.nrows() or col < 0 or col >= self.ncols():
            raise Error(
                IndexError(
                    file="src/matmojo/types/matrix.mojo",
                    function=(
                        "Matrix.__setitem__(self, row: Int, col: Int, value:"
                        " Self.ElementType)"
                    ),
                    message="Index out of bounds.",
                    previous_error=None,
                )
            )
        self.storage.store(row, col, value)

    # ===--------------------------------------------------------------------===#
    # Basic Matrix characteristics
    # ===--------------------------------------------------------------------===#
    @always_inline
    fn nrows(self) -> Int:
        """Returns the number of rows in the matrix."""
        return self.storage.nrows()

    @always_inline
    fn ncols(self) -> Int:
        """Returns the number of columns in the matrix."""
        return self.storage.ncols()

    @always_inline
    fn shape(self) -> Tuple[Int, Int]:
        """Returns the shape of the matrix as a tuple (rows, cols)."""
        return (self.nrows(), self.ncols())

    @always_inline
    fn row_stride(self) -> Int:
        """Returns the row stride of the matrix."""
        return self.storage.row_stride()

    @always_inline
    fn col_stride(self) -> Int:
        """Returns the column stride of the matrix."""
        return self.storage.col_stride()

    @always_inline
    fn strides(self) -> Tuple[Int, Int]:
        """Returns the strides of the matrix as a tuple (row_stride, col_stride).
        """
        return (self.row_stride(), self.col_stride())

    @always_inline
    fn size(self) -> Int:
        """Returns the total number of elements in the matrix."""
        return self.nrows() * self.ncols()

    # ===--------------------------------------------------------------------===#
    # String Representation and Writing
    # ===--------------------------------------------------------------------===#
    fn __str__(self) -> String:
        """Returns a string representation of the matrix."""
        result = String("")
        for i in range(self.nrows()):
            for j in range(self.ncols()):
                try:
                    result += String(self.storage.load(i, j)) + "\t"
                except e:
                    result += "Error: " + String(e) + "\t"
            if i < self.nrows() - 1:
                result += "\n"
        return result

    fn write_to[W: Writer](self, mut writer: W) -> None:
        """Writes the matrix to a writer."""
        try:
            writer.write("Matrix, ")
            writer.write("Float64")
            writer.write(", ")
            writer.write(self.nrows())
            writer.write("x")
            writer.write(self.ncols())
            writer.write(", strides: ")
            writer.write(self.row_stride())
            writer.write("-")
            writer.write(self.col_stride())
            writer.write(":\n")
            for i in range(self.nrows()):
                if i == 0:
                    writer.write("[[\t")
                else:
                    writer.write(" [\t")
                for j in range(self.ncols()):
                    writer.write(self.storage.load(i, j))
                    writer.write("\t")
                writer.write("]")
                if i < self.nrows() - 1:
                    writer.write("\n")
                else:
                    writer.write("]\n")
        except e:
            writer.write("Error writing matrix: " + String(e) + "\n")
