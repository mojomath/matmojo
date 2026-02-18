"""
This module defines the `StaticMatrix` type which is a statically sized 2D matrix.
"""

from matmojo.traits.matrix_like import MatrixLike
from matmojo.types.errors import IndexError, ValueError
from matmojo.types.matrix_view import MatrixView
import matmojo.routines.math
from matmojo.utils.indexing import (
    indices_within_bounds,
)


fn next_power_of_two(x: Int) -> Int:
    """Returns the next power of two greater than or equal to x."""
    if x <= 1:
        return 1
    var v = x - 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v |= v >> 32
    return v + 1


struct StaticMatrix[dtype: DType, nrows: Int, ncols: Int](
    Copyable, MatrixLike, Stringable, Writable
):

    """A statically sized 2D matrix type.

    Parameters:
        dtype: The data type of the matrix elements.
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
    """

    comptime BUFFER_ROW_LEN = next_power_of_two(Self.nrows)
    comptime BUFFER_COL_LEN = next_power_of_two(Self.ncols)
    comptime BUFFER_SIZE = Self.BUFFER_ROW_LEN * Self.BUFFER_COL_LEN
    comptime row_stride = Self.BUFFER_COL_LEN
    comptime col_stride = 1

    comptime ElementType = Scalar[Self.dtype]
    """The type of the elements in the matrix, derived from the dtype."""

    var data: SIMD[Self.dtype, Self.BUFFER_SIZE]
    """A SIMD array representing the data of the matrix."""

    # ===--------------------------------------------------------------------===#
    # Retrieve attributes
    # ===--------------------------------------------------------------------===#
    fn get_nrows(self) -> Int:
        """Returns the number of rows in the matrix."""
        return self.nrows

    fn get_ncols(self) -> Int:
        """Returns the number of columns in the matrix."""
        return self.ncols

    fn get_row_stride(self) -> Int:
        """Returns the row stride of the matrix."""
        return self.row_stride

    fn get_col_stride(self) -> Int:
        """Returns the column stride of the matrix."""
        return self.col_stride

    fn get_offset(self) -> Int:
        """Returns the offset in the underlying data buffer for the matrix."""
        return 0

    fn get_size(self) -> Int:
        """Returns the total number of elements in the matrix."""
        return self.nrows * self.ncols

    # ===--------------------------------------------------------------------===#
    # Life Cycle Management
    # ===--------------------------------------------------------------------===#

    fn __init__(out self):
        """Initializes the matrix with all zeros."""
        # [Mojo Miji]
        # SIMD() initializes the buffer with zeros at compile time, so we don't
        # need to explicitly fill it with zeros here.
        self.data = SIMD[Self.dtype, Self.BUFFER_SIZE]()

    fn __init__(out self, simd: SIMD[Self.dtype, Self.BUFFER_SIZE]):
        """Initializes the matrix with SIMD that match the size of buffer."""
        self.data = simd

    fn __copyinit__(out self, other: Self):
        """Initializes the matrix by copying another matrix."""
        self.data = other.data

    # ===--------------------------------------------------------------------===#
    # Element Access and Mutation
    # ===--------------------------------------------------------------------===#

    fn __getitem__(self, row: Int, col: Int) -> Self.ElementType:
        """Accesses an element of the matrix view using row and column indices.
        """
        return self.data[row * self.row_stride + col * self.col_stride]

    # ===--------------------------------------------------------------------===#
    # String Representation and Writing
    # ===--------------------------------------------------------------------===#

    fn __str__(self) -> String:
        """Returns a string representation of the matrix."""
        result = String("")
        for i in range(self.nrows):
            for j in range(self.ncols):
                result += (
                    String(self.data[i * self.row_stride + j * self.col_stride])
                    + "\t"
                )
            if i < self.nrows - 1:
                result += "\n"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        """Writes the matrix to a writer."""
        writer.write("StaticMatrix, ")
        writer.write(self.dtype)
        writer.write(", ")
        writer.write(self.nrows)
        writer.write("x")
        writer.write(self.ncols)
        writer.write(":\n")
        for i in range(self.nrows):
            if i == 0:
                writer.write("[[\t")
            else:
                writer.write(" [\t")
            for j in range(self.ncols):
                writer.write(
                    round(
                        self.data[i * self.row_stride + j * self.col_stride], 4
                    )
                )
                writer.write("\t")
            writer.write("]")
            if i < self.nrows - 1:
                writer.write("\n")
            else:
                writer.write("]")

    # ===------------------------------------------------------------------ ===#
    # Basic math dunders
    # ===------------------------------------------------------------------ ===#

    fn __add__(self, other: Self) -> Self:
        """Performs element-wise addition of two matrices."""
        return matmojo.routines.math.add(self, other)

    fn __matmul__[
        other_ncols: Int
    ](
        self, other: StaticMatrix[Self.dtype, Self.ncols, other_ncols]
    ) -> StaticMatrix[Self.dtype, Self.nrows, other_ncols]:
        """Performs matrix multiplication of two matrices."""
        return matmojo.routines.math.matmul(self, other)
