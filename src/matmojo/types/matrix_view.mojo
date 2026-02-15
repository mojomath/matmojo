"""
This module defines the `MatrixView` type.
"""

import math as builtin_math

from matmojo.traits.matrix_like import MatrixLike
from matmojo.types.matrix import Matrix
from matmojo.utils.indexing import get_offset
from memory import Pointer


struct MatrixView[mut: Bool, //, dtype: DType, origin: Origin[mut=mut]](
    MatrixLike, Stringable, Writable
):
    """A 2D matrix view type that references another Matrix.

    Parameters:
        mut: Whether the reference to the matrix is mutable.
        dtype: The data type of the matrix elements. Defaults to `DType.float64`.
        origin: The origin of the matrix.
    """

    comptime ElementType = Scalar[Self.dtype]
    """The type of the elements in the matrix view, derived from the dtype."""

    var src: Pointer[Matrix[Self.dtype], Self.origin]
    """A pointer to the base matrix that this view references."""
    var shape: Tuple[Int, Int]
    """The shape of the matrix view as a tuple (rows, cols)."""
    var strides: Tuple[Int, Int]
    """The strides of the matrix in memory (row stride, col stride)."""
    var offset: Int
    """The offset in the data where the matrix view starts."""
    var size: Int
    """The total number of elements in the matrix view."""

    fn __init__(
        out self,
        src: Pointer[Matrix[Self.dtype], Self.origin],
        *,
        shape: Tuple[Int, Int],
        strides: Tuple[Int, Int],
        offset: Int,
    ) raises:
        """Initializes a MatrixView instance that references a Matrix.

        Args:
            src: A pointer to the matrix to create a view of.
            shape: The shape of the view (rows, cols).
            strides: The strides for accessing elements.
            offset: The starting offset in the matrix data.
        """
        self.src = src
        self.shape = shape
        self.strides = strides
        self.offset = offset
        self.size = shape[0] * shape[1]

    fn __init__(
        out self,
        src: Pointer[Matrix[Self.dtype], Self.origin],
        *,
        slice_x: Slice,
        slice_y: Slice,
        initial_shape: Tuple[Int, Int],
        initial_strides: Tuple[Int, Int],
        initial_offset: Int,
    ) raises:
        """Initializes a MatrixView instance using slicing parameters."""
        self.src = src
        var start_x, end_x, step_x = slice_x.indices(initial_shape[0])
        var start_y, end_y, step_y = slice_y.indices(initial_shape[1])
        self.offset = (
            initial_offset
            + start_x * initial_strides[0]
            + start_y * initial_strides[1]
        )
        self.shape = (
            Int(builtin_math.ceildiv(end_x - start_x, step_x)),
            Int(builtin_math.ceildiv(end_y - start_y, step_y)),
        )
        self.strides = (
            initial_strides[0] * step_x,
            initial_strides[1] * step_y,
        )
        self.size = self.shape[0] * self.shape[1]

    # ===--------------------------------------------------------------------===#
    # Element Access and Mutation
    # ===--------------------------------------------------------------------===#

    fn __getitem__(self, x: Int, y: Int) -> Self.ElementType:
        """Accesses an element of the matrix view using row and column indices.
        """
        var index = get_offset((x, y), self.strides, self.offset)
        return self.src[].data[index]

    # [Mojo Miji]
    # The return type can also be written as:
    # `MatrixView[Self.dtype, Self.origin]`
    # It means that the view on view has the same data type and origin as the
    # original view.
    fn __getitem__(
        self, x: Slice, y: Slice
    ) raises -> MatrixView[Self.dtype, Self.origin]:
        """Gets a view of the specified row with a slice of columns."""
        return Self(
            src=self.src,
            slice_x=x,
            slice_y=y,
            initial_shape=self.shape,
            initial_strides=self.strides,
            initial_offset=self.offset,
        )

    # ===--------------------------------------------------------------------===#
    # String Representation and Writing
    # ===--------------------------------------------------------------------===#

    fn __str__(self) -> String:
        """Returns a string representation of the matrix."""
        result = String("")
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result += (
                    String(
                        self.src[].data[
                            get_offset((i, j), self.strides, self.offset)
                        ]
                    )
                    + "\t"
                )
            if i < self.shape[0] - 1:
                result += "\n"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        """Writes the matrix view to a writer."""
        for i in range(self.shape[0]):
            if i == 0:
                writer.write("[[\t")
            else:
                writer.write(" [\t")
            for j in range(self.shape[1]):
                writer.write(
                    self.src[].data[
                        get_offset((i, j), self.strides, self.offset)
                    ]
                )
                writer.write("\t")
            writer.write("]")
            if i < self.shape[0] - 1:
                writer.write("\n")
            else:
                writer.write("]")
