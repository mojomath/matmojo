"""
This module defines the `Matrix` type.
"""

from matmojo.traits.matrix_like import MatrixLike
from matmojo.types.errors import IndexError, ValueError
from matmojo.types.matrix_view import MatrixView
from matmojo.utils.indexing import get_offset


struct Matrix[dtype: DType = DType.float64](
    Copyable, MatrixLike, Movable, Stringable, Writable
):
    """A 2D matrix type.
    A matrix owns its data and can write to it. The elements are stored in a
    contiguous block of memory in either row-major (C-contiguous) or
    column-major (Fortran-contiguous) order.

    Parameters:
        dtype: The data type of the matrix elements. Defaults to `DType.float64`.
    """

    # [Mojo Miji]
    # `comptime` can be used to define a type alias that can be translated back
    # to the original type at compile time. We do this for convenience.
    comptime ElementType = Scalar[Self.dtype]
    """The type of the elements in the matrix, derived from the dtype."""

    # [Mojo Miji]
    # If we want to implement a simple 2D matrix type,
    # the following three attributes are essential:
    # - data: A contiguous block of memory that holds the elements of the matrix.
    # - shape: A tuple that specifies the dimensions of the matrix (rows, cols).
    # - strides: A tuple that specifies the number of bytes to step in each dimension.
    # The size attribute can be derived from the shape (size = rows * cols) and
    # is not strictly necessary to store separately, but it can be convenient for
    # quick access.
    #
    # About the "data" attribute:
    # We use a single list to store the elements of the matrix in a contiguous
    # block of memory. This is a "safe" way to manage memory in Mojo, as it
    # avoids the complexities of manual memory management while still providing
    # efficient access to the elements. It is also aligned with our philosophy
    # of "using safe Mojo as much as possible".
    # The disadvantage of this approach is that you cannot easily design a
    # shared-memory model where multiple matrices share the same underlying data
    # without defining different data types. In MatMojo, we have to define both
    # a "Matrix" type that owns its data and a "MatrixView" type that references
    # the data of another matrix. Thanks to the generic programming capabilities
    # of Mojo, we can still achieve a high level of code reuse between these
    # types.
    #
    # About the "shape" and "strides" attributes:
    # We use the tuple (rows, cols) to represent the shape of the matrix, which
    # is an efficient way to store the dimensions. For n-D arrays, we have to
    # use the list type to store the shape because the dimension is not fixed at
    # compile time. This also applies to the strides.
    #
    # CORE ATTRIBUTES
    var data: List[Scalar[Self.dtype]]
    """The elements of the matrix stored in a contiguous block of memory."""
    var shape: Tuple[Int, Int]
    """The shape of the matrix as a tuple (rows, cols)."""
    var strides: Tuple[Int, Int]
    """The strides of the matrix in memory (row stride, col stride)."""
    # DERIVED ATTRIBUTE
    var size: Int
    """The total number of elements in the matrix."""

    # ===--------------------------------------------------------------------===#
    # Life Cycle Management
    # ===--------------------------------------------------------------------===#

    fn __init__(
        out self,
        data: List[List[Self.ElementType]],
        order: String = "C",
    ) raises:
        """Initializes the matrix with a nested list and memory layout order.

        Args:
            data: A nested list of elements to initialize the matrix.
            order: The memory layout order, either "C" for row-major or
                "F" for column-major. Defaults to "C".

        Raises:
            ValueError: If the length of the data list does not match the
            product of the shape dimensions.
        """
        if len(data) == 0:
            raise Error(
                ValueError(
                    file="src/matmojo/types/matrix.mojo",
                    function=(
                        "Matrix.__init__(out self, var data:"
                        " List[List[ElementType]], order: String)"
                    ),
                    message="Data cannot be an empty list.",
                    previous_error=None,
                )
            )
        self.shape = (len(data), len(data[0]))
        self.size = self.shape[0] * self.shape[1]
        if order == "C":
            self.strides = (self.shape[1], 1)  # Row-major order
        elif order == "F":
            self.strides = (1, self.shape[0])  # Column-major order
        else:
            raise Error(
                ValueError(
                    file="src/matmojo/types/matrix.mojo",
                    function=(
                        "Matrix.__init__(out self, var data:"
                        " List[List[Self.ElementType]], order: String)"
                    ),
                    message="Invalid order. Must be 'C' or 'F'.",
                    previous_error=None,
                )
            )
        for row in data:
            if len(row) != self.shape[1]:
                raise Error(
                    ValueError(
                        file="src/matmojo/types/matrix.mojo",
                        function=(
                            "Matrix.__init__(out self, var data:"
                            " List[List[Self.ElementType]], order: String)"
                        ),
                        message="All rows must have the same length.",
                        previous_error=None,
                    )
                )
        if order == "C":
            self.data = List[Self.ElementType](capacity=self.size)
            for row in data:
                for element in row:
                    self.data.append(element)
        else:  # order == "F"
            self.data = List[Self.ElementType](unsafe_uninit_length=self.size)
            var row_index = 0
            for row in data:
                var col_index = 0
                for element in row:
                    self.data[
                        get_offset((row_index, col_index), self.strides)
                    ] = element
                    col_index += 1
                row_index += 1

    fn __init__(
        out self,
        data: List[Self.ElementType],
        shape: Tuple[Int, Int],
        order: String = "C",
    ) raises:
        """Initializes the matrix with a list and shape.

        Args:
            data: A list of elements to initialize the matrix.
            shape: A tuple specifying the shape of the matrix (rows, cols).
            order: The memory layout order, either "C" for row-major or
                "F" for column-major. Defaults to "C".

        Raises:
            ValueError: If the length of the data list does not match the
            product of the shape dimensions.
        """
        if len(data) == 0:
            raise Error(
                ValueError(
                    file="src/matmojo/types/matrix.mojo",
                    function=(
                        "Matrix.__init__(out self, var data:"
                        " List[Self.ElementType], shape: Tuple[Int, Int],"
                        " order: String)"
                    ),
                    message="Data cannot be an empty list.",
                    previous_error=None,
                )
            )
        self.data = data.copy()
        self.size = shape[0] * shape[1]
        self.shape = shape
        if order == "C":
            self.strides = (shape[1], 1)  # Row-major order
        elif order == "F":
            self.strides = (1, shape[0])  # Column-major order
        else:
            raise Error(
                ValueError(
                    file="src/matmojo/types/matrix.mojo",
                    function=(
                        "Matrix.__init__(out self, var data:"
                        " List[Self.ElementType], shape: Tuple[Int, Int],"
                        " order: String)"
                    ),
                    message="Invalid order. Must be 'C' or 'F'.",
                    previous_error=None,
                )
            )
        if len(self.data) != self.size:
            raise Error(
                ValueError(
                    file="src/matmojo/types/matrix.mojo",
                    function=(
                        "Matrix.__init__(out self, var data:"
                        " List[Self.ElementType], shape: Tuple[Int, Int],"
                        " order: String)"
                    ),
                    message="Data length does not match the specified shape.",
                    previous_error=None,
                )
            )

    fn __copyinit__(out self, other: Self):
        """Initializes the matrix by copying another matrix."""
        self.data = other.data.copy()
        self.shape = other.shape
        self.strides = other.strides
        self.size = other.size

    fn __moveinit__(out self, deinit other: Self):
        """Initializes the matrix by moving another matrix."""
        self.data = other.data^
        self.shape = other.shape^
        self.strides = other.strides^
        self.size = other.size

    # ===--------------------------------------------------------------------===#
    # Element Access and Mutation
    # View Access
    # ===--------------------------------------------------------------------===#

    # [Mojo Miji]
    # This method returns a reference to the element at the specified indices.
    # The mutability of the reference is determined by the mutability of the
    # underlying data (self.data). Since self.data is a mutable list, the
    # reference returned by __getitem__ is mutable, allowing for both reading
    # and writing to the matrix elements.
    # Thus, `__setitem__` is not needed as a separate method.
    fn __getitem__(
        ref self, x: Int, y: Int
    ) raises -> ref[self.data] Self.ElementType:
        """Gets the element at the specified indices.

        Args:
            x: The row index.
            y: The column index.

        Raises:
            IndexError: If the indices are out of bounds.

        Returns:
            The element at the specified indices.
        """
        if x < 0 or x >= self.shape[0] or y < 0 or y >= self.shape[1]:
            raise Error(
                IndexError(
                    file="src/matmojo/types/matrix.mojo",
                    function=(
                        "Matrix.__getitem__(self, x: Int, y: Int) ->"
                        " Self.ElementType"
                    ),
                    message="Index out of bounds.",
                    previous_error=None,
                )
            )
        return self.data[x * self.strides[0] + y * self.strides[1]]

    # [Mojo Miji]
    # When you pass `Self.dtype` and `origin_of(self)` as parameters to the
    # `MatrixView` type, you are creating a new, specific instantiation of the
    # generic `MatrixView` type that is tailored to the certain data type and
    # the origin of the current matrix instance.
    # In another word, if you have a matrix of type `int64` and is called `a`,
    # then this method will create a specific `MatrixView_int64_origin_a` type
    # at compile time, and then return an instance of this type.
    # Mojo compiler will ensure that `a` will not be destroyed as long as the
    # matrix view is still alive.
    # The approach of recording the origin, which is `a`, into the parameter of
    # the `MatrixView` type is called "parameterized origin".
    fn __getitem__(
        self, x: Slice, y: Slice
    ) raises -> MatrixView[dtype = Self.dtype, origin = origin_of(self)]:
        """Gets a view of the specified row with a slice of columns."""
        return MatrixView(
            src=Pointer(to=self),
            slice_x=x,
            slice_y=y,
            initial_shape=self.shape,
            initial_strides=self.strides,
            initial_offset=0,
        )

    fn view(self) -> MatrixView[Self.dtype, origin_of(self)]:
        """Gets a view of the entire matrix."""
        return MatrixView(
            src=Pointer(to=self),
            slice_x=Slice(0, self.shape[0], 1),
            slice_y=Slice(0, self.shape[1], 1),
            initial_shape=self.shape,
            initial_strides=self.strides,
            initial_offset=0,
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
                    String(self.data[get_offset((i, j), self.strides)]) + "\t"
                )
            if i < self.shape[0] - 1:
                result += "\n"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        """Writes the matrix to a writer."""
        writer.write("Matrix, ")
        writer.write(self.dtype)
        writer.write(", ")
        writer.write(self.shape[0])
        writer.write("x")
        writer.write(self.shape[1])
        writer.write(", strides: ")
        writer.write(self.strides[0])
        writer.write("-")
        writer.write(self.strides[1])
        writer.write(":\n")
        for i in range(self.shape[0]):
            if i == 0:
                writer.write("[[\t")
            else:
                writer.write(" [\t")
            for j in range(self.shape[1]):
                writer.write(self.data[get_offset((i, j), self.strides)])
                writer.write("\t")
            writer.write("]")
            if i < self.shape[0] - 1:
                writer.write("\n")
            else:
                writer.write("]\n")
