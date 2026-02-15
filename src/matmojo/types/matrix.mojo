from matmojo.types.errors import IndexError, ValueError


struct Matrix[dtype: DType = DType.float64](Stringable, Writable):
    """A 2D matrix type.
    A matrix owns its data and can write to it. The elements are stored in a
    contiguous block of memory in either row-major (C-contiguous) or
    column-major (Fortran-contiguous) order.

    Parameters:
        dtype: The data type of the matrix elements. Defaults to `DType.float64`.
    """

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
        data: List[List[Scalar[Self.dtype]]],
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
                        " List[List[Scalar[Self.dtype]]], order: String)"
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
                        " List[List[Scalar[Self.dtype]]], order: String)"
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
                            " List[List[Scalar[Self.dtype]]], order: String)"
                        ),
                        message="All rows must have the same length.",
                        previous_error=None,
                    )
                )
        if order == "C":
            self.data = List[Scalar[Self.dtype]](capacity=self.size)
            for row in data:
                for element in row:
                    self.data.append(element)
        else:  # order == "F"
            self.data = List[Scalar[Self.dtype]](unsafe_uninit_length=self.size)
            var row_index = 0
            for row in data:
                var col_index = 0
                for element in row:
                    self.data[
                        row_index * self.strides[0]
                        + col_index * self.strides[1]
                    ] = element
                    col_index += 1
                row_index += 1

    fn __init__(
        out self,
        data: List[Scalar[Self.dtype]],
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
                        " List[Scalar[Self.dtype]], shape: Tuple[Int, Int],"
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
                        " List[Scalar[Self.dtype]], shape: Tuple[Int, Int],"
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
                        " List[Scalar[Self.dtype]], shape: Tuple[Int, Int],"
                        " order: String)"
                    ),
                    message="Data length does not match the specified shape.",
                    previous_error=None,
                )
            )

    # ===--------------------------------------------------------------------===#
    # Element Access and Mutation
    # ===--------------------------------------------------------------------===#

    fn __getitem__(self, x: Int, y: Int) raises -> Scalar[Self.dtype]:
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
                        " Scalar[Self.dtype]"
                    ),
                    message="Index out of bounds.",
                    previous_error=None,
                )
            )
        return self.data[x * self.strides[0] + y * self.strides[1]]

    # ===--------------------------------------------------------------------===#
    # String Representation and Writing
    # ===--------------------------------------------------------------------===#

    fn __str__(self) -> String:
        """Returns a string representation of the matrix."""
        result = String("")
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result += (
                    String(self.data[i * self.strides[0] + j * self.strides[1]])
                    + "\t"
                )
            if i < self.shape[0] - 1:
                result += "\n"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        """Writes the matrix to a writer."""
        for i in range(self.shape[0]):
            if i == 0:
                writer.write("[[\t")
            else:
                writer.write(" [\t")
            for j in range(self.shape[1]):
                writer.write(
                    self.data[i * self.strides[0] + j * self.strides[1]]
                )
                writer.write("\t")
            writer.write("]")
            if i < self.shape[0] - 1:
                writer.write("\n")
            else:
                writer.write("]")
