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

    fn __getitem__(self, indices: Tuple[Int, Int]) -> Scalar[Self.dtype]:
        """Accesses an element of the matrix view using row and column indices.
        """
        var index = get_offset(indices, self.strides, self.offset)
        return self.src[].data[index]

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
