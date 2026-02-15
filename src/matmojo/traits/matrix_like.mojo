trait MatrixLike:
    """A trait for types that behave like matrices, providing a common interface
    for matrix operations.
    """

    fn __str__(self) -> String:
        """Returns a string representation of the matrix-like object."""
        ...
