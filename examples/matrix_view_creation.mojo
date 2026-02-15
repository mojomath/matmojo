from matmojo.prelude import *


fn main() raises:
    var mat1 = Matrix(
        [
            [1.1, 1.2, 1.3, 1.4],
            [2.1, 2.2, 2.3, 2.4],
            [3.1, 3.2, 3.3, 3.4],
            [4.1, 4.2, 4.3, 4.4],
            [5.1, 5.2, 5.3, 5.4],
        ],
        order="C",
    )
    print("Matrix:\n", mat1)

    var matview1 = MatrixView(
        src=Pointer(to=mat1),
        shape=(3, 3),
        strides=(4, 1),
        offset=5,
    )
    print("MatrixView:\n", matview1)
