from matmojo.prelude import *


fn main() raises:
    var mat1 = mm.matrix[6, 5, float64](
        [
            [1.1, 1.2, 1.3, 1.4, 1.5],
            [2.1, 2.2, 2.3, 2.4, 2.5],
            [3.1, 3.2, 3.3, 3.4, 3.5],
            [4.1, 4.2, 4.3, 4.4, 4.5],
            [5.1, 5.2, 5.3, 5.4, 5.5],
            [6.1, 6.2, 6.3, 6.4, 6.5],
        ],
    )
    print(mat1)
    print(mat1.data)

    var mat2 = mm.matrix[6, 5, float64](
        flat_list=[
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            2.1,
            2.2,
            2.3,
            2.4,
            2.5,
            3.1,
            3.2,
            3.3,
            3.4,
            3.5,
            4.1,
            4.2,
            4.3,
            4.4,
            4.5,
            5.1,
            5.2,
            5.3,
            5.4,
            5.5,
            6.1,
            6.2,
            6.3,
            6.4,
            6.5,
        ],
    )
    print(mat2)
    print(mat2.data)

    var mat3 = mm.matrix[3, 2, int64](
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ],
    )
    print(mat3)
    print(mat3.data)
