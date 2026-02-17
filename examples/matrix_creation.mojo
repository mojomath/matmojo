import matmojo as mm


fn main() raises:
    var mat1 = mm.routines.creation.matrix(
        [
            [1.1, 1.2, 1.3, 1.4],
            [2.1, 2.2, 2.3, 2.4],
            [3.1, 3.2, 3.3, 3.4],
            [4.1, 4.2, 4.3, 4.4],
            [5.1, 5.2, 5.3, 5.4],
        ],
        order="C",
    )
    print("Memory layout (row-major):\n", mat1)

    var mat2 = mm.routines.creation.matrix(
        [
            [1.1, 1.2, 1.3, 1.4],
            [2.1, 2.2, 2.3, 2.4],
            [3.1, 3.2, 3.3, 3.4],
            [4.1, 4.2, 4.3, 4.4],
            [5.1, 5.2, 5.3, 5.4],
        ],
        order="F",
    )
    print("Memory layout (column-major):\n", mat2)

    # var mat3 = Matrix[int64](
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    #     shape=(3, 4),
    #     order="C",
    # )
    # print("Matrix (row-major with shape):\n", mat3)

    # var mat4 = Matrix[int64](
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    #     shape=(3, 4),
    #     order="F",
    # )
    # print("Matrix (column-major with shape):\n", mat4)
