from ctypes import POINTER, c_float, c_int64, c_void_p, cast

import llaisys


def test_invalid_embedding_index():
    index = llaisys.Tensor((1,), dtype=llaisys.DataType.I64)
    weight = llaisys.Tensor((2, 3), dtype=llaisys.DataType.F32)
    out = llaisys.Tensor((1, 3), dtype=llaisys.DataType.F32)

    index_values = (c_int64 * 1)(2)
    weight_values = (c_float * 6)(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    index.load(cast(index_values, c_void_p))
    weight.load(cast(weight_values, c_void_p))

    try:
        llaisys.Ops.embedding(out, index, weight)
    except RuntimeError as error:
        assert "Embedding index is out of range" in str(error)
    else:
        raise AssertionError("Expected invalid embedding index to raise RuntimeError")

    # 验证发生一次错误后，后续合法调用不会读到残留错误。
    valid_index = (c_int64 * 1)(1)
    index.load(cast(valid_index, c_void_p))
    llaisys.Ops.embedding(out, index, weight)

    result = cast(out.data_ptr(), POINTER(c_float))
    assert [result[i] for i in range(3)] == [4.0, 5.0, 6.0]


if __name__ == "__main__":
    test_invalid_embedding_index()
    print("\n\033[92mOperator error handling tests passed!\033[0m\n")