from types import SimpleNamespace

import llaisys


def make_test_model():
    model = object.__new__(llaisys.models.Qwen2)
    model._model = None
    model._used = False
    model._end_token = 2
    model._meta = SimpleNamespace(voc=8)
    return model


def expect_error(tokens, expected_error, expected_message):
    model = make_test_model()

    try:
        model.generate(tokens, max_new_tokens=0)
    except expected_error as error:
        assert expected_message in str(error), f"Unexpected error message: {error}"
    else:
        raise AssertionError(f"Expected {expected_error.__name__}")


def test_negative_token():
    expect_error(
        [-1],
        ValueError,
        "Invalid Qwen2 token id at position 0: -1; expected [0, 8)",
    )


def test_token_above_vocabulary():
    expect_error(
        [1, 8],
        ValueError,
        "Invalid Qwen2 token id at position 1: 8; expected [0, 8)",
    )


def test_non_integer_token():
    expect_error(
        [1.5],
        TypeError,
        "Qwen2 token id at position 0 must be an integer, got float",
    )


def test_boolean_token():
    expect_error(
        [True],
        TypeError,
        "Qwen2 token id at position 0 must be an integer, got bool",
    )


def test_valid_no_op_generation():
    model = make_test_model()
    result = model.generate([0, 7], max_new_tokens=0)

    assert result == [0, 7]
    assert model._used is False


if __name__ == "__main__":
    test_negative_token()
    test_token_above_vocabulary()
    test_non_integer_token()
    test_boolean_token()
    test_valid_no_op_generation()

    print("\n\033[92mQwen2 input validation tests passed!\033[0m\n")