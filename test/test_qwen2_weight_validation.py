import json
import tempfile

from pathlib import Path

import torch

from safetensors.torch import save_file

import llaisys


def write_config(model_path: Path) -> None:
    config = {
        "torch_dtype": "float32",
        "hidden_size": 4,
        "num_attention_heads": 1,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "intermediate_size": 8,
        "max_position_embeddings": 16,
        "vocab_size": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "eos_token_id": 2,
    }

    with open(
        model_path / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file)


def expect_weight_error(
    weight: torch.Tensor,
    expected_error,
    expected_message: str,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir)

        write_config(model_path)

        save_file(
            {
                "model.embed_tokens.weight":
                    weight,
            },
            str(model_path / "model.safetensors"),
        )

        try:
            llaisys.models.Qwen2(
                model_path,
                llaisys.DeviceType.CPU,
            )
        except expected_error as error:
            message = str(error)

            assert expected_message in message, (
                f"Unexpected error message: {message}"
            )
        else:
            raise AssertionError(
                f"Expected {expected_error.__name__}"
            )


def test_wrong_shape() -> None:
    # config 中预期形状是 (vocab_size, hidden_size)
    # 即 (8, 4)，这里故意提供 (7, 4)。
    weight = torch.zeros(
        (7, 4),
        dtype=torch.float32,
    )

    expect_weight_error(
        weight,
        ValueError,
        (
            "Weight shape mismatch for "
            "model.embed_tokens.weight: "
            "expected (8, 4), got (7, 4)"
        ),
    )


def test_wrong_dtype() -> None:
    # 形状正确，但 config 声明 float32，
    # 这里故意提供 float16。
    weight = torch.zeros(
        (8, 4),
        dtype=torch.float16,
    )

    expect_weight_error(
        weight,
        TypeError,
        (
            "Weight dtype mismatch for "
            "model.embed_tokens.weight: "
            "expected torch.float32, "
            "got torch.float16"
        ),
    )


if __name__ == "__main__":
    test_wrong_shape()
    test_wrong_dtype()

    print(
        "\n"
        "\033[92m"
        "Qwen2 weight validation tests passed!"
        "\033[0m"
        "\n"
    )