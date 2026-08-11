import ctypes
import json

from ctypes import (
    byref,
    c_int,
    c_int64,
    c_size_t,
    c_void_p,
)

from pathlib import Path
from typing import Sequence

from safetensors import safe_open

from ..libllaisys import (
    LIB_LLAISYS,
    DataType,
    DeviceType,
    LlaisysQwen2Meta,
)


class Qwen2:

    def __init__(
        self,
        model_path,
        device: DeviceType = DeviceType.CPU,
    ):
        self._model = None
        self._used = False

        model_path = Path(model_path)

        with open(
            model_path / "config.json",
            "r",
            encoding="utf-8",
        ) as file:
            config = json.load(file)

        dtype_map = {
            "bfloat16": DataType.BF16,
            "float16": DataType.F16,
            "float32": DataType.F32,
        }

        dtype_name = config["torch_dtype"]

        if dtype_name not in dtype_map:
            raise ValueError(
                f"Unsupported model dtype: {dtype_name}"
            )

        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by "
                "num_attention_heads"
            )

        head_dim = hidden_size // num_heads

        self._end_token = int(
            config["eos_token_id"]
        )

        self._meta = LlaisysQwen2Meta(
            dtype=int(dtype_map[dtype_name]),

            nlayer=config["num_hidden_layers"],
            hs=hidden_size,
            nh=num_heads,
            nkvh=config["num_key_value_heads"],
            dh=head_dim,
            di=config["intermediate_size"],
            maxseq=config["max_position_embeddings"],
            voc=config["vocab_size"],

            epsilon=config["rms_norm_eps"],
            theta=config["rope_theta"],

            end_token=self._end_token,
        )

        device_ids = (c_int * 1)(0)

        self._model = (
            LIB_LLAISYS.llaisysQwen2ModelCreate(
                byref(self._meta),
                int(device),
                device_ids,
                1,
            )
        )

        if not self._model:
            raise RuntimeError(
                "Failed to create Qwen2 model"
            )

        weights_ptr = (
            LIB_LLAISYS.llaisysQwen2ModelWeights(
                self._model
            )
        )

        if not weights_ptr:
            raise RuntimeError(
                "Failed to get Qwen2 weights"
            )

        weights = weights_ptr.contents

        handles = self._build_weight_map(weights)
        self._load_weights(model_path, handles)

    def _build_weight_map(self, weights):
        handles = {
            "model.embed_tokens.weight":
                weights.in_embed,

            "lm_head.weight":
                weights.out_embed,

            "model.norm.weight":
                weights.out_norm_w,
        }

        for layer in range(self._meta.nlayer):
            prefix = f"model.layers.{layer}"

            handles[
                f"{prefix}.input_layernorm.weight"
            ] = weights.attn_norm_w[layer]

            handles[
                f"{prefix}.self_attn.q_proj.weight"
            ] = weights.attn_q_w[layer]

            handles[
                f"{prefix}.self_attn.q_proj.bias"
            ] = weights.attn_q_b[layer]

            handles[
                f"{prefix}.self_attn.k_proj.weight"
            ] = weights.attn_k_w[layer]

            handles[
                f"{prefix}.self_attn.k_proj.bias"
            ] = weights.attn_k_b[layer]

            handles[
                f"{prefix}.self_attn.v_proj.weight"
            ] = weights.attn_v_w[layer]

            handles[
                f"{prefix}.self_attn.v_proj.bias"
            ] = weights.attn_v_b[layer]

            handles[
                f"{prefix}.self_attn.o_proj.weight"
            ] = weights.attn_o_w[layer]

            handles[
                f"{prefix}."
                "post_attention_layernorm.weight"
            ] = weights.mlp_norm_w[layer]

            handles[
                f"{prefix}.mlp.gate_proj.weight"
            ] = weights.mlp_gate_w[layer]

            handles[
                f"{prefix}.mlp.up_proj.weight"
            ] = weights.mlp_up_w[layer]

            handles[
                f"{prefix}.mlp.down_proj.weight"
            ] = weights.mlp_down_w[layer]

        return handles

    def _load_weights(self, model_path, handles):
        expected = set(handles)
        loaded = set()

        for file in sorted(
            model_path.glob("*.safetensors")
        ):
            with safe_open(
                file,
                framework="pt",
                device="cpu",
            ) as tensors:

                for name in tensors.keys():
                    if name not in handles:
                        continue

                    tensor = tensors.get_tensor(name)

                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()

                    handle = handles[name]

                    if not handle:
                        raise RuntimeError(
                            f"Null backend handle for {name}"
                        )

                    LIB_LLAISYS.tensorLoad(
                        c_void_p(handle),
                        c_void_p(tensor.data_ptr()),
                    )

                    loaded.add(name)

                    # 不把全部 PyTorch Tensor 留在列表中。
                    del tensor

        missing = expected - loaded

        if missing:
            missing_text = "\n".join(sorted(missing))

            raise RuntimeError(
                "Missing model weights:\n"
                f"{missing_text}"
            )

        print(
            f"Loaded {len(loaded)} Qwen2 tensors"
        )

    def __del__(self):
        model = getattr(self, "_model", None)

        if model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(
                model
            )
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # 阶段 2、3 再实现。
        raise NotImplementedError(
            "Qwen2 inference is not implemented yet"
        )