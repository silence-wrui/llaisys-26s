import ctypes
import json
import torch

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
            "bfloat16": (DataType.BF16, torch.bfloat16,),
            "float16": (DataType.F16, torch.float16,),
            "float32": (DataType.F32, torch.float32,),
        }

        dtype_name = config["torch_dtype"]

        if dtype_name not in dtype_map:
            raise ValueError(
                f"Unsupported model dtype: {dtype_name}"
            )

        backend_dtype, torch_dtype = dtype_map[dtype_name]
        self._torch_dtype = torch_dtype

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
            dtype=int(backend_dtype),

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
        hs = int(self._meta.hs)
        nh = int(self._meta.nh)
        nkvh = int(self._meta.nkvh)
        dh = int(self._meta.dh)
        di = int(self._meta.di)
        voc = int(self._meta.voc)
        nlayer = int(self._meta.nlayer)

        q_size = nh * dh
        kv_size = nkvh * dh

        handles = {
            "model.embed_tokens.weight": (
                weights.in_embed,
                (voc, hs),
            ),

            "lm_head.weight": (
                weights.out_embed,
                (voc, hs),
            ),

            "model.norm.weight": (
                weights.out_norm_w,
                (hs,),
            ),
        }

        for layer in range(nlayer):
            prefix = f"model.layers.{layer}"

            handles[
                f"{prefix}.input_layernorm.weight"
            ] = (weights.attn_norm_w[layer], (hs,),)

            handles[
                f"{prefix}.self_attn.q_proj.weight"
            ] = (weights.attn_q_w[layer], (q_size, hs),)

            handles[
                f"{prefix}.self_attn.q_proj.bias"
            ] = (weights.attn_q_b[layer], (q_size,),)

            handles[
                f"{prefix}.self_attn.k_proj.weight"
            ] = (weights.attn_k_w[layer], (kv_size, hs),)

            handles[
                f"{prefix}.self_attn.k_proj.bias"
            ] = (weights.attn_k_b[layer], (kv_size,),)

            handles[
                f"{prefix}.self_attn.v_proj.weight"
            ] = (weights.attn_v_w[layer], (kv_size, hs),)

            handles[
                f"{prefix}.self_attn.v_proj.bias"
            ] = (weights.attn_v_b[layer], (kv_size,),)

            handles[
                f"{prefix}.self_attn.o_proj.weight"
            ] = (weights.attn_o_w[layer], (hs, q_size),)

            handles[
                f"{prefix}."
                "post_attention_layernorm.weight"
            ] = (weights.mlp_norm_w[layer], (hs,),)

            handles[
                f"{prefix}.mlp.gate_proj.weight"
            ] = (weights.mlp_gate_w[layer], (di, hs),)

            handles[
                f"{prefix}.mlp.up_proj.weight"
            ] = (weights.mlp_up_w[layer], (di, hs),)

            handles[
                f"{prefix}.mlp.down_proj.weight"
            ] = (weights.mlp_down_w[layer], (hs, di),)

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

                    handle, expected_shape = handles[name]

                    actual_shape = tuple(tensor.shape)

                    if actual_shape != expected_shape:
                        raise ValueError(
                            f"Weight shape mismatch for {name}: "
                            f"expected {expected_shape}, "
                            f"got {actual_shape}"
                        )

                    if tensor.dtype != self._torch_dtype:
                        raise TypeError(
                            f"Weight dtype mismatch for {name}: "
                            f"expected {self._torch_dtype}, "
                            f"got {tensor.dtype}"
                        )

                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()

                    if not handle:
                        raise RuntimeError(
                            f"Null backend handle for {name}"
                        )

                    LIB_LLAISYS.tensorLoad(
                        c_void_p(handle),
                        c_void_p(tensor.data_ptr()),
                    )

                    loaded.add(name)

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
        if self._used:
            raise RuntimeError(
                "This Qwen2 model instance has already "
                "been used for generation"
            )

        if not inputs:
            raise ValueError(
                "Qwen2 input tokens must not be empty"
            )

        if max_new_tokens is None:
            max_new_tokens = 128

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens must not be negative"
            )

        # 当前作业要求 argmax，因此暂时不使用采样参数。
        _ = top_k, top_p, temperature

        self._used = True

        output_tokens = [
            int(token) for token in inputs
        ]

        if max_new_tokens == 0:
            return output_tokens

        # 第一次必须传入完整提示词。
        current_input = output_tokens.copy()

        for _ in range(max_new_tokens):
            token_array = (
                c_int64 * len(current_input)
            )(*current_input)

            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model,
                    token_array,
                    len(current_input),
                )
            )

            output_tokens.append(next_token)

            if next_token == self._end_token:
                break

            # KV Cache 已保存此前内容，
            # 下一轮只输入刚生成的一个 token。
            current_input = [next_token]

        return output_tokens