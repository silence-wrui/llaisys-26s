## 提交概述

本 PR 完成 LLAISYS 作业 1～4，包括张量操作、CPU 算子、Qwen2 推理，以及 NVIDIA CUDA 和摩尔线程 MUSA 两个平台的设备适配与推理验证。

## 完成内容

### 作业 1：张量

- 实现 Tensor `load`
- 实现连续性判断 `isContiguous`
- 实现连续及兼容非连续布局的 `view`
- 实现 `permute`
- 实现 `slice`
- 完成张量相关测试

### 作业 2：算子

完成以下 CPU 算子及对应测试：

- Argmax
- Embedding
- Linear
- RMS Norm
- RoPE
- Self Attention
- SwiGLU

### 作业 3：Qwen2 推理

- 实现 Qwen2 C/C++ 模型结构
- 实现模型创建、销毁、权重访问和推理 C API
- 使用 ctypes 实现 Python 包装
- 从 safetensors 加载模型权重到 LLAISYS 后端
- 实现完整 Transformer Forward
- 实现 KV Cache 及逐 Token 生成
- 使用 Argmax 生成与 PyTorch 参考结果一致的 Token

模型推理逻辑由 LLAISYS C/C++ 后端完成，Python 仅负责模型调用、权重加载和结果验证。

### 作业 4：双 GPU 平台适配

#### NVIDIA CUDA

- 增加 NVIDIA 设备类型和构建选项
- 实现 CUDA Runtime API
- 实现 Stream、设备内存和同步/异步复制
- 实现 NVIDIA CUDA 算子
- 使用 cuBLAS 加速 Linear
- 支持 Qwen2 NVIDIA 推理

#### 摩尔线程 MUSA

- 增加 MUSA 设备类型和 `--musa-gpu` 构建选项
- 实现 MUSA Runtime API
- 实现 Stream、设备内存和同步/异步复制
- 实现 Add、RMS Norm、SwiGLU、Embedding、Linear、RoPE、Self Attention、Argmax 算子
- 使用 muBLAS 加速 Linear
- 支持 Qwen2 MUSA 推理

## 支持平台及状态

| 平台 | 测试环境 | 状态 |
| --- | --- | --- |
| CPU | GitHub Actions：Windows / Ubuntu | Fork GitHub Actions 构建及作业测试通过 |
| NVIDIA | NVIDIA GeForce RTX 4090 D、CUDA 12.8 | 构建、Runtime API、CUDA 算子及模型推理验证完成 |
| 摩尔线程 MUSA | MTT S5000 80 GB、MUSA 4.3.5、Driver 3.3.5-server、torch_musa 2.7.1 | 构建、Runtime API、算子回归及 Qwen2 推理验证完成 |

## 复现步骤

### CPU / GitHub Actions

```bash
xmake f -c -m release
xmake -y
xmake install -y

python test/test_runtime.py --device cpu
python test/test_tensor.py
python test/ops/add.py --device cpu
python test/ops/argmax.py --device cpu
python test/ops/embedding.py --device cpu
python test/ops/linear.py --device cpu
python test/ops/rms_norm.py --device cpu
python test/ops/rope.py --device cpu
python test/ops/self_attention.py --device cpu
python test/ops/swiglu.py --device cpu
```

### NVIDIA CUDA

```bash
export XMAKE_ROOT=y

xmake f -c --nv-gpu=y -m release
xmake -y
xmake install -y

python test/test_runtime.py --device nvidia
python test/ops/add.py --device nvidia
python test/ops/argmax.py --device nvidia
python test/ops/embedding.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia

python test/test_infer.py \
  --device nvidia \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --prompt "Who are you?" \
  --max_steps 128 \
  --test
```

### 摩尔线程 MUSA

```bash
source .venv-musa/bin/activate

export MUSA_HOME=/usr/local/musa
export PATH="$MUSA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MUSA_HOME/lib:$MUSA_HOME/lib64:$LD_LIBRARY_PATH"
export XMAKE_ROOT=y

xmake f -c --musa-gpu=y -m release
xmake -y
xmake install -y

python test/test_runtime.py --device musa

for op in add rms_norm swiglu embedding linear rope self_attention argmax
do
  python "test/ops/$op.py" --device musa || exit 1
done

python test/test_infer.py \
  --device musa \
  --model /data/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --prompt "Who are you?" \
  --max_steps 128 \
  --test
```

## 复现结果

### GitHub Actions

- 工作流：[Build and test #5](https://github.com/silence-wrui/llaisys-26s/actions/runs/31633012650)
- 提交：[`17d8b2a`](https://github.com/silence-wrui/llaisys-26s/commit/17d8b2ac79d7a5588708027b9954c4e194bc5850)
- Windows：通过
- Ubuntu：通过
- 总耗时：13 分 33 秒

### NVIDIA

- NVIDIA 构建与安装通过。
- Runtime API 的设备查询、切换、同步、Stream、设备内存及同步/异步复制测试通过。
- NVIDIA CUDA 算子及 Qwen2 推理验证完成。

### 摩尔线程 MUSA

- MUSA 构建与安装通过。
- Runtime API 测试通过。
- Add、RMS Norm、SwiGLU、Embedding、RoPE、Self Attention、Argmax 测试通过。
- Linear 的有/无 Bias、F32/F16/BF16 快速回归通过。
- DeepSeek-R1-Distill-Qwen-1.5B 的 128 Token 推理与 PyTorch 参考结果逐 Token 一致。
- 记录的 128 Token 推理耗时：LLAISYS 约 1.71 秒，PyTorch 参考约 12.28 秒。

> 说明：上游 PR 页面当前没有独立触发检查，以上 CI 结果来自 fork 仓库相同最新提交的 GitHub Actions。

## 作业完成文档

完整实现过程、代码说明和测试截图见附件：

https://github.com/silence-wrui/llaisys-26s/blob/main/%E6%88%91%E7%9A%84%E4%BD%9C%E4%B8%9A%E5%AE%8C%E6%88%90%E6%96%87%E6%A1%A3.md
