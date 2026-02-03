from __future__ import annotations

# 一些基础的 Python 类型和路径工具
from typing import Sequence, Dict
from pathlib import Path
import json

# 数值计算相关
import numpy as np
import ctypes
import safetensors

# 从 llaisys 的 Python 绑定中导入需要的接口
from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    LlaisysQwen2Meta,
)


def _as_bf16_storage(arr: np.ndarray) -> np.ndarray:
    """
    将 numpy 数组转换为 BF16 的底层存储格式（uint16）。

    这里不直接用 numpy 的 bfloat16，是因为某些 numpy 版本不支持，
    所以手动把 float32 转成 BF16 的 bit 表示。
    """
    if arr.dtype == np.uint16:
        # 已经是 BF16 存储格式
        return np.ascontiguousarray(arr)

    if arr.dtype.name == "bfloat16":
        # numpy 原生 bfloat16，直接 reinterpret 成 uint16
        return np.ascontiguousarray(arr.view(np.uint16))

    if arr.dtype == np.float16:
        # 先转成 float32 再处理
        arr = arr.astype(np.float32)

    if arr.dtype == np.float32 or arr.dtype == np.float64:
        # BF16 = float32 的高 16 位
        f32 = arr.astype(np.float32, copy=False)
        u32 = f32.view(np.uint32)
        bf16 = (u32 >> 16).astype(np.uint16)
        return np.ascontiguousarray(bf16)

    # 兜底处理
    f32 = arr.astype(np.float32)
    u32 = f32.view(np.uint32)
    bf16 = (u32 >> 16).astype(np.uint16)
    return np.ascontiguousarray(bf16)


def _as_f16(arr: np.ndarray) -> np.ndarray:
    """
    转换为 float16（目前基本没用到，留作备用）
    """
    if arr.dtype == np.float16:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(arr.astype(np.float16))


class Qwen2:
    """
    使用 LLAISYS 后端实现的 Qwen2 推理模型（仅支持 CPU）。

    本实现只支持 greedy / argmax 解码，
    用于课程作业，不考虑性能和完整功能。
    """

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # 本作业只要求 CPU
        if device != DeviceType.CPU:
            raise NotImplementedError("This implementation only supports CPU.")

        model_path = Path(model_path)

        # 读取模型配置文件
        cfg_path = model_path / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"config.json not found under: {model_path}")

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        # 从 config.json 中读取模型结构参数
        hs = int(cfg.get("hidden_size"))
        nlayer = int(cfg.get("num_hidden_layers"))
        nh = int(cfg.get("num_attention_heads"))
        nkvh = int(cfg.get("num_key_value_heads", cfg.get("num_kv_heads", nh)))
        di = int(cfg.get("intermediate_size"))
        maxseq = int(cfg.get("max_position_embeddings", cfg.get("seq_length", 4096)))
        voc = int(cfg.get("vocab_size"))
        eps = float(cfg.get("rms_norm_eps", 1e-6))
        theta = float(cfg.get("rope_theta", 10000.0))

        # 处理结束 token
        eos = cfg.get("eos_token_id", cfg.get("eos_token_ids", None))
        if isinstance(eos, list):
            end_token = int(eos[0])
        elif eos is None:
            end_token = 151643
        else:
            end_token = int(eos)

        # 本模型权重使用 BF16
        dtype = DataType.BF16

        # 构造传给 C++ 后端的 meta 结构体
        meta = LlaisysQwen2Meta()
        meta.dtype = int(dtype)
        meta.nlayer = nlayer
        meta.hs = hs
        meta.nh = nh
        meta.nkvh = nkvh
        meta.dh = hs // nh
        meta.di = di
        meta.maxseq = maxseq
        meta.voc = voc
        meta.epsilon = eps
        meta.theta = theta
        meta.end_token = end_token

        self.meta = meta
        self.end_token = end_token

        # 创建 C++ 后端模型
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            meta, int(device), None, 0
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model.")

        # 获取后端中已经分配好的权重 tensor 指针
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents

        # 帮助函数：生成每一层的参数名
        def layer(name: str, i: int) -> str:
            return f"model.layers.{i}.{name}"

        # safetensors 中的名字 -> 后端 tensor 的映射
        name_map: Dict[str, object] = {
            "model.embed_tokens.weight": self._weights.in_embed,
            "lm_head.weight": self._weights.out_embed,
            "model.norm.weight": self._weights.out_norm_w,
        }

        # 每一层的参数映射
        for i in range(nlayer):
            name_map[layer("input_layernorm.weight", i)] = self._weights.attn_norm_w[i]
            name_map[layer("self_attn.q_proj.weight", i)] = self._weights.attn_q_w[i]
            name_map[layer("self_attn.q_proj.bias", i)] = self._weights.attn_q_b[i]
            name_map[layer("self_attn.k_proj.weight", i)] = self._weights.attn_k_w[i]
            name_map[layer("self_attn.k_proj.bias", i)] = self._weights.attn_k_b[i]
            name_map[layer("self_attn.v_proj.weight", i)] = self._weights.attn_v_w[i]
            name_map[layer("self_attn.v_proj.bias", i)] = self._weights.attn_v_b[i]
            name_map[layer("self_attn.o_proj.weight", i)] = self._weights.attn_o_w[i]

            name_map[layer("post_attention_layernorm.weight", i)] = self._weights.mlp_norm_w[i]
            name_map[layer("mlp.gate_proj.weight", i)] = self._weights.mlp_gate_w[i]
            name_map[layer("mlp.up_proj.weight", i)] = self._weights.mlp_up_w[i]
            name_map[layer("mlp.down_proj.weight", i)] = self._weights.mlp_down_w[i]

        # 加载 safetensors 权重
        loaded = 0
        for file in sorted(model_path.glob("*.safetensors")):
            try:
                # 优先尝试用 numpy 读取
                with safetensors.safe_open(file, framework="numpy", device="cpu") as data_:
                    for name_ in data_.keys():
                        if name_ not in name_map:
                            continue
                        arr = data_.get_tensor(name_)
                        dst = name_map[name_]
                        buf = _as_bf16_storage(arr)
                        LIB_LLAISYS.tensorLoad(
                            dst, buf.ctypes.data_as(ctypes.c_void_p)
                        )
                        loaded += 1

            except TypeError:
                # 如果 numpy 不支持 bfloat16，就退化用 torch 读取
                import torch

                with safetensors.safe_open(file, framework="pt", device="cpu") as data_:
                    for name_ in data_.keys():
                        if name_ not in name_map:
                            continue
                        t = data_.get_tensor(name_)
                        dst = name_map[name_]

                        if t.dtype == torch.bfloat16:
                            buf = t.view(torch.uint16).contiguous().cpu().numpy()
                        else:
                            buf = t.contiguous().cpu().numpy()

                        LIB_LLAISYS.tensorLoad(
                            dst, buf.ctypes.data_as(ctypes.c_void_p)
                        )
                        loaded += 1

        # 简单检查是否加载了足够多的参数
        expected_min = 3 + nlayer * 12
        if loaded < expected_min:
            raise RuntimeError(
                f"Loaded too few tensors ({loaded}), weight names may not match."
            )

    def __del__(self):
        # 析构时释放后端模型
        try:
            if getattr(self, "_model", None):
                LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
                self._model = None
        except Exception:
            pass

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """
        文本生成接口（只实现 greedy / argmax）。

        返回完整 token 序列（包含输入 prompt）。
        """
        if max_new_tokens is None:
            max_new_tokens = 128

        prompt = list(map(int, inputs))
        if len(prompt) == 0:
            return []

        # 每次生成前重置 KV cache
        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)

        outputs = prompt.copy()

        # 先把 prompt 整体跑一遍，填充 KV cache
        arr = np.array(prompt, dtype=np.int64)
        next_tok = int(
            LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                arr.size,
            )
        )
        outputs.append(next_tok)

        # 之后每次只输入一个 token，进行增量生成
        for _ in range(max_new_tokens - 1):
            if next_tok == self.end_token:
                break
            one = np.array([next_tok], dtype=np.int64)
            next_tok = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model,
                    one.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    1,
                )
            )
            outputs.append(next_tok)

        return outputs
